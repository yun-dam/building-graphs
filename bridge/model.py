"""BRIDGE model: soft seeding, ULTRA-style relation GNN, NBFNet-style entity GNN, heads.

All node states are batched over questions: h has shape [B, N, d] (B questions, same building).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


def mlp(i, h, o, p=0.1):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, o))


class ConditionalConv(MessagePassing):
    """h^t(v) = LN( SUM_{(u,r,v)} h^{t-1}(u) * w_r  +  h^0(v) ) + h^{t-1}(v).

    rel_emb is an input of shape [B, R, d] (question-conditioned), so there is no
    per-relation parameter table and unseen relation types work.
    """

    def __init__(self, d, dropout=0.1):
        super().__init__(aggr="add", node_dim=1)
        self.ln = nn.LayerNorm(d)
        self.lin = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, h0, edge_index, edge_type, rel_emb):
        agg = self.propagate(edge_index, x=h, edge_type=edge_type, rel_emb=rel_emb)
        return self.drop(F.relu(self.ln(self.lin(agg) + h0))) + h

    def message(self, x_j, edge_type, rel_emb):
        return x_j * rel_emb[:, edge_type]


class RelationGNN(nn.Module):
    """Conditional message passing over the relation graph (4 meta-relations: h2h, h2t, t2h, t2t)."""

    def __init__(self, d, layers, n_entity_layers, txt_dim=384, dropout=0.1, use_graph=True):
        super().__init__()
        self.use_graph = use_graph
        self.rel_in = nn.Linear(txt_dim, d)
        self.q_in = nn.Linear(txt_dim, d)
        self.meta = nn.Parameter(torch.randn(layers, 4, d) * 0.1 + 1.0)
        self.convs = nn.ModuleList(ConditionalConv(d, dropout) for _ in range(layers))
        self.out = nn.ModuleList(nn.Linear(d, d) for _ in range(n_entity_layers))
        self.log_tau = nn.Parameter(torch.tensor(-2.3))  # tau = 0.1

    def forward(self, q, rel_x, rel_graph, rel_graph_type):
        # q [B, 384] (normalized), rel_x [R, 384] (normalized)
        sim = torch.sigmoid((q @ rel_x.t()) / self.log_tau.exp() - 2.0)          # [B, R]
        h0 = sim.unsqueeze(-1) * self.q_in(q).unsqueeze(1) + self.rel_in(rel_x).unsqueeze(0)
        h = h0
        if self.use_graph and rel_graph.numel():
            for t, conv in enumerate(self.convs):
                h = conv(h, h0, rel_graph, rel_graph_type, self.meta[t].unsqueeze(0).expand(q.size(0), -1, -1))
        return [o(h) for o in self.out]                                              # L x [B, R, d]


class LearnedRelations(nn.Module):
    """Ablation: per-relation embedding table keyed by relation name (no relation graph)."""

    def __init__(self, d, n_entity_layers, txt_dim=384, vocab=4096):
        super().__init__()
        self.vocab = vocab
        self.table = nn.Embedding(vocab, d)
        self.q_in = nn.Linear(txt_dim, d)
        self.out = nn.ModuleList(nn.Linear(d, d) for _ in range(n_entity_layers))

    def forward(self, q, rel_ids):
        h = self.table(rel_ids % self.vocab).unsqueeze(0) * torch.sigmoid(self.q_in(q)).unsqueeze(1)
        return [o(h) for o in self.out]


class BRIDGE(nn.Module):
    def __init__(self, d=64, rel_layers=6, ent_layers=6, txt_dim=384, top_seeds=32, dropout=0.1,
                 hard_seeding=False, relation_graph=True):
        super().__init__()
        self.d, self.top_seeds, self.hard = d, top_seeds, hard_seeding
        self.relation_graph = relation_graph
        if relation_graph:
            self.rel = RelationGNN(d, rel_layers, ent_layers, txt_dim, dropout)
        else:
            self.rel = LearnedRelations(d, ent_layers, txt_dim)
        self.W0 = nn.Linear(txt_dim, d)
        self.log_tau = nn.Parameter(torch.tensor(-2.3))  # tau init 0.1
        self.bias = nn.Parameter(torch.tensor(-2.0))
        self.convs = nn.ModuleList(ConditionalConv(d, dropout) for _ in range(ent_layers))
        self.xproj = nn.Linear(txt_dim, d)
        self.qproj = nn.Linear(txt_dim, d)
        self.node_head = mlp(3 * d, 2 * d, 1, dropout)
        self.ans_head = mlp(3 * d, 2 * d, 1, dropout)
        self.edge_mlp = mlp(3 * d, 2 * d, d, dropout)
        self.edge_out = nn.Linear(d, 1)
        self.path_q = nn.Linear(txt_dim, d)

    # ------------------------------------------------------------------ seeding
    def seed(self, q, x):
        cos = q @ x.t()                                                    # [B, N]
        a = torch.sigmoid(cos / self.log_tau.exp() + self.bias)
        k = min(self.top_seeds, x.size(0))
        top = cos.topk(k, dim=1).indices
        mask = torch.zeros_like(a).scatter_(1, top, 1.0)
        a = mask if self.hard else a * mask
        return a, cos

    def forward(self, g, q):
        """g: dict with x [N,384], edge_index, edge_type, rel_x, rel_graph, rel_graph_type, rel_ids."""
        x = g["x"].float()
        B = q.size(0)
        a, cos = self.seed(q, x)
        h0 = a.unsqueeze(-1) * self.W0(q).unsqueeze(1)                       # [B, N, d]
        if self.relation_graph:
            rels = self.rel(q, g["rel_x"].float(), g["rel_graph"], g["rel_graph_type"])
        else:
            rels = self.rel(q, g["rel_ids"])
        h = h0
        for t, conv in enumerate(self.convs):
            h = conv(h, h0, g["edge_index"], g["edge_type"], rels[t])
        xq = self.xproj(x).unsqueeze(0).expand(B, -1, -1)
        qq = self.qproj(q).unsqueeze(1).expand(-1, x.size(0), -1)
        z = torch.cat([h, xq, qq], -1)
        return {"h": h, "w": rels[-1], "node_logit": self.node_head(z).squeeze(-1),
                "ans_logit": self.ans_head(z).squeeze(-1), "seed": a, "cos": cos}

    def edge_repr(self, out, b, eidx, g):
        """Edge representations for question b and edge ids eidx -> ([n, d], logits [n])."""
        s, t = g["edge_index"][0, eidx], g["edge_index"][1, eidx]
        r = g["edge_type"][eidx]
        h, w = out["h"][b], out["w"][b]
        rep = self.edge_mlp(torch.cat([h[s], w[r], h[t]], -1))
        return rep, self.edge_out(F.relu(rep)).squeeze(-1)
