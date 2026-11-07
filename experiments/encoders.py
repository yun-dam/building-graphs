"""Encode-then-match GNN baselines with BRIDGE's exact heads, so only the encoder differs.

BRIDGE runs *question-conditioned* message passing (NBFNet-style: the question is injected at
seed nodes and every node state is relative to it, shape [B, N, d]). These baselines run a
*question-agnostic* encoder once per graph (shape [N, d]) and only meet the question in the
heads. The comparison therefore isolates conditioning (the labeling-trick argument in
proposal §3.2) from heads, losses, data and training budget.

Encoders (all inductive: no node-ID or relation-ID parameters):
  compgcn  CompGCN (Vashishth et al., ICLR 2020), ported from Vaishnavi's bridge-cs224w repo.
           h_v' = h_v + drop(GELU(LN( mean_{u->v fwd} W_O phi(h_u, z_r)
                                         + mean_{u->v inv} W_I phi(h_u, z_r)
                                         + W_S phi(h_v, z_self) ) / 3)),   z_r' = W_rel z_r
           phi = mult | sub; z_r^0 = Linear(text(r)). The graph already stores an explicit
           inverse edge (type 2k+1) for every forward edge (type 2k), so direction is the
           edge-type parity rather than a separate pass over reversed edges.
  gcn      Same layer with phi(h, z) = h: relation types ignored, direction kept (W_O / W_I).
  mlp      No message passing (0 layers): node text + node kind only.

Heads are copied from bridge.model.BRIDGE (node, answer, edge MLP, path_q), so bridge.train's
batch_loss / evaluate / path_logits and bridge.eval_retriever's eval_bridge work unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from bridge.model import mlp


def scatter_mean(src, index, n):
    out = torch.zeros(n, src.size(1), dtype=src.dtype, device=src.device).index_add_(0, index, src)
    cnt = torch.zeros(n, dtype=src.dtype, device=src.device).index_add_(
        0, index, torch.ones_like(index, dtype=src.dtype))
    return out / cnt.clamp(min=1).unsqueeze(1)


class CompGCNLayer(nn.Module):
    def __init__(self, d, comp="mult", use_rel=True, dropout=0.1):
        super().__init__()
        self.comp, self.use_rel = comp, use_rel
        self.w_out = nn.Linear(d, d, bias=False)
        self.w_in = nn.Linear(d, d, bias=False)
        self.w_self = nn.Linear(d, d, bias=False)
        self.w_rel = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def phi(self, h, z):
        if not self.use_rel:
            return h
        return h * z if self.comp == "mult" else h - z

    def forward(self, h, z, z_self, src, dst, rel, fwd):
        n = h.size(0)
        msg = self.phi(h[src], z[rel])
        m_out = scatter_mean(self.w_out(msg[fwd]), dst[fwd], n)
        m_in = scatter_mean(self.w_in(msg[~fwd]), dst[~fwd], n)
        m_self = self.w_self(self.phi(h, z_self.expand_as(h)))
        h = self.drop(F.gelu(self.norm((m_out + m_in + m_self) / 3))) + h
        return h, self.w_rel(z), self.w_rel(z_self)


class EncodeThenMatch(nn.Module):
    def __init__(self, encoder="compgcn", d=64, layers=2, comp="mult", txt_dim=384, dropout=0.1):
        super().__init__()
        assert encoder in ("compgcn", "gcn", "mlp")
        self.encoder, self.d = encoder, d
        self.use_rel = encoder == "compgcn"
        n_layers = 0 if encoder == "mlp" else layers
        self.node_in = nn.Linear(txt_dim, d)
        self.kind_emb = nn.Embedding(2, d)  # instance / class
        self.rel_in = nn.Linear(txt_dim, d)
        self.z_self = nn.Parameter(torch.randn(1, d) * 0.1 + 1.0)
        self.layers = nn.ModuleList(CompGCNLayer(d, comp, self.use_rel, dropout) for _ in range(n_layers))
        # --- heads: identical to bridge.model.BRIDGE
        self.xproj = nn.Linear(txt_dim, d)
        self.qproj = nn.Linear(txt_dim, d)
        self.node_head = mlp(3 * d, 2 * d, 1, dropout)
        self.ans_head = mlp(3 * d, 2 * d, 1, dropout)
        self.edge_mlp = mlp(3 * d, 2 * d, d, dropout)
        self.edge_out = nn.Linear(d, 1)
        self.path_q = nn.Linear(txt_dim, d)

    def encode(self, g):
        x = g["x"].float()
        kind = g["is_class"].long() if "is_class" in g else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = self.node_in(x) + self.kind_emb(kind)
        z = self.rel_in(g["rel_x"].float())
        z_self = self.z_self
        src, dst = g["edge_index"]
        rel = g["edge_type"]
        fwd = (rel % 2) == 0
        for layer in self.layers:
            h, z, z_self = layer(h, z, z_self, src, dst, rel, fwd)
        if not self.use_rel:
            z = torch.zeros_like(z)
        return h, z

    def forward(self, g, q):
        x = g["x"].float()
        B, N = q.size(0), x.size(0)
        h, z = self.encode(g)
        h = h.unsqueeze(0).expand(B, -1, -1)
        w = z.unsqueeze(0).expand(B, -1, -1)
        xq = self.xproj(x).unsqueeze(0).expand(B, -1, -1)
        qq = self.qproj(q).unsqueeze(1).expand(-1, N, -1)
        feat = torch.cat([h, xq, qq], -1)
        return {"h": h, "w": w, "node_logit": self.node_head(feat).squeeze(-1),
                "ans_logit": self.ans_head(feat).squeeze(-1)}

    def edge_repr(self, out, b, eidx, g):
        s, t = g["edge_index"][0, eidx], g["edge_index"][1, eidx]
        r = g["edge_type"][eidx]
        h, w = out["h"][b], out["w"][b]
        rep = self.edge_mlp(torch.cat([h[s], w[r], h[t]], -1))
        return rep, self.edge_out(F.relu(rep)).squeeze(-1)


def build(cfg):
    return EncodeThenMatch(encoder=cfg["encoder"], d=cfg["d"], layers=cfg["layers"], comp=cfg["comp"],
                           dropout=cfg["dropout"])


def load(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = build(ck["config"]).to(dev)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck["config"]
