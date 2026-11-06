"""Non-learned retrieval baselines on the same graphs: text-kNN over verbalized triples, and PCST
(PyG G-Retriever `retrieval_via_pcst`, prizes from the same frozen MiniLM embeddings)."""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from bridge.retrieve import TripleIndex, n_tokens


class KNNBaseline:
    def __init__(self, ig):
        self.ig = ig
        self.idx = TripleIndex(ig)
        self.N = ig.g["x"].size(0)
        self.x = ig.t["x"].float()

    def rank(self, q):
        """q: [384] normalized -> (edge order, node order) by cosine over verbalized triples."""
        es = self.idx.emb @ q
        eorder = es.argsort(descending=True)
        ei = self.idx.ei.to(es.device)
        # node score = max cosine over incident triples (first appearance in the ranked triple list)
        ns = torch.full((self.N,), -2.0, device=es.device)
        ns.scatter_reduce_(0, ei[0], es, reduce="amax")
        ns.scatter_reduce_(0, ei[1], es, reduce="amax")
        return eorder, ns.argsort(descending=True), es

    def context_tokens(self, eorder, budget=2000):
        used, n = 0, 0
        for e in eorder.tolist():
            t = n_tokens(self.idx.lines[e]) + 1
            if used + t > budget:
                break
            used += t
            n += 1
        return used, n


class PCSTBaseline:
    def __init__(self, ig, knn: KNNBaseline, topk=20, topk_e=20, cost_e=0.5):
        from torch_geometric.datasets.web_qsp_dataset import retrieval_via_pcst
        self.f = retrieval_via_pcst
        self.ig, self.knn = ig, knn
        self.topk, self.topk_e, self.cost_e = topk, topk_e, cost_e
        E = ig.g["n_fwd_edges"]
        self.data = Data(x=ig.g["x"].float(), edge_index=ig.g["edge_index"][:, :E],
                         edge_attr=knn.idx.emb.float().cpu())
        self.data.node_idx = np.arange(self.data.num_nodes)
        self.data.edge_idx = np.arange(E)
        self.tn = pd.DataFrame({"node_id": np.arange(self.data.num_nodes), "node_attr": [""] * self.data.num_nodes})
        ei = self.data.edge_index
        self.te = pd.DataFrame({"src": ei[0].numpy(), "edge_attr": [""] * E, "dst": ei[1].numpy()})

    def rank(self, q):
        sub, _ = self.f(self.data, q.cpu(), self.tn, self.te, topk=self.topk, topk_e=self.topk_e, cost_e=self.cost_e)
        eorder_knn, norder_knn, es = self.knn.rank(q)
        sel_e = set(np.asarray(getattr(sub, "edge_idx", [])).tolist())
        sel_n = set(np.asarray(getattr(sub, "node_idx", [])).tolist())
        eo = [e for e in eorder_knn.tolist() if e in sel_e] + [e for e in eorder_knn.tolist() if e not in sel_e]
        no = [n for n in norder_knn.tolist() if n in sel_n] + [n for n in norder_knn.tolist() if n not in sel_n]
        return torch.tensor(eo), torch.tensor(no), len(sel_n), len(sel_e)
