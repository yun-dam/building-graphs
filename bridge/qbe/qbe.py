"""BRIDGE-QBE: the GNN finds example answer tuples; the SPARQL is induced from them (query-by-example).

No LLM is involved. Steps:
 1. v2 GNN scores every node (answer head) -> nodes grouped into candidate COLUMNS:
    entity columns keyed by class, attribute columns keyed by (owner key, literal predicate),
    untyped-node columns keyed by (parent class, predicate).
 2. Anchor = best entity column. From its top nodes, a BFS (<= max_hops) over the graph restricted to
    the GNN's most relevant nodes finds, for every other column, the best-scored reachable member.
 3. Each anchor->column path is lifted to a class-level pattern; the majority pattern over anchors is used.
 4. Patterns are joined into one star-shaped SPARQL query around the anchor and executed; if it returns
    nothing, the weakest column is dropped and the query rebuilt.
"""
import collections
import threading

import torch

from bridge import kg
from bridge.retrieve import bridge_scores, embed, var_name

RDF_TYPE, SUBCLASS = kg.RDF_TYPE, kg.SUBCLASS


class QBEGraph:
    def __init__(self, ig):
        self.ig = ig
        g = ig.g
        self.g = g
        ei, et = g["edge_index"], g["edge_type"]
        names = g["rel_names"]
        self.names = names
        skip = {i for i, n in enumerate(names) if n.lstrip("^") in (RDF_TYPE, SUBCLASS)}
        self.adj = collections.defaultdict(list)
        for s, o, r in zip(ei[0].tolist(), ei[1].tolist(), et.tolist()):
            if r not in skip:
                self.adj[s].append((o, r))
        self.pt = g["primary_type"].tolist()
        self.is_attr = g["is_attr"].tolist()
        self.is_class = g["is_class"].tolist()
        self.n_inst = g["n_inst"]
        self._key = {}

    def parent(self, v):
        """An incoming forward edge (u, r) into v from a typed node, if any."""
        for u, r in self.adj[v]:
            if self.names[r].startswith("^") and self.pt[u] >= 0:
                return u, r ^ 1  # forward relation id u -r-> v
        return None

    def key(self, v):
        if v in self._key:
            return self._key[v]
        if self.is_attr[v]:
            par = None
            for u, r in self.adj[v]:
                if self.names[r].startswith("^"):
                    par = (u, r ^ 1)
                    break
            k = ("A", self.key(par[0]) if par else None, par[1] if par else None)
        elif self.pt[v] >= 0:
            k = ("E", self.pt[v])
        else:
            p = self.parent(v)
            k = ("U", self.pt[p[0]], p[1]) if p else ("U", None, None)
        self._key[v] = k
        return k


def _bfs(qg, src, targets, allowed, max_hops):
    """Shortest paths from src to any node in targets (restricted to allowed); returns {target: path}."""
    prev = {src: None}
    frontier = [src]
    found = {}
    for _ in range(max_hops):
        nxt = []
        for u in frontier:
            for v, r in qg.adj[u]:
                if v in prev or (v not in allowed and v not in targets):
                    continue
                if qg.is_class[v]:
                    continue
                prev[v] = (u, r)
                if v in targets:
                    found[v] = True
                elif not qg.is_attr[v]:
                    nxt.append(v)
        if found:
            break
        frontier = nxt
    out = {}
    for t in found:
        path, v = [], t
        while prev[v] is not None:
            u, r = prev[v]
            path.append((r, v))
            v = u
        out[t] = list(reversed(path))
    return out


def _pattern(qg, path):
    """Class-level signature of an instance path: ((rel, node_class_or_None, is_attr), ...)."""
    return tuple((r, qg.pt[v] if qg.pt[v] >= 0 else None, bool(qg.is_attr[v])) for r, v in path)


def _var(base, used):
    base = "".join(ch for ch in base if ch.isalnum() or ch == "_")[:28] or "x"
    used[base] += 1
    return base if used[base] == 1 else f"{base}{used[base]}"


def build_query(qg, anchor_cls, col_patterns):
    used = collections.Counter()
    g = qg.g
    cname = lambda c: var_name(g["node_ids"][c])[1:]
    a = _var(cname(anchor_cls), used)
    pats = [f"?{a} <{RDF_TYPE}> <{g['node_ids'][anchor_cls]}> ."]
    proj = [a]
    for pat in col_patterns:
        cur = a
        for i, (r, c, is_attr) in enumerate(pat):
            last = i == len(pat) - 1
            rel = qg.names[r]
            if last:
                if is_attr:
                    nv = _var(f"{cur}_{kg.local_name(rel.lstrip('^'))}", used)
                elif c is not None:
                    nv = _var(cname(c), used)
                else:
                    nv = _var(f"{cur}_{kg.local_name(rel.lstrip('^'))}", used)
            else:
                nv = _var(f"n{len(used)}", used)
            if rel.startswith("^"):
                pats.append(f"?{nv} <{rel[1:]}> ?{cur} .")
            else:
                pats.append(f"?{cur} <{rel}> ?{nv} .")
            if c is not None and not is_attr:
                pats.append(f"?{nv} <{RDF_TYPE}> <{g['node_ids'][c]}> .")
            cur = nv
        proj.append(cur)
    return f"SELECT DISTINCT {' '.join('?' + p for p in proj)} WHERE {{\n  " + "\n  ".join(pats) + "\n}"


def run_query(store, sparql, timeout=60, cap=20000):
    box = {}

    def go():
        try:
            sols = store.query(sparql, use_default_graph_as_union=True)
            names = [v.value for v in sols.variables]
            rows = []
            for i, sol in enumerate(sols):
                if i >= cap:
                    break
                d = {n: {"value": kg.term_str(sol[n])} for n in names if sol[n] is not None}
                if d:
                    rows.append(d)
            box["rows"] = rows
        except Exception as e:  # noqa: BLE001
            box["err"] = str(e)[:200]

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, "timeout"
    return box.get("rows"), box.get("err")


@torch.no_grad()
def qbe(model, qg, question, store, tau=0.5, max_cols=6, n_anchor=3, max_hops=4, n_allowed=3000, top_nodes=400,
        rho=None, shortest=False):
    ig = qg.ig
    q = embed([question]).to(ig.dev)
    out = bridge_scores(model, ig, q)
    ans = torch.sigmoid(out["ans_logit"][0]).cpu()
    node = out["node_logit"][0].cpu()
    n = qg.n_inst
    ans_i = ans[:n]
    cand = ans_i.topk(min(top_nodes, n)).indices.tolist()
    groups = collections.OrderedDict()
    for v in cand:
        groups.setdefault(qg.key(v), []).append(v)
    gscore = {k: float(ans[vs[0]]) for k, vs in groups.items()}
    ranked = sorted(groups, key=lambda k: -gscore[k])
    if rho is not None:  # relative cut: keep columns scoring >= rho * best column score
        sel = [k for k in ranked if gscore[k] >= max(tau, rho * gscore[ranked[0]])][:max_cols] or ranked[:1]
    else:
        sel = [k for k in ranked if gscore[k] >= tau][:max_cols] or ranked[:1]
    ent = [k for k in sel if k[0] == "E"] or [k for k in ranked if k[0] == "E"][:1]
    if not ent:
        return None
    anchor_key = ent[0]
    others = [k for k in sel if k != anchor_key]
    allowed = set(node[:n].topk(min(n_allowed, n)).indices.tolist())
    anchors = groups[anchor_key][:n_anchor]
    col_pats = []
    for k in others:
        targets = set(groups[k][:50])
        votes, depth = collections.Counter(), {}
        for a in anchors:
            paths = _bfs(qg, a, targets, allowed, max_hops)
            if not paths:
                continue
            best = max(paths, key=lambda t: (float(ans[t]), -len(paths[t])))
            pat = _pattern(qg, paths[best])
            votes[pat] += 1
            depth[pat] = len(pat)
        if votes:
            if shortest:  # prefer the most direct connection, then the most frequent one
                pat = min(votes, key=lambda p: (depth[p], -votes[p]))
            else:
                pat = votes.most_common(1)[0][0]
            col_pats.append((k, pat))
    attempts = []
    pats = list(col_pats)
    while True:
        sparql = build_query(qg, anchor_key[1], [p for _, p in pats])
        rows, err = run_query(store, sparql)
        attempts.append({"sparql": sparql, "n_rows": len(rows or []), "err": err})
        if rows:
            return {"sparql": sparql, "rows": rows, "columns": [anchor_key] + [k for k, _ in pats],
                    "group_scores": {str(k): round(gscore[k], 3) for k in sel}, "attempts": attempts}
        if not pats:
            return {"sparql": sparql, "rows": [], "columns": [anchor_key], "attempts": attempts,
                    "group_scores": {str(k): round(gscore[k], 3) for k in sel}}
        if shortest:  # drop the longest-path (then weakest) column first
            drop = max(range(len(pats)), key=lambda i: (len(pats[i][1]), -gscore[pats[i][0]]))
            pats = pats[:drop] + pats[drop + 1:]
        else:
            pats = pats[:-1]  # drop the weakest column and retry
