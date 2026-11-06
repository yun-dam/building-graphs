"""Leakage checks.

python -m bridge.check_leakage            # pre-training: split / BuildingQA isolation
python -m bridge.check_leakage --e2e      # post-E2E: per-question isolation + prompt grep
"""
import argparse
import json
import re
import sys
from pathlib import Path

from bridge import kg

ROOT = kg.ROOT
BQ = ROOT / "data" / "BuildingQA" / "Benchmark_QA_pairs"
B2S = ROOT / "data" / "Build2SPARQL_v1.0"
OUT = ROOT / "runs" / "leakage_check.txt"


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"\?\w+", "?v", s)).strip().lower()


def buildingqa_items():
    qs, sparqls, blds = [], [], set()
    for f in sorted(BQ.glob("*_combined.json")):
        for rec in json.loads(f.read_text(encoding="utf-8")):
            blds.add(rec["building_id"].replace(".ttl", ""))
            for q in rec["queries"]:
                sparqls.append((q["query_id"], q["sparql_query"]))
                for x in q["questions"]:
                    qs.append((q["query_id"], x["question_number"], x["text"]))
    return qs, sparqls, blds


def pre_training():
    lines, ok = [], True
    sp = kg.split()
    tr, va, te = set(sp["train"]), set(sp["val"]), set(sp["test"])
    for a, b, n in ((tr, va, "train/val"), (tr, te, "train/test"), (va, te, "val/test")):
        inter = a & b
        lines.append(f"[{'OK' if not inter else 'FAIL'}] split overlap {n}: {sorted(inter)}")
        ok &= not inter
    excl = set(sp["excluded"]) | set(sp["deduplicated"])
    bad = excl & (tr | va | te)
    lines.append(f"[{'OK' if not bad else 'FAIL'}] excluded/deduplicated buildings absent from splits: {sorted(bad)}")
    ok &= not bad
    qs, sparqls, blds = buildingqa_items()
    bad = blds & (tr | va | te)
    lines.append(f"[{'OK' if not bad else 'FAIL'}] BuildingQA buildings {sorted(blds)} not in any split: {sorted(bad)}")
    ok &= not bad
    # labels actually used for training/val
    qtexts = {norm(t) for _, _, t in qs}
    sq = {norm(s) for _, s in sparqls}
    hits_q, hits_s, n_lab = [], [], 0
    for b in sorted(tr | va):
        f = ROOT / "runs" / "labels" / f"{b}.json"
        if not f.exists():
            continue
        lab = json.loads(f.read_text())
        meta = {q["query_id"]: q for q in json.loads((B2S / f"{b}.json").read_text(encoding="utf-8"))[0]["queries"]}
        for rec in lab["queries"]:
            n_lab += 1
            for t in rec["questions"]:
                if norm(t) in qtexts:
                    hits_q.append((b, rec["query_id"], t))
            if norm(meta[rec["query_id"]]["sparql_query"]) in sq:
                hits_s.append((b, rec["query_id"]))
    lines.append(f"[{'OK' if not hits_q else 'FAIL'}] BuildingQA question texts in train/val labels ({n_lab} queries): {hits_q[:5]}")
    lines.append(f"[{'OK' if not hits_s else 'FAIL'}] BuildingQA gold SPARQL (var-normalized) in train/val labels: {hits_s[:5]}")
    ok &= not hits_q and not hits_s
    # graph-level: BuildingQA eval graph node IRIs in train/val graphs
    lines.append("[INFO] bldg11 excluded; b59 vs lbnl-bdg3-1 audited as different buildings in research_plan_1_updated.md")
    return ok, lines


def e2e():
    lines, ok = [], True
    qs, sparqls, _ = buildingqa_items()
    edir = ROOT / "runs" / "e2e"
    qtext = {f"{qid}#{n}": t for qid, n, t in qs}
    sparql_by_qid = dict(sparqls)
    n_rec, pids, starts = 0, set(), set()
    leaks, dup, fresh_ok = [], [], 0
    files = sorted(list((ROOT / "runs").glob("e2e*/*/*/results.jsonl")) + list((ROOT / "runs").glob("dev*/*/*/*/results.jsonl")))
    for res in files:
        for line in res.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            n_rec += 1
            key = (res.parent.parent.parent.name, res.parent.parent.name, res.parent.name)
            ident = (key, r.get("pid"), r.get("agent_start"))  # Windows reuses PIDs over time
            if ident in pids or r.get("pid") is None or r.get("agent_start") is None:
                dup.append(f"non-unique/missing process identity {ident[1:]} in {key}")
            pids.add(ident)
            starts.add((key, r.get("agent_start")))
            logf = res.parent / "logs" / f"{r['qkey']}.jsonl"
            if not logf.exists():
                leaks.append(f"missing log {logf}")
                continue
            loglines = [json.loads(l) for l in logf.read_text(encoding="utf-8").splitlines()]
            st = next((l for l in loglines if l.get("event") == "start"), {})
            if st.get("pid") == r.get("pid") and st.get("agent_start") == r.get("agent_start"):
                fresh_ok += 1
            prompts = " ".join(l.get("prompt", "") for l in loglines)
            pn = re.sub(r"\s+", " ", prompts).lower()
            own_q = r["qid"]
            for k, t in qtext.items():
                if k == own_q:
                    continue
                tn = re.sub(r"\s+", " ", t).lower().strip()
                own_text = re.sub(r"\s+", " ", qtext.get(own_q, "")).lower().strip()
                if len(tn) > 25 and tn in pn and tn != own_text:
                    leaks.append(f"{key} {own_q}: prompt contains other question {k}")
            for qid, s in sparql_by_qid.items():
                body = re.search(r"WHERE\s*\{(.*)\}", s, re.S | re.I)
                if body and len(body.group(1)) > 80:
                    bn = re.sub(r"\s+", " ", body.group(1)).lower().strip()
                    if bn in pn:
                        leaks.append(f"{key} {own_q}: prompt contains gold SPARQL body of {qid}")
    lines.append(f"[INFO] result records checked: {n_rec}")
    uniq = len(pids) == n_rec and n_rec > 0 and not dup
    lines.append(f"[{'OK' if uniq else 'FAIL'}] unique (pid, agent start) per question record within each config/run: {dup[:5]} (total {len(dup)})")
    lines.append(f"[{'OK' if fresh_ok == n_rec else 'FAIL'}] fresh agent start logged by the record's own process "
                 f"(log start event pid+timestamp == record): {fresh_ok}/{n_rec} "
                 f"({len(starts)} distinct timestamps; simultaneous starts share a timestamp at Windows clock resolution)")
    lines.append(f"[{'OK' if not leaks else 'FAIL'}] no other-question text / gold SPARQL in prompts: {leaks[:10]} (total {len(leaks)})")
    ok &= uniq and fresh_ok == n_rec and not leaks
    return ok, lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e", action="store_true")
    a = ap.parse_args()
    ok, lines = e2e() if a.e2e else pre_training()
    txt = "\n".join(lines) + f"\nRESULT: {'PASS' if ok else 'FAIL'}\n"
    print(txt)
    out = OUT.with_name("leakage_check_e2e.txt" if a.e2e else "leakage_check.txt")
    out.write_text(txt)
    sys.exit(0 if ok else 1)
