"""Question sets for end-to-end runs: BuildingQA (188 q) and the Build2SPARQL-test dev set (60 q)."""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BQ = ROOT / "data" / "BuildingQA"
ONTO = BQ / "bschema" / "ontologies"


def safe(qkey):
    return qkey.replace("#", "_q").replace("::", "__")


def is_dev(dataset):
    return dataset.startswith("dev")


def out_root(dataset, agent):
    if is_dev(dataset):
        return ROOT / "runs" / dataset / agent
    return ROOT / "runs" / ("e2e" if agent == "brickqa" else "e2e_bkgqa")


def contexts_dir(dataset):
    return ROOT / "runs" / (dataset if is_dev(dataset) else "e2e")


@lru_cache(maxsize=None)
def questions(dataset):
    out = []
    if is_dev(dataset):
        for it in json.loads((ROOT / "runs" / "dev" / f"{dataset.replace('dev', 'dev_set', 1)}.json").read_text(encoding="utf-8")):
            out.append({k: it[k] for k in ("qkey", "qid", "building", "group", "text", "source", "stratum")}
                       | {"ttl": str(ROOT / it["ttl"]),
                          "tbox": str(ONTO / ("223p.ttl" if it["stratum"] == "s223" else "Brick.ttl"))})
        return tuple(out)
    for f in sorted((BQ / "Benchmark_QA_pairs").glob("*_combined.json")):
        for rec in json.loads(f.read_text(encoding="utf-8")):
            b = rec["building_id"].replace(".ttl", "")
            for q in rec["queries"]:
                for x in q["questions"]:
                    out.append({"qkey": f"{q['query_id']}#{x['question_number']}", "qid": q["query_id"],
                                "building": b, "group": b, "text": x["text"], "source": x["source"],
                                "ttl": str(BQ / "eval_buildings" / f"{b}.ttl"),
                                "tbox": str(ONTO / ("223p.ttl" if b == "b59" else "Brick.ttl"))})
    return tuple(out)


def question(dataset, qkey):
    for q in questions(dataset):
        if q["qkey"] == qkey:
            return q
    raise KeyError(qkey)


@lru_cache(maxsize=None)
def gold(dataset):
    """qid -> gold rows (scoring only)."""
    if is_dev(dataset):
        return {it["qid"]: it["gold_rows"] for it in json.loads((ROOT / "runs" / "dev" / f"{dataset.replace('dev', 'dev_set', 1)}.json").read_text(encoding="utf-8"))}
    g = json.loads((ROOT / "runs" / "data" / "buildingqa_gold.json").read_text())
    return {k: v["rows"] for k, v in g.items()}
