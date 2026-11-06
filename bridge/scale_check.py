"""Scale check (no QA): BRIDGE retrieval on the large anonymized graphs, 10 synthetic questions each.
Reports latency, peak GPU memory and context size. Questions are generic, written for this check
(not taken from BuildingQA)."""
import json
import time

import numpy as np
import torch

from bridge import kg
from bridge.retrieve import bq_graph, bridge_context, load_model

QS = {
    "bldg_brick_anon": [
        "Which VAV boxes feed which HVAC zones?",
        "List the zone air temperature sensors and the equipment they belong to.",
        "What are the timeseries IDs of all supply air temperature sensors on air handling units?",
        "Which air handling units have a supply fan and what are its status points?",
        "Which rooms are part of each HVAC zone?",
        "Find the discharge air temperature setpoints of every VAV.",
        "Which chillers supply chilled water to which air handling units?",
        "List all occupancy sensors and the rooms where they are located.",
        "What damper position commands exist for each VAV box?",
        "Which floors contain which rooms?",
    ],
    "bldg_223p_anon": [
        "Which fans are contained in which air handling units?",
        "List the temperature sensors and the properties they observe.",
        "Which zones are served by which terminal units?",
        "What connection points does each damper have?",
        "Which domain spaces belong to each zone?",
        "Find the coils and the equipment that contains them.",
        "Which properties have external timeseries references?",
        "List the pumps and what they are connected to.",
        "Which valves are connected to which coils?",
        "What sensors measure air flow and where are they located?",
    ],
}


def main(ckpt=kg.ROOT / "runs" / "ckpt" / "main.pt"):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(ckpt, dev)
    out = {}
    for b, qs in QS.items():
        t0 = time.time()
        ig = bq_graph(b, dev)
        load_s = time.time() - t0
        lat, toks, npaths, mem = [], [], [], []
        bridge_context(model, ig, qs[0])  # warm-up
        for q in qs:
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t = time.time()
            blk, info = bridge_context(model, ig, q)
            if dev.type == "cuda":
                torch.cuda.synchronize()
                mem.append(torch.cuda.max_memory_allocated() / 2**20)
            lat.append(time.time() - t)
            toks.append(info["tokens"])
            npaths.append(info["n_paths"])
        out[b] = {"nodes": len(ig.g["node_ids"]), "fwd_edges": ig.g["n_fwd_edges"],
                  "subgraph_mode": ig.g["n_fwd_edges"] > 100_000, "graph_build_or_load_s": round(load_s, 1),
                  "latency_s_mean": float(np.mean(lat)), "latency_s_max": float(np.max(lat)),
                  "peak_gpu_mem_mb": float(np.max(mem)) if mem else None,
                  "context_tokens_mean": float(np.mean(toks)), "n_paths_mean": float(np.mean(npaths)),
                  "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
                  "example_block": blk[:1500]}
        print(b, {k: v for k, v in out[b].items() if k != "example_block"}, flush=True)
    (kg.ROOT / "runs" / "scale_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
