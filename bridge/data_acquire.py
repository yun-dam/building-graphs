"""Download and resolve the source graphs for every Build2SPARQL building in the split.

Writes graphs to runs/data/graphs/{building}.ttl (open223: also {building}.compiled.ttl),
a manifest runs/data/manifest.csv and runs/data/missing.csv.
"""
import csv
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B2S = ROOT / "data" / "Build2SPARQL_v1.0"
SPLIT = ROOT / "data" / "splits" / "brickqa_g_split.json"
OUT = ROOT / "runs" / "data"
GDIR = OUT / "graphs"

MORTAR = "https://brickschema.org/ttl/mortar/{}.ttl"
OPEN223 = "https://raw.githubusercontent.com/open223/models.open223.info/main/models/{}{}.ttl"
BRICK_RAW = "https://raw.githubusercontent.com/BrickSchema/Brick/v1.4.4/{}"
ZENODO_ZIP = "https://zenodo.org/api/records/20015123/files/buildings.zip/content"

BRICK_EXAMPLES = {
    "air_quality_sensor_example": "examples/air_quality_sensors/air_quality_sensor_example.ttl",
    "apartment": "examples/simple_apartment/apartment.ttl",
    "building_meter": "examples/building_meter/building_meter.ttl",
    "chiller-plant": "examples/chiller-system/chiller-plant.ttl",
    "controller": "examples/controller/controller.ttl",
    "main-and-submeter": "examples/submeter_hierarchies/main-and-submeter.ttl",
    "multiple-submeters": "examples/submeter_hierarchies/multiple-submeters.ttl",
    "single-meter": "examples/submeter_hierarchies/single-meter.ttl",
    "virtual-meter": "examples/submeter_hierarchies/virtual-meter.ttl",
    "rvav": "examples/connections/rvav.ttl",
    "solar_array": "examples/solar_array/solar_array.ttl",
    "rice_brick": "examples/rice_brick.ttl",
    "soda_brick": "examples/soda_brick.ttl",
    "g36-ahu-a9": "examples/g36/g36-ahu-a9.ttl",
    "g36-combined-ahu-vav": "examples/g36/g36-combined-ahu-vav.ttl",
    "g36-vav-a2": "examples/g36/g36-vav-a2.ttl",
}
# IBAL is listed as an exact duplicate of NIST-IBAL; the repo only hosts NIST-IBAL.ttl
OPEN223_ALIAS = {"IBAL": "NIST-IBAL"}
OPEN223_IDS = {
    "HPL", "NIST-HPL", "IBAL", "NIST-IBAL", "nist-bdg1-1", "lbnl-bdg3-1", "lbnl-example-radiant",
    "nrel-example", "pnnl-bdg1-2", "pnnl-bdg2-1", "pnnl-bdg3-2", "scb-vrf", "design-patterns",
} | {f"guideline36-2021-A-{i}" for i in (1, 2, 3, 4, 7, 8, 9)}


def fetch(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "bridge-research/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url}: {err}")


def split_buildings():
    s = json.loads(SPLIT.read_text())
    return {b: k for k in ("train", "val", "test") for b in s[k]}


def corpus_of(b):
    rec = json.loads((B2S / f"{b}.json").read_text(encoding="utf-8"))[0]
    return rec["corpus"]


def stratum_of(b):
    if b.startswith("BG_"):
        return "buildgraph"
    if b in OPEN223_IDS:
        return "s223"
    if b in BRICK_EXAMPLES:
        return "brick-reference"
    return "mortar-brick"


def main():
    GDIR.mkdir(parents=True, exist_ok=True)
    blds = split_buildings()
    rows, missing = [], []
    need_bg = [b for b in blds if b.startswith("BG_") and not (GDIR / f"{b}.ttl").exists()]
    if need_bg:
        print("downloading BuildGraph zip ...", flush=True)
        z = zipfile.ZipFile(io.BytesIO(fetch(ZENODO_ZIP)))
        names = {Path(n).stem: n for n in z.namelist() if n.endswith(".ttl")}
        for b in need_bg:
            if b in names:
                (GDIR / f"{b}.ttl").write_bytes(z.read(names[b]))
    for b, split in sorted(blds.items()):
        dst = GDIR / f"{b}.ttl"
        src = ""
        try:
            if b.startswith("BG_"):
                src = "zenodo:20015123"
            elif b in OPEN223_IDS:
                src = OPEN223.format("", OPEN223_ALIAS.get(b, b))
                if not dst.exists():
                    dst.write_bytes(fetch(src))
                comp = GDIR / f"{b}.compiled.ttl"
                if not comp.exists():
                    comp.write_bytes(fetch(OPEN223.format("compiled/", OPEN223_ALIAS.get(b, b))))
            elif b in BRICK_EXAMPLES:
                # v1.4.4 first; `controller` only exists from v1.5.0-rc1 on
                for tag in ("v1.4.4", "v1.5.0-rc1"):
                    src = BRICK_RAW.replace("v1.4.4", tag).format(BRICK_EXAMPLES[b])
                    try:
                        if not dst.exists():
                            dst.write_bytes(fetch(src, retries=1))
                        break
                    except RuntimeError:
                        continue
            else:
                src = MORTAR.format(b)
                if not dst.exists():
                    dst.write_bytes(fetch(src))
        except Exception as e:  # noqa: BLE001
            print("FAIL", b, e, flush=True)
        if dst.exists() and dst.stat().st_size > 0:
            rows.append({"building": b, "split": split, "stratum": stratum_of(b), "source": src,
                         "path": str(dst.relative_to(ROOT)), "bytes": dst.stat().st_size})
        else:
            missing.append({"building": b, "split": split, "stratum": stratum_of(b), "source": src})
    for name, data in (("manifest.csv", rows), ("missing.csv", missing)):
        with open(OUT / name, "w", newline="") as f:
            fields = ["building", "split", "stratum", "source", "path", "bytes"]
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(data)
    print(f"resolved {len(rows)} / {len(blds)}; missing {len(missing)}")


if __name__ == "__main__":
    sys.exit(main())
