"""Pack the lightweight part of a QC folder so the review can run on another machine.

Run on the server (after ``qc_prepare.sh``). The bundle only holds what
``qc_review.py`` needs — no NIfTI images:

    qc_index.json  config.json  metrics/metrics.json  metrics/previews/*.png
    review_manifest.json   which samples have a prediction + its sha256

The sha256 is copied into every "prediction" decision, so ``qc_apply.py`` refuses
to write a prediction file that changed since it was reviewed (e.g. inference re-run).

Round trip:
    # server
    python qc_bundle.py <qc_dir>                 # -> <qc_dir>/qc_review_bundle.tar.gz
    # laptop
    scp server:<qc_dir>/qc_review_bundle.tar.gz . && tar xzf qc_review_bundle.tar.gz
    python qc_review.py <bundle_dir>             # writes <bundle_dir>/qc_decisions.json
    scp <bundle_dir>/qc_decisions.json server:<qc_dir>/qc_decisions.review.json
    # server
    python qc_apply.py <qc_dir> --decisions <qc_dir>/qc_decisions.review.json --dry-run

Usage:
    python qc_bundle.py <qc_dir> [--out <file.tar.gz>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import tarfile
from datetime import datetime

MANIFEST = "review_manifest.json"


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("qc_dir", help="Folder produced by qc_prepare.sh.")
    parser.add_argument("--out", default=None, help="Bundle path (default: <qc_dir>/qc_review_bundle.tar.gz).")
    args = parser.parse_args()

    qc_dir = os.path.abspath(args.qc_dir)
    out = os.path.abspath(args.out or os.path.join(qc_dir, "qc_review_bundle.tar.gz"))
    with open(os.path.join(qc_dir, "qc_index.json")) as f:
        index = json.load(f)

    predictions = {}
    for stem, s in index["samples"].items():
        pred = os.path.join(qc_dir, "predictions", f"{stem}_discs.nii.gz")
        if not s.get("missing") and os.path.isfile(pred):
            predictions[stem] = sha256_of(pred)
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_host": socket.gethostname(),
        "source_qc_dir": qc_dir,
        "predictions": predictions,
    }
    with open(os.path.join(qc_dir, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    root = os.path.basename(qc_dir)
    members = [MANIFEST, "qc_index.json", "config.json", os.path.join("metrics", "metrics.json")]
    previews = os.path.join(qc_dir, "metrics", "previews")
    n_png = 0
    tmp = out + ".tmp"
    with tarfile.open(tmp, "w:gz") as tar:
        for rel in members:
            if os.path.isfile(os.path.join(qc_dir, rel)):
                tar.add(os.path.join(qc_dir, rel), arcname=os.path.join(root, rel))
            else:
                print(f"[bundle] WARN: {rel} not found, skipped")
        if os.path.isdir(previews):
            for name in sorted(os.listdir(previews)):
                if name.endswith(".png"):
                    tar.add(os.path.join(previews, name), arcname=os.path.join(root, "metrics", "previews", name))
                    n_png += 1
    os.replace(tmp, out)

    size_mb = os.path.getsize(out) / 1e6
    print(f"[bundle] {len(index['samples'])} samples, {len(predictions)} predictions, {n_png} previews "
          f"-> {out} ({size_mb:.1f} MB)")
    print(f"[bundle] laptop:  scp {socket.gethostname()}:{out} . && tar xzf {os.path.basename(out)} "
          f"&& python qc_review.py {root}")
    print(f"[bundle] back:    scp {root}/qc_decisions.json {socket.gethostname()}:{qc_dir}/qc_decisions.review.json")
    print(f"[bundle] apply:   python qc_apply.py {qc_dir} --decisions {qc_dir}/qc_decisions.review.json --dry-run")


if __name__ == "__main__":
    main()
