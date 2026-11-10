"""Apply the decisions taken with ``qc_review.py``.

For every sample in ``<qc_dir>/qc_decisions.json``:
  * ``prediction`` — the BIDS ground-truth label file is replaced by
    ``<qc_dir>/predictions/<stem>_discs.nii.gz``. The original is first backed up to
    ``<qc_dir>/backup_gt/<bids-relative label path>`` (never overwritten, so rerunning
    is safe). git-annex symlinks are replaced by a regular file; the label's ``.json``
    sidecar, when there is one, gets a ``GeneratedBy`` entry describing the change.
  * ``drop``       — the sample's entry is removed from every discslabel JSON
    (data JSON + evaluation.json) listed in ``qc_index.json``. BIDS is untouched.
  * ``keep``       — nothing.

Applied samples are logged in ``<qc_dir>/qc_applied.json``.

Usage:
    python qc_apply.py <qc_dir> [--dry-run] [--decisions <file>]
                       [--json-dir <repo>/discslabel/datasets] [--bids-root <bids>]

``--decisions`` applies a ``qc_decisions.json`` reviewed on another machine (see
``qc_bundle.py``). On a real run it becomes ``<qc_dir>/qc_decisions.json`` (the previous
one is kept as ``qc_decisions.<timestamp>.bak.json``). When a decision carries a
``pred_sha256`` (offline review), the prediction is only written if the file on disk
is still the one that was reviewed.

``--json-dir`` / ``--bids-root`` override the paths stored at staging time (e.g. when
the review was done on another machine than the one that ran ``qc_prepare.sh``).
Afterwards, commit the label changes in the BIDS repos (``git annex add`` / ``git commit``)
and the JSONs in discs-labeling-multicontrast, then re-run the ``prepare`` step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime

SPLITS = ("TRAINING", "VALIDATION", "TESTING")


def sidecar_of(label_path: str) -> str:
    for ext in (".nii.gz", ".nii"):
        if label_path.endswith(ext):
            return label_path[: -len(ext)] + ".json"
    return label_path + ".json"


def load_json(path: str):
    """``json.load`` that names the offending file on a syntax error."""
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"[apply] invalid JSON in {path}: {exc}\n"
                             f"        nothing was changed by this run; fix the file and rerun") from None


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sidecar(path: str) -> dict:
    """Label sidecar content, or ``{}`` when it is not valid JSON (it then gets overwritten)."""
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:
            print(f"  WARN invalid sidecar, its content will be replaced: {path} ({exc})")
            return {}


def replace_label(pred: str, dst: str, backup: str, run_name: str, dry_run: bool) -> None:
    if not os.path.isfile(pred):
        raise FileNotFoundError(f"prediction missing: {pred}")
    if not os.path.exists(dst):
        raise FileNotFoundError(f"BIDS label missing (annex content not fetched?): {dst}")
    print(f"  replace {dst}")
    if dry_run:
        return
    if not os.path.exists(backup):
        os.makedirs(os.path.dirname(backup), exist_ok=True)
        shutil.copyfile(dst, backup)  # follows the annex symlink -> real content
    # Remove first: an annex symlink points to a read-only object we must not write through.
    os.remove(dst)
    shutil.copyfile(pred, dst)

    sidecar = sidecar_of(dst)
    if os.path.lexists(sidecar):
        meta = read_sidecar(sidecar)
        gen = []
        gen.append({
            "Name": "discslabel prediction",
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        gen.append({
            "Name": "Manual",
            "Author": "Nathan Molinier",
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        meta["GeneratedBy"] = gen
        os.remove(sidecar)
        with open(sidecar, "w") as f:
            json.dump(meta, f, indent=4)
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("qc_dir")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would change.")
    parser.add_argument("--decisions", default=None,
                        help="Decisions file to apply (default: <qc_dir>/qc_decisions.json).")
    parser.add_argument("--json-dir", default=None, help="Folder holding the discslabel JSONs (overrides qc_index.json).")
    parser.add_argument("--bids-root", default=None, help="BIDS root (overrides qc_index.json).")
    args = parser.parse_args()

    qc_dir = os.path.abspath(args.qc_dir)
    index = load_json(os.path.join(qc_dir, "qc_index.json"))
    local_dec = os.path.join(qc_dir, "qc_decisions.json")
    dec_path = os.path.abspath(args.decisions) if args.decisions else local_dec
    if not os.path.isfile(dec_path):
        raise SystemExit(f"no decisions yet ({dec_path}); run qc_review.py first")
    decisions = load_json(dec_path)
    unknown = sorted(st for st in decisions if st not in index["samples"])
    if unknown:
        print(f"[apply] WARN: {len(unknown)} decision(s) for stems not in qc_index.json "
              f"(different QC run?), ignored: {', '.join(unknown[:5])}{' ...' if len(unknown) > 5 else ''}")
    applied_path = os.path.join(qc_dir, "qc_applied.json")
    applied = {}
    if os.path.isfile(applied_path):
        applied = load_json(applied_path)

    run_name = ""
    cfg_path = os.path.join(qc_dir, "config.json")
    if os.path.isfile(cfg_path):
        run_name = load_json(cfg_path).get("run_name", "")

    bids_root = args.bids_root or index["bids_root"]

    def json_path(p: str) -> str:
        return os.path.join(args.json_dir, os.path.basename(p)) if args.json_dir else p

    by_kind = {"prediction": [], "drop": [], "keep": []}
    for stem, rec in sorted(decisions.items()):
        d = rec.get("decision")
        if d in by_kind and stem in index["samples"]:
            by_kind[d].append(stem)
    for stem, done in sorted(applied.items()):
        now = decisions.get(stem, {}).get("decision")
        if now != done:
            print(f"[apply] WARN: {stem} was already applied as '{done}' but is now '{now}' — "
                  f"not reverted automatically (backup GT: {os.path.join(qc_dir, 'backup_gt')})")
    print(f"[apply] prediction={len(by_kind['prediction'])} drop={len(by_kind['drop'])} "
          f"keep={len(by_kind['keep'])}{'  (dry run)' if args.dry_run else ''}")

    # ---- Pre-flight: parse every dataset JSON this run will touch, before changing anything ----
    # (invalid label sidecars are not an error: they are overwritten in replace_label)
    for stem in by_kind["prediction"]:
        if applied.get(stem) != "prediction":
            sidecar = sidecar_of(os.path.join(bids_root, index["samples"][stem]["label"]))
            if os.path.lexists(sidecar) and args.dry_run:
                read_sidecar(sidecar)  # just report it
    for stem in by_kind["drop"]:
        for use in index["samples"][stem]["uses"]:
            load_json(json_path(use["json"]))

    if dec_path != local_dec and not args.dry_run:
        # Keep the qc_dir self-describing: the applied decisions live next to qc_applied.json.
        if os.path.isfile(local_dec):
            old = load_json(local_dec)
            if old != decisions:
                bak = os.path.join(qc_dir, f"qc_decisions.{datetime.now():%Y%m%d-%H%M%S}.bak.json")
                os.replace(local_dec, bak)
                print(f"[apply] previous {local_dec} moved to {bak}")
        shutil.copyfile(dec_path, local_dec)

    # ---- Replace GT by prediction in BIDS ----
    n_replaced = 0
    changed = []
    for stem in by_kind["prediction"]:
        if applied.get(stem) == "prediction":
            continue
        label_rel = index["samples"][stem]["label"]
        pred = os.path.join(qc_dir, "predictions", f"{stem}_discs.nii.gz")
        want = decisions[stem].get("pred_sha256")
        if want and os.path.isfile(pred) and sha256_of(pred) != want:
            changed.append(stem)
            print(f"  SKIP {stem}: prediction changed since it was reviewed")
            continue
        replace_label(
            pred=pred,
            dst=os.path.join(bids_root, label_rel),
            backup=os.path.join(qc_dir, "backup_gt", label_rel),
            run_name=run_name,
            dry_run=args.dry_run,
        )
        n_replaced += 1
        if not args.dry_run:
            applied[stem] = "prediction"

    # ---- Drop samples from the JSONs ----
    drop_images = {}  # json path -> set of IMAGE paths
    for stem in by_kind["drop"]:
        s = index["samples"][stem]
        for use in s["uses"]:
            drop_images.setdefault(json_path(use["json"]), set()).add(s["image"])
    n_dropped = 0
    for jp, images in sorted(drop_images.items()):
        data = load_json(jp)
        removed = []
        for split in SPLITS:
            if split not in data:
                continue
            keep = [e for e in data[split] if e.get("IMAGE") not in images]
            removed += [f"{split}: {e['IMAGE']}" for e in data[split] if e.get("IMAGE") in images]
            data[split] = keep
        if not removed:
            continue
        print(f"  {os.path.basename(jp)}: remove {len(removed)} entr{'y' if len(removed) == 1 else 'ies'}")
        for r in removed:
            print(f"    - {r}")
        n_dropped += len(removed)
        if not args.dry_run:
            tmp = jp + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=4)  # same layout as the files in discslabel/datasets
            os.replace(tmp, jp)
    if not args.dry_run:
        for stem in by_kind["drop"]:
            applied[stem] = "drop"
        with open(applied_path, "w") as f:
            json.dump(applied, f, indent=2, sort_keys=True)

    prefix = "[apply] (dry run) would have" if args.dry_run else "[apply]"
    print(f"{prefix} replaced {n_replaced} BIDS label(s), removed {n_dropped} JSON entr(ies)")
    if changed:
        print(f"[apply] {len(changed)} prediction(s) skipped because they changed after review — "
              f"re-bundle and review them again: {', '.join(changed)}")


if __name__ == "__main__":
    main()
