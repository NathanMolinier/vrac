"""
Correct a TotalSpineSeg segmentation:
  - Remap L6 vertebra to sacrum and L5-L6 disc to L5-S when they were
    wrongly identified.
  - Optionally shift every disc and vertebra label by one anatomical
    level toward the sacrum when the model mislabeled a sacrum disc as
    L5-S (the current L4-L5 becomes L5-S, L3-L4 becomes L4-L5, ...).
"""

import argparse
import os

import numpy as np

from vrac.data_management.image import Image


tss_label = {
    "C1": 11,
    "C2": 12,
    "C3": 13,
    "C4": 14,
    "C5": 15,
    "C6": 16,
    "C7": 17,
    "T1": 21,
    "T2": 22,
    "T3": 23,
    "T4": 24,
    "T5": 25,
    "T6": 26,
    "T7": 27,
    "T8": 28,
    "T9": 29,
    "T10": 30,
    "T11": 31,
    "T12": 32,
    "L1": 41,
    "L2": 42,
    "L3": 43,
    "L4": 44,
    "L5": 45,
    "L6": 46,
    "L7": 47,
    "sacrum": 50,
    "C2-C3": 63,
    "C3-C4": 64,
    "C4-C5": 65,
    "C5-C6": 66,
    "C6-C7": 67,
    "C7-T1": 71,
    "T1-T2": 72,
    "T2-T3": 73,
    "T3-T4": 74,
    "T4-T5": 75,
    "T5-T6": 76,
    "T6-T7": 77,
    "T7-T8": 78,
    "T8-T9": 79,
    "T9-T10": 80,
    "T10-T11": 81,
    "T11-T12": 82,
    "T12-L1": 91,
    "L1-L2": 92,
    "L2-L3": 93,
    "L3-L4": 94,
    "L4-L5": 95,
    "L5-L6": 96,
    "L5-S": 100,
}

# Ordered top → bottom, used to shift labels one level toward the sacrum.
VERT_ORDER = [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7",
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
    "L1", "L2", "L3", "L4", "L5",
    "sacrum",
]

DISC_ORDER = [
    "C2-C3", "C3-C4", "C4-C5", "C5-C6", "C6-C7", "C7-T1",
    "T1-T2", "T2-T3", "T3-T4", "T4-T5", "T5-T6", "T6-T7",
    "T7-T8", "T8-T9", "T9-T10", "T10-T11", "T11-T12", "T12-L1",
    "L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S",
]


def get_parser():
    parser = argparse.ArgumentParser(
        description="Correct a TotalSpineSeg segmentation (L6/L5-L6 remap, optional level shift)."
    )
    parser.add_argument(
        "-i", "--image", required=True,
        help="Path to the TotalSpineSeg segmentation NIfTI to correct (Required).",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output path for the corrected segmentation. Defaults to <input>_corrected.nii.gz.",
    )
    parser.add_argument(
        "--shift", action="store_true",
        help="Shift every disc and vertebra label one level toward the sacrum "
             "(use when a sacrum disc was mislabeled as L5-S).",
    )
    parser.add_argument(
        "--l6-fix", action="store_true",
        help="Do L6 → sacrum and L5-L6 → L5-S remap.",
    )
    return parser


def extract_present(data, keys):
    present = {}
    values = np.unique(data)
    for name in keys:
        v = tss_label[name]
        if v in values:
            present[name] = v
    return present


def fix_l6(data):
    """Remap the wrongly identified L6 vertebra and L5-L6 disc."""
    changed = []
    if tss_label["L5-S"] in data:
        data[data == tss_label["L5-S"]] = 0
        changed.append("L5-S disc → background")
    if tss_label["L5-L6"] in data:
        data[data == tss_label["L5-L6"]] = tss_label["L5-S"]
        changed.append("L5-L6 disc → L5-S")
    if tss_label["L6"] in data:
        data[data == tss_label["L6"]] = tss_label["sacrum"]
        changed.append("L6 vertebra → sacrum")
    return changed


def shift_labels(data):
    """
    Shift every disc/vertebra one anatomical level toward the sacrum.

    Rationale: the model mislabeled a sacrum disc as L5-S. The disc named
    L4-L5 is really L5-S, L3-L4 is really L4-L5, etc. Vertebrae follow
    the same pattern (L4 → L5, L3 → L4, ...). The current L5-S disc and
    L5 vertebra are absorbed into the sacrum since they are actually part
    of it.
    """
    changed = []

    # Build the disc remap: current name → new name (one level lower).
    disc_remap = {DISC_ORDER[i]: DISC_ORDER[i + 1] for i in range(len(DISC_ORDER) - 1)}
    # The current L5-S is really a sacrum disc; drop it (0 = background).
    l5s_val = tss_label["L5-S"]
    if l5s_val in data:
        data[data == l5s_val] = 0
        changed.append("L5-S disc → background (sacrum disc)")

    # Apply the disc remap bottom-up so we do not overwrite values twice.
    for src in reversed(DISC_ORDER[:-1]):
        dst = disc_remap[src]
        src_v, dst_v = tss_label[src], tss_label[dst]
        mask = data == src_v
        if mask.any():
            data[mask] = dst_v
            changed.append(f"{src} disc → {dst}")

    # Same shift for vertebrae.
    vert_remap = {VERT_ORDER[i]: VERT_ORDER[i + 1] for i in range(len(VERT_ORDER) - 1)}
    l5_val = tss_label["L5"]
    if l5_val in data:
        data[data == l5_val] = tss_label["sacrum"]
        changed.append("L5 → sacrum")

    for src in reversed(VERT_ORDER[:-1]):
        if src == "L5":
            continue
        dst = vert_remap[src]
        src_v, dst_v = tss_label[src], tss_label[dst]
        mask = data == src_v
        if mask.any():
            data[mask] = dst_v
            changed.append(f"{src} → {dst}")

    return changed


def main():
    args = get_parser().parse_args()

    in_path = args.image
    out_path = args.output
    l6_fix = args.l6_fix
    shift = args.shift

    in_path = os.path.abspath(in_path)
    if out_path is None:
        base, ext = os.path.splitext(in_path)
        if base.endswith(".nii.gz"):
            base, ext2 = os.path.splitext(base)
            ext = ext2 + ext
        out_path = base + "_corrected" + ext
    else:
        out_path = os.path.abspath(out_path)

    img = Image(in_path).change_orientation("RSP")
    data = img.data.astype(np.int8)

    print(f"Loaded: {in_path}")
    print(f"Orientation: {img.orientation}")

    discs = extract_present(data, DISC_ORDER + ["L5-L6"])
    verts = extract_present(data, VERT_ORDER + ["L6", "L7"])
    print(f"Discs found: {sorted(discs)}")
    print(f"Vertebrae found: {sorted(verts)}")

    if l6_fix:
        for msg in fix_l6(data):
            print(f"[l6-fix] {msg}")

    if shift:
        for msg in shift_labels(data):
            print(f"[shift] {msg}")

    img.data = data
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    img.change_orientation('LPI').save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
