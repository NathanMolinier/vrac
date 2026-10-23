import glob
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import center_of_mass

bids_dataset = '/Users/nathan/data/spider-challenge-2023'
pred_dir = '/Users/nathan/Desktop/step2_output'
qc_dir = os.path.join(bids_dataset, 'derivatives', 'qc_fix_segmentation')
log_file = os.path.join(qc_dir, 'wrong_files.txt')


def fix_values(cur, pred):
    """Remap every cur label to the dominant pred label on its mask.
    Values matching pred stay unchanged (mode is themselves)."""
    fixed = cur.copy()
    replacements = {}
    for v in np.unique(cur):
        if v == 0:
            continue
        mask = cur == v
        candidates = pred[mask]
        candidates = candidates[candidates > 0]
        if candidates.size == 0:
            continue
        counts = np.bincount(candidates.astype(np.int64))
        new_val = int(np.argmax(counts))
        if new_val != int(v):
            fixed[mask] = new_val
            replacements[int(v)] = new_val
    return fixed, replacements


def mid_sagittal(data):
    axis = int(np.argmin(data.shape))
    idx = data.shape[axis] // 2
    return np.take(data, idx, axis=axis)


def _annotate_labels(ax, slice_2d):
    for v in np.unique(slice_2d):
        if v == 0:
            continue
        r, c = center_of_mass(slice_2d == v)
        txt = ax.text(c, r, str(int(v)), color='white', fontsize=7,
                      ha='center', va='center', fontweight='bold')
        txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground='black')])


def save_qc(before, after, out_png, title):
    vmax = int(max(before.max(), after.max()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    for ax, data, name in zip(axes, (before, after), ('before', 'after')):
        sl = np.rot90(mid_sagittal(data))
        ax.imshow(np.ma.masked_where(sl == 0, sl),
                  cmap='nipy_spectral', vmin=1, vmax=vmax, interpolation='nearest')
        _annotate_labels(ax, sl)
        ax.set_title(name)
        ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)


def pred_path_for(seg_fname):
    base = seg_fname[:-len('_label-spine_dseg.nii.gz')] + '.nii.gz'
    return os.path.join(pred_dir, base)


def main():
    os.makedirs(qc_dir, exist_ok=True)
    pattern = os.path.join(bids_dataset, 'derivatives', 'labels',
                           'sub-*', 'anat', '*_label-spine_dseg.nii.gz')
    seg_files = sorted(glob.glob(pattern))
    wrong_files = []

    for seg_path in seg_files:
        pred_file = pred_path_for(os.path.basename(seg_path))
        if not os.path.isfile(pred_file):
            print(f'MISSING prediction, skipped: {seg_path}')
            continue

        seg_img = nib.load(seg_path)
        seg_data = np.asanyarray(seg_img.dataobj).astype(np.int64)
        pred_data = np.asanyarray(nib.load(pred_file).dataobj).astype(np.int64)
        if pred_data.shape != seg_data.shape:
            print(f'SHAPE mismatch, skipped: {seg_path} '
                  f'({seg_data.shape} vs {pred_data.shape})')
            continue

        fixed, replacements = fix_values(seg_data, pred_data)
        if not replacements:
            continue

        out_img = nib.Nifti1Image(fixed.astype(seg_img.get_data_dtype()),
                                  seg_img.affine, seg_img.header)
        nib.save(out_img, seg_path)

        png_path = os.path.join(
            qc_dir, os.path.basename(seg_path).replace('.nii.gz', '.png'))
        save_qc(seg_data, fixed, png_path, os.path.basename(seg_path))

        wrong_files.append(seg_path)
        print(f'fixed: {seg_path}  mapping={replacements}')

    with open(log_file, 'w') as f:
        f.write('\n'.join(wrong_files) + ('\n' if wrong_files else ''))

    print(f'\n{len(wrong_files)} files fixed. List -> {log_file}')
    print(f'QC figures -> {qc_dir}')


if __name__ == '__main__':
    main()
