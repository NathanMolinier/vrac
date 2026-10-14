import os, argparse, textwrap
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import center_of_mass
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_parser():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent('''
            For each NIfTI rootlets segmentation in the input folder (label values 2-9),
            write a NIfTI to the output folder containing a single center-of-mass voxel
            per label, keeping the original label value.
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--seg-folder-in', '-i', type=str, required=True,
        help='Input folder with rootlets segmentation NIfTI files'
    )
    parser.add_argument(
        '--seg-folder-out', '-o', type=str, required=True,
        help='Output folder for center-of-mass NIfTI files'
    )
    parser.add_argument(
        '--preview-folder-out', '-p', type=str, default=None,
        help='Output folder for sagittal preview PNGs (default: <seg-folder-out>/preview)'
    )
    parser.add_argument(
        '--verbose', '-v', type=int, default=1, choices=[0, 1],
        help='Verbosity level. 0: Errors/warnings only, 1: Errors/warnings + info (default: 1)'
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    seg_folder_in = args.seg_folder_in
    seg_folder_out = args.seg_folder_out
    preview_folder_out = args.preview_folder_out or os.path.join(seg_folder_out, 'preview')
    verbose = args.verbose

    if verbose:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            seg_folder_in = "{seg_folder_in}"
            seg_folder_out = "{seg_folder_out}"
            preview_folder_out = "{preview_folder_out}"
            verbose = {verbose}
        '''))

    os.makedirs(seg_folder_out, exist_ok=True)
    os.makedirs(preview_folder_out, exist_ok=True)

    for fname in sorted(os.listdir(seg_folder_in)):
        if not (fname.endswith('.nii') or fname.endswith('.nii.gz')):
            continue
        seg_in = os.path.join(seg_folder_in, fname)
        seg_out = os.path.join(seg_folder_out, fname)
        preview_name = fname.replace('.nii.gz', '.png').replace('.nii', '.png')
        preview_out = os.path.join(preview_folder_out, preview_name)
        if verbose:
            print(f'Processing: {seg_in} -> {seg_out}')
        center_of_mass_per_label(seg_in, seg_out)
        save_sagittal_preview(seg_in, seg_out, preview_out)


def center_of_mass_per_label(seg_in, seg_out):
    seg = nib.load(seg_in)
    seg_data = seg.get_fdata().astype(np.uint8)

    out_data = np.zeros_like(seg_data)

    for l in [v for v in np.unique(seg_data) if v != 0]:
        mask = seg_data == l
        com = center_of_mass(mask)
        voxel = tuple(int(round(c)) for c in com)
        # Snap to the nearest in-label voxel if the raw COM falls outside the mask
        if not mask[voxel]:
            coords = np.argwhere(mask)
            distances = np.linalg.norm(coords - np.array(com), axis=1)
            voxel = tuple(coords[distances.argmin()])
        out_data[voxel] = l

    out = nib.Nifti1Image(out_data, seg.affine, seg.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, seg_out)


def save_sagittal_preview(seg_in, com_in, preview_out):
    """Sagittal max-projection of the full segmentation with the COM points
    overlaid as colored markers (one color per label)."""
    seg = nib.as_closest_canonical(nib.load(seg_in)).get_fdata().astype(np.uint8)
    com = nib.as_closest_canonical(nib.load(com_in)).get_fdata().astype(np.uint8)

    # Canonical RAS: axis 0 = L-R, axis 1 = A-P, axis 2 = I-S.
    # Max-project along the L-R axis to get a sagittal view.
    bg = (seg > 0).any(axis=0).astype(float)  # (y, z)

    # Display with z as vertical (superior up) and y as horizontal.
    bg_disp = np.rot90(bg)

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(bg_disp, cmap='gray', interpolation='nearest')

    labels = [v for v in np.unique(com) if v != 0]
    cmap = plt.get_cmap('tab10')
    for l in labels:
        coords = np.argwhere(com == l)
        for _, y, z in coords:
            # After rot90, display (col, row) = (y, bg.shape[1] - 1 - z)
            ax.scatter(y, bg.shape[1] - 1 - z,
                       s=40, color=cmap((int(l) - 2) % 10),
                       edgecolors='white', linewidths=0.8,
                       label=f'label {int(l)}')

    # Deduplicate legend entries.
    handles, lbls = ax.get_legend_handles_labels()
    seen = {}
    for h, lb in zip(handles, lbls):
        seen.setdefault(lb, h)
    ax.legend(seen.values(), seen.keys(), loc='upper right',
              fontsize=8, framealpha=0.7)

    ax.set_title(os.path.basename(seg_in))
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(preview_out, dpi=120, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
