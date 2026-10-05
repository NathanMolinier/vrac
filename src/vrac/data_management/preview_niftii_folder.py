"""
Scan a folder tree for NIfTI (.nii.gz) files and write a sagittal mid-slice
preview (PNG or JPG) for each 3D volume into an output folder.

Images are reoriented to RSP (via vrac.data_management.image.Image) so the
sagittal slice is always the middle index along axis 0.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

from vrac.data_management.image import Image


def get_parser():
    parser = argparse.ArgumentParser(description='Generate sagittal mid-slice previews for all .nii.gz files in a folder tree.')
    parser.add_argument('--input', '-i', required=True, type=str, help='Input folder to scan recursively for .nii.gz files.')
    parser.add_argument('--output', '-o', required=True, type=str, help='Output folder where preview images will be written.')
    parser.add_argument('--format', '-f', default='png', choices=['png', 'jpg'], help='Output image format. Default: png.')
    parser.add_argument('--jobs', '-j', default=0, type=int, help='Number of worker processes. 0 (default) uses all available CPUs.')
    parser.add_argument('--percentile', '-p', default=95.0, type=float, help='Intensity percentile used for clipping before normalization. Default: 95.0.')
    return parser


def _find_nii_gz(root):
    hits = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith('.nii.gz'):
                hits.append(os.path.join(dirpath, name))
    return hits


def _relative_output_name(nii_path, input_root, ext):
    rel = os.path.relpath(nii_path, input_root)
    if rel.endswith('.nii.gz'):
        rel = rel[:-len('.nii.gz')]
    return rel.replace(os.sep, '_') + '.' + ext


def _process_one(nii_path, input_root, output_folder, ext, percentile):
    try:
        img = Image(nii_path).change_orientation('RSP')
        data = img.data
        if data.ndim == 4:
            data = data[..., 0]
        if data.ndim != 3:
            return nii_path, f'unsupported ndim={data.ndim}'

        mid = data.shape[0] // 2
        sl = np.asarray(data[mid, :, :], dtype=np.float32)

        hi = np.percentile(sl, percentile)
        lo = sl.min()
        if hi <= lo:
            hi = lo + 1.0
        np.clip(sl, lo, hi, out=sl)
        sl -= lo
        sl *= 255.0 / (hi - lo)
        out = sl.astype(np.uint8)

        # Flip rows so Superior ends up at the top in the displayed image.
        out = np.flipud(out)

        out_name = _relative_output_name(nii_path, input_root, ext)
        out_path = os.path.join(output_folder, out_name)
        cv2.imwrite(out_path, out)
        return nii_path, None
    except Exception as exc:
        return nii_path, repr(exc)


def main():
    args = get_parser().parse_args()

    input_root = os.path.abspath(args.input)
    output_folder = os.path.abspath(args.output)
    ext = args.format
    percentile = args.percentile

    if not os.path.isdir(input_root):
        raise SystemExit(f'Input folder does not exist: {input_root}')

    os.makedirs(output_folder, exist_ok=True)

    nii_files = _find_nii_gz(input_root)
    if not nii_files:
        print(f'No .nii.gz files found under {input_root}')
        return

    workers = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    print(f'Found {len(nii_files)} file(s). Using {workers} worker(s). Writing previews to {output_folder}')

    done = 0
    failed = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_process_one, p, input_root, output_folder, ext, percentile)
            for p in nii_files
        ]
        for fut in as_completed(futures):
            path, err = fut.result()
            done += 1
            if err is not None:
                failed.append((path, err))
            if done % 50 == 0 or done == len(nii_files):
                print(f'  [{done}/{len(nii_files)}] processed')

    if failed:
        print(f'{len(failed)} file(s) failed:')
        for path, err in failed[:20]:
            print(f'  {path}: {err}')
        if len(failed) > 20:
            print(f'  ... and {len(failed) - 20} more')


if __name__ == '__main__':
    main()
