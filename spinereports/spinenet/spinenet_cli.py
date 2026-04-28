#!/usr/bin/env python
"""
SpineNet CLI - Easy-to-use command-line interface for vertebra detection.

This tool provides a simple interface to run SpineNet vertebra detection on 
MRI scans (NIfTI format) and generate visualization outputs.

Installation:
- python=3.9
- pip install matplotlib nibabel
- pip install -r requirements.txt
- python3 -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118 --upgrade
- numpy<2

Usage:
    python spinenet_cli.py input_image.nii.gz --output output_dir
    python spinenet_cli.py input_image.nii.gz --output output_dir --device cuda:0 --save-viz
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import nibabel as nib
import cv2
import torch
import spinenet
from vrac.data_management.image import Image
from scipy.ndimage import zoom


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='SpineNet CLI for vertebra detection in MRI scans',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage with visualization
  python spinenet_cli.py scan.nii.gz --output results --save-viz
  
  # Run on GPU and save detailed results
  python spinenet_cli.py scan.nii.gz --output results --device cuda:0 --save-json
  
  # Customize scan type and visualization
  python spinenet_cli.py scan.nii.gz --output results --scan-type cervical --num-slices 15
        '''
    )
    
    parser.add_argument(
        'input_image',
        type=str,
        help='Path to input MRI scan (NIfTI format, .nii.gz or .nii)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='spinenet_results',
        help='Output directory for results (default: spinenet_results)'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='cuda:0',
        help='Device to use for inference (default: cuda:0). Use "cpu" for CPU-only.'
    )
    
    parser.add_argument(
        '--scan-type', '-s',
        type=str,
        choices=['whole', 'cervical', 'thoracic', 'lumbar'],
        default='whole',
        help='Type of scan (default: whole). Affects detection model used.'
    )
    
    parser.add_argument(
        '--save-viz', '-v',
        action='store_true',
        help='Save visualization images (axial, sagittal, coronal views)'
    )
    
    parser.add_argument(
        '--save-json', '-j',
        action='store_true',
        help='Save detection results as JSON'
    )
    
    parser.add_argument(
        '--num-slices',
        type=int,
        default=20,
        help='Number of slices to show in visualization grid (default: 20)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def enhance_local_contrast(image, kernel_size=5):
    """
    Enhance local contrast using CLAHE.
    
    Parameters
    ----------
    image : np.ndarray
        3D image volume, normalized to [0, 1]
    kernel_size : int
        Kernel size for CLAHE (default: 5)
    
    Returns
    -------
    np.ndarray
        Contrast-enhanced image
    """
    image = image.copy() / (image.max() + 1e-8)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(kernel_size, kernel_size))
    
    for slice_idx in range(image.shape[-1]):
        image[:, :, slice_idx] = clahe.apply(
            (image[:, :, slice_idx] * 255).astype(np.uint8)
        ) / 255
    
    return image


def visualize_midsagittal(volume, vert_dicts, output_path=None):
    """
    Create visualization on mid-sagittal plane with vertebra detections.
    
    Parameters
    ----------
    volume : np.ndarray
        3D image volume (axial, coronal, sagittal)
    vert_dicts : list
        List of vertebra detection dictionaries
    output_path : str, optional
        Path to save the figure
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Find mid-sagittal slice
    mid_sag = int(np.median([np.median(v['slice_nos']) for v in vert_dicts]))
    
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(volume[:, :, mid_sag], cmap='gray')
    
    # Draw vertebra polygons
    for vert_dict in vert_dicts:
        avg_poly = np.median(vert_dict['polys'], axis=0)
        ax.add_patch(Polygon(avg_poly, ec='yellow', fc='none', lw=2))
        ax.text(
            avg_poly[:, 0].mean(), avg_poly[:, 1].mean(),
            vert_dict['predicted_label'],
            color='yellow', fontsize=11, ha='center', va='center',
            fontweight='bold'
        )
    
    ax.set_title('Mid-Sagittal View with Vertebra Detections', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved mid-sagittal visualization: {output_path}")
    
    return fig


def visualize_slices_grid(volume, num_slices=20, num_per_row=5, output_path=None):
    """
    Create grid visualization of image slices.
    
    Parameters
    ----------
    volume : np.ndarray
        3D image volume
    num_slices : int
        Number of slices to display
    num_per_row : int
        Number of slices per row
    output_path : str, optional
        Path to save the figure
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    volume_norm = volume.copy() / (volume.max() + 1e-8)
    
    slices_to_show = [int(i) for i in np.linspace(0, volume.shape[-1] - 1, num_slices)]
    volume_slices = torch.Tensor(volume_norm[:, :, slices_to_show]).permute(2, 0, 1).unsqueeze(1)
    
    from torchvision.utils import make_grid
    grid = make_grid(volume_slices, nrow=num_per_row).permute(1, 2, 0).numpy()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(grid, cmap='gray')
    ax.set_title(f'Sagittal Slices Grid ({num_slices} slices)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved slices grid visualization: {output_path}")
    
    return fig


def save_results_json(vert_dicts, image_shape, pixel_spacing, slice_thickness, output_path):
    """
    Save vertebra detection results as JSON.
    
    Parameters
    ----------
    vert_dicts : list
        List of vertebra detection dictionaries
    image_shape : tuple
        Shape of the image volume
    pixel_spacing : tuple
        Pixel spacing (x, y)
    slice_thickness : float
        Slice thickness
    output_path : str
        Path to save JSON file
    """
    results = {
        'num_vertebrae': len(vert_dicts),
        'image_shape': image_shape,
        'pixel_spacing': [float(x) for x in pixel_spacing],
        'slice_thickness': float(slice_thickness),
        'vertebrae': []
    }
    
    for vert_dict in vert_dicts:
        vert_entry = {
            'label': vert_dict['predicted_label'],
            'confidence': float(vert_dict.get('confidence', 0)),
            'slice_nos': [int(s) for s in vert_dict['slice_nos']],
            'centroid': [float(x) for x in np.median(
                np.median(vert_dict['polys'], axis=0), axis=0
            )]
        }
        results['vertebrae'].append(vert_entry)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved detection results: {output_path}")


def main():
    """Main CLI function."""
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify input file exists
    input_path = Path(args.input_image)
    if not input_path.exists():
        raise FileNotFoundError(f"Error: Input file not found: {input_path}")
    
    print(f"SpineNet CLI - Vertebra Detection")
    print(f"{'=' * 50}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Scan type: {args.scan_type}")
    print()

    # Download model weights if not already present
    print("Checking for model weights...")
    spinenet.download_weights(verbose=True)
    
    # Load image
    print("Loading image...")
    image = Image(str(input_path))
    
    # Reorient to RAS
    print("Reorienting image...")
    image = image.change_orientation('RPI')
    
    # Get spacing information
    nx, ny, nz, nt, sx, sy, sz, st = image.dim
    pixel_spacing = np.array([sx, sy])
    slice_thickness = sz
    
    # Get image data
    print(f"Image shape: {image.data.shape}")
    print(f"Pixel spacing: ({sz:.2f}, {sy:.2f}, {sx:.2f}) mm")

    nb_slice = 12 # Use less slices
    skip_slices = 4
    nx = nx//skip_slices
    slice_thickness = sx*skip_slices
    image_data = np.moveaxis(zoom(image.data, (1/skip_slices, 1, 1)), 0, -1)[:, :, nx//2-nb_slice//2:nx//2+nb_slice//2]
    
    # Enhance contrast
    print("Enhancing local contrast...")
    image_data = enhance_local_contrast(image_data)

    # Create SpinalScan object
    print("Creating SpinalScan object...")
    scan = spinenet.io.SpinalScan(image_data, pixel_spacing, slice_thickness)
    
    # Run SpineNet detection
    print(f"Running SpineNet detection (scan_type={args.scan_type})...")
    spnt = spinenet.SpineNet(
        device=args.device,
        verbose=args.verbose,
        scan_type=args.scan_type
    )
    
    vert_dicts = spnt.detect_vb(scan.volume, scan.pixel_spacing[0])
    
    print()
    print(f"{'=' * 50}")
    print(f"Detection Results")
    print(f"{'=' * 50}")
    print(f"Found {len(vert_dicts)} vertebrae")
    print()
    
    # Print detected vertebrae
    for idx, vert_dict in enumerate(vert_dicts, 1):
        print(f"  {idx}. {vert_dict['predicted_label']}")
    print()
    
    # Save visualizations
    if args.save_viz:
        print("Generating visualizations...")
        
        # Mid-sagittal view
        viz_midsag = output_dir / 'midsagittal_detections.png'
        visualize_midsagittal(scan.volume, vert_dicts, str(viz_midsag))
        
        # Slices grid
        viz_grid = output_dir / 'slices_grid.png'
        visualize_slices_grid(scan.volume, args.num_slices, output_path=str(viz_grid))
        
        print()
    
    # Save JSON results
    if args.save_json:
        json_path = output_dir / 'detections.json'
        save_results_json(
            vert_dicts,
            image_data.shape,
            pixel_spacing,
            slice_thickness,
            str(json_path)
        )
        print()
    
    print(f"{'=' * 50}")
    print(f"Processing complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 50}")

if __name__ == '__main__':
    main()
