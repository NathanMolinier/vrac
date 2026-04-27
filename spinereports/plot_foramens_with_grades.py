"""
Create a table/grid visualization of foramen images organized by stenosis grade and level.

Uses foraminal stenosis grades from READER 1 (Senior) and foraminal images from metrics_output.
Displays examples of images for each grade (columns) and spinal level (rows).
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image as PILImage
from collections import defaultdict
import glob
import random


def get_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='Create a table with foramen images organized by stenosis grade and level.'
    )
    parser.add_argument(
        '--csv-path',
        required=True,
        help='Path to the readout CSV file (e.g., Readout_lumbar_23112025.csv)'
    )
    parser.add_argument(
        '--metrics-folder',
        required=True,
        help='Path to the metrics_output folder containing subject results'
    )
    parser.add_argument(
        '--output',
        default='foramens_by_grade.png',
        help='Output path for the figure (default: foramens_by_grade.png)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for the output figure (default: 150)'
    )
    parser.add_argument(
        '--max-examples-per-cell',
        type=int,
        default=20,
        help='Maximum number of images to display per grade/level cell (default: 20)'
    )
    return parser


def map_level_to_disc(level):
    """
    Map numeric spinal level to disc name.
    
    Parameters
    ----------
    level : int
        Numeric level (1-5 for lumbar)
    
    Returns
    -------
    str
        Disc name (e.g., 'L1-L2', 'L5-S')
    """
    level_map = {
        2: 'L1-L2',
        3: 'L2-L3',
        4: 'L3-L4',
        5: 'L4-L5',
        1: 'L5-S'
    }
    return level_map.get(int(level), f'L{level}')


def map_side_to_string(side):
    """
    Map German side names to English.
    
    Parameters
    ----------
    side : str
        Side in German (links=left, rechts=right)
    
    Returns
    -------
    str
        Side in English (left or right)
    """
    side_map = {
        'links': 'left',
        'rechts': 'right',
        'left': 'left',
        'right': 'right'
    }
    return side_map.get(str(side).lower(), str(side).lower())


def get_subject_folder_name(pid):
    """
    Get the subject folder name from PID.
    
    In metrics_output, subjects are named as sub-001, sub-002, etc.
    assuming PID maps to subject number.
    
    Parameters
    ----------
    pid : str
        Patient ID from CSV
    
    Returns
    -------
    str or list
        Possible subject folder names to check
    """
    # Try direct mapping first, then try to extract from PID
    # Since we don't have exact mapping, we'll search for folders
    return None


def find_foramen_image(metrics_folder, sub, level, side):
    """
    Find a foramen image for a given subject, level, and side.
    
    Parameters
    ----------
    metrics_folder : Path
        Path to metrics_output folder
    sub : str
        Subject ID
    level : int
        Spinal level
    side : str
        Side (left or right)
    
    Returns
    -------
    Path or None
        Path to the image file if found
    """
    metrics_path = Path(metrics_folder)
    
    # Map level to disc name
    disc_name = map_level_to_disc(level)
    side_str = map_side_to_string(side)
    
    # Look through all subject folders in metrics_output
    subject_folder = Path(glob.glob(f"{str(metrics_path)}/sub-{sub:03}*")[0]) if glob.glob(f"{str(metrics_path)}/sub-{sub:03}*") else None
    if subject_folder is None or not subject_folder.is_dir():
        print(f"Subject folder not found for sub-{sub:03} in {metrics_folder}")
        return None
    
    # Try to find: foramens_L1-L2_left_img.png
    image_path = subject_folder / 'imgs' / f'foramens_{disc_name}_{side_str}_img.png'
    if image_path.exists():
        return image_path
    else:
        return None

def read_and_organize_data(csv_path, metrics_folder):
    """
    Read CSV and organize foramen image data by grade, level, and side.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to the readout CSV
    metrics_folder : str or Path
        Path to metrics_output folder
    
    Returns
    -------
    dict
        Nested dictionary: {grade: {level: {side: [image_paths]}}}
    """
    df = pd.read_csv(csv_path)
    
    # Organize data by grade, level, side
    data_by_grade = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for idx, row in df.iterrows():
        sub = row['Lfd_Nr']
        level = row['Level']
        side = row['Side']
        grade = row['foraminal stenosis_READER 1 (Senior)']
        
        try:
            # Skip rows with missing data
            if pd.isna(grade) or pd.isna(level) or pd.isna(side):
                continue
            
            grade = int(grade)
            level = int(level)
            
            # Find the corresponding image
            image_path = find_foramen_image(metrics_folder, sub, level, side)
            
            if image_path and image_path.exists():
                data_by_grade[grade][level][side].append(str(image_path))
                print(f"Found image: Grade {grade}, Level {level}, Side {side}: {image_path.name}")

        except (KeyError, ValueError, TypeError) as e:
            print(f"Skipping row subject {sub:03} due to error: {e}")
    
    return data_by_grade


def get_max_image_size(data_by_grade):
    """
    Scan all images and find the maximum dimensions.
    
    Parameters
    ----------
    data_by_grade : dict
        Data organized by grade, level, side
    
    Returns
    -------
    tuple
        Maximum (height, width) found among all images
    """
    max_h = 0
    max_w = 0
    
    for grade_dict in data_by_grade.values():
        for level_dict in grade_dict.values():
            for side_list in level_dict.values():
                for img_path in side_list:
                    try:
                        img = PILImage.open(img_path)
                        img = img.convert('L')
                        w, h = img.size
                        max_h = max(max_h, h)
                        max_w = max(max_w, w)
                    except Exception as e:
                        continue
    
    # Add some padding for display
    return (max_h + 20, max_w + 20)


def load_image(image_path, target_size):
    """
    Load an image, convert to grayscale, and pad to target size with black background.
    
    Parameters
    ----------
    image_path : str or Path
        Path to image file
    target_size : tuple
        Target size (height, width) for the image
    
    Returns
    -------
    np.ndarray or None
        Padded grayscale image array, or None if loading fails
    """
    try:
        img = PILImage.open(image_path)
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Resize to fit within target dimensions while maintaining aspect ratio
        img.thumbnail(target_size, PILImage.Resampling.LANCZOS)
        
        # Get image dimensions
        img_array = np.array(img)
        h, w = img_array.shape
        
        # Pad to target size with black background (0)
        padded = np.zeros(target_size, dtype=np.uint8)
        
        # Calculate padding to center the image
        y_offset = (target_size[0] - h) // 2
        x_offset = (target_size[1] - w) // 2
        
        padded[y_offset:y_offset + h, x_offset:x_offset + w] = img_array
        
        return padded
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def create_grade_table(data_by_grade, max_examples=20, target_size=None):
    """
    Create a figure with subplots showing images organized by grade.
    
    Parameters
    ----------
    data_by_grade : dict
        Data organized by grade, level, side
    max_examples : int
        Maximum images per grade
    target_size : tuple, optional
        Target size (height, width) for images. If None, computed from data.
    
    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object
    """
    # Get or compute target size
    print(f"Image size: {target_size}")
    
    # Determine grades
    grades = sorted(data_by_grade.keys())
    
    # Collect all images with metadata for each grade
    images_by_grade = {}
    for grade in grades:
        grade_images = []
        for level in sorted(data_by_grade[grade].keys()):
            for side in sorted(data_by_grade[grade][level].keys()):
                images_list = data_by_grade[grade][level][side]
                for img_path in images_list:
                    grade_images.append({
                        'path': img_path,
                        'level': level,
                        'side': side
                    })
        # Shuffle and limit
        random.shuffle(grade_images)
        images_by_grade[grade] = grade_images[:max_examples]
    
    n_rows = max(len(images_by_grade[g]) for g in grades)
    n_cols = len(grades)
    
    # Calculate figure size based on image dimensions
    img_height_inches = target_size[0] / 80  # convert pixels to inches (80 dpi baseline)
    img_width_inches = target_size[1] / 80
    
    cell_width = img_width_inches * 1.3
    cell_height = img_height_inches * 1.4
    fig_width = n_cols * cell_width
    fig_height = n_rows * cell_height
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
        gridspec_kw={'hspace': 0.4, 'wspace': 0.2}
    )
    
    # Ensure axes is always 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Fill in the subplots
    for col_idx, grade in enumerate(grades):
        # Add grade label as column header
        axes[0, col_idx].text(
            0.5, 1.10, f'Grade {grade}',
            ha='center', va='bottom',
            fontsize=14, fontweight='bold',
            transform=axes[0, col_idx].transAxes
        )
        
        for row_idx in range(n_rows):
            ax = axes[row_idx, col_idx]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # Check if we have an image for this cell
            if row_idx < len(images_by_grade[grade]):
                img_data = images_by_grade[grade][row_idx]
                img_path = img_data['path']
                level = img_data['level']
                side = img_data['side']
                
                # Load image
                img_array = load_image(img_path, target_size)
                
                if img_array is not None:
                    # Display image
                    ax.imshow(img_array, cmap='gray')
                    
                    # Add level and side label below the image
                    disc_name = map_level_to_disc(level)
                    side_label = 'L' if side == 'left' else 'R'
                    ax.text(
                        0.5, -0.08,
                        f'{disc_name}-{side_label}',
                        ha='center', va='top',
                        fontsize=9, style='italic', fontweight='bold',
                        transform=ax.transAxes
                    )
    
    # Add title
    fig.suptitle(
        'Foraminal Stenosis: Examples by Grade',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    return fig


def main():
    """Main function."""
    parser = get_parser()
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    metrics_folder = Path(args.metrics_folder)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not metrics_folder.exists():
        raise FileNotFoundError(f"Metrics folder not found: {metrics_folder}")
    
    print(f"Reading CSV from: {csv_path}")
    print(f"Looking for images in: {metrics_folder}")
    
    # Read and organize data
    data_by_grade = read_and_organize_data(csv_path, metrics_folder)
    
    if not data_by_grade:
        print("No images found. Check your CSV path and metrics folder.")
        return
    
    print(f"\nFound data for {len(data_by_grade)} grades:")
    for grade in sorted(data_by_grade.keys()):
        count_by_grade = 0
        for level in sorted(data_by_grade[grade].keys()):
            for side in sorted(data_by_grade[grade][level].keys()):
                count_by_grade += len(data_by_grade[grade][level][side])
        print(f"  Grade {grade}: {count_by_grade} entries")
    
    # Create figure
    print("\nCreating figure...")
    target_size = get_max_image_size(data_by_grade)
    fig = create_grade_table(
        data_by_grade,
        max_examples=args.max_examples_per_cell,
        target_size=target_size
    )
    
    # Save figure
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(
        output_path,
        dpi=args.dpi,
        bbox_inches='tight',
        facecolor='white'
    )
    
    print(f"Figure saved to: {output_path}")
    
    plt.show()


if __name__ == '__main__':
    main()
