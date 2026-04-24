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
        default=3,
        help='Maximum number of images to display per grade/level cell (default: 3)'
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
    if not subject_folder is None and not subject_folder.is_dir():
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
                key = (level, side)
                data_by_grade[grade][level][side].append(str(image_path))
                print(f"Found image: Grade {grade}, Level {level}, Side {side}: {image_path.name}")
        
        except (KeyError, ValueError, TypeError) as e:
            print(f"Skipping row subject {sub:03} due to error: {e}")
    
    return data_by_grade


def load_image(image_path, max_width=250, max_height=250):
    """
    Load an image and resize if necessary.
    
    Parameters
    ----------
    image_path : str or Path
        Path to image file
    max_width : int
        Maximum width in pixels
    max_height : int
        Maximum height in pixels
    
    Returns
    -------
    np.ndarray
        Image array
    """
    try:
        img = PILImage.open(image_path)
        # Resize to fit within max dimensions while maintaining aspect ratio
        img.thumbnail((max_width, max_height), PILImage.Resampling.LANCZOS)
        return np.array(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def create_grade_table(data_by_grade, grades=None, levels=None, sides=None, 
                       max_examples=3, img_size=(200, 200)):
    """
    Create a figure with subplots showing images organized by grade and level/side.
    
    Parameters
    ----------
    data_by_grade : dict
        Data organized by grade, level, side
    grades : list, optional
        List of grades to display (default: all grades found)
    levels : list, optional
        List of levels to display (default: all levels found)
    sides : list, optional
        List of sides to display (default: all sides found)
    max_examples : int
        Maximum images per cell
    img_size : tuple
        Size of individual images (height, width)
    
    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object
    """
    # Determine grades, levels, sides to display
    if grades is None:
        grades = sorted(data_by_grade.keys())
    if levels is None:
        all_levels = set()
        for level_dict in data_by_grade.values():
            all_levels.update(level_dict.keys())
        levels = sorted(all_levels)
    if sides is None:
        all_sides = set()
        for level_dict in data_by_grade.values():
            for side_set in level_dict.values():
                all_sides.update(side_set.keys())
        sides = sorted(all_sides, key=lambda x: ('left', 'right').index(x) if x in ('left', 'right') else 0)
    
    # Create figure
    n_row_groups = len(levels) * len(sides)
    n_cols = len(grades)
    
    # Calculate figure size
    cell_width = 3
    cell_height = 2.5
    fig_width = n_cols * cell_width
    fig_height = n_row_groups * cell_height
    
    fig, axes = plt.subplots(
        n_row_groups, n_cols,
        figsize=(fig_width, fig_height),
        gridspec_kw={'hspace': 0.3, 'wspace': 0.2}
    )
    
    # Ensure axes is always 2D
    if n_row_groups == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Fill in the subplots
    for col_idx, grade in enumerate(grades):
        # Add grade label as column header
        axes[0, col_idx].text(
            0.5, 1.15, f'Grade {grade}',
            ha='center', va='bottom',
            fontsize=14, fontweight='bold',
            transform=axes[0, col_idx].transAxes
        )
        
        for row_idx, (level, side) in enumerate([(l, s) for l in levels for s in sides]):
            ax = axes[row_idx, col_idx]
            
            # Add row label (only on first column)
            if col_idx == 0:
                disc_name = map_level_to_disc(level)
                side_label = 'L' if side == 'left' else 'R'
                ax.text(
                    -0.15, 0.5, f'{disc_name}\n{side_label}',
                    ha='right', va='center',
                    fontsize=11, fontweight='bold',
                    transform=ax.transAxes
                )
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # Get images for this cell
            try:
                images_list = data_by_grade[grade][level][side][:max_examples]
            except KeyError:
                images_list = []
            
            if images_list:
                # Display images
                n_images = len(images_list)
                
                for img_idx, img_path in enumerate(images_list):
                    img_array = load_image(img_path, max_width=180, max_height=180)
                    
                    if img_array is not None:
                        # Calculate position for this image
                        x_offset = (img_idx % 2) * 0.5
                        y_offset = 1.0 - (img_idx // 2) * 0.6
                        
                        # Create inset axis for the image
                        ax_inset = ax.inset_axes(
                            [x_offset + 0.05, y_offset - 0.55, 0.45, 0.5],
                            transform=ax.transAxes
                        )
                        ax_inset.imshow(img_array)
                        ax_inset.axis('off')
            else:
                # Display "No data" message
                ax.text(
                    0.5, 0.5, 'No data',
                    ha='center', va='center',
                    fontsize=10, style='italic', color='gray',
                    transform=ax.transAxes
                )
    
    # Add title
    fig.suptitle(
        'Foraminal Stenosis: Examples by Grade and Level',
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
        print(f"  Grade {grade}: {sum(len(v) for v in data_by_grade[grade].values())} entries")
    
    # Create figure
    print("\nCreating figure...")
    fig = create_grade_table(
        data_by_grade,
        max_examples=args.max_examples_per_cell
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
