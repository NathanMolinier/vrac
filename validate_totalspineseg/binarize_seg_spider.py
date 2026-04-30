
import argparse
from pathlib import Path
import os
from vrac.data_management.image import Image, zeros_like
import numpy as np


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Binarize segmentation masks for Spineps evaluation."
    )
    p.add_argument(
        "input_folder",
        type=Path,
        help="Path to input folder (contains segmentation masks)",
    )
    p.add_argument(
        "--output_folder",
        type=Path,
        required=True,
        help="Output directory",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    # Load variables
    input_folder = args.input_folder
    output_folder = args.output_folder
    output_folder.mkdir(exist_ok=True)

    # Process each file in the input folder
    for file in os.listdir(input_folder):
        if file.endswith(".nii.gz"):
            input_path = input_folder / file
            output_path = output_folder / file

            # Load the image
            img = Image(str(input_path))

            # Set vertebrae values to 1 and discs values to 2
            vert_data = (10 < img.data < 46).astype(np.uint8)
            disc_data = (60 < img.data < 101).astype(np.uint8) * 2

            # Create a new Image object with the binarized data
            out_img = zeros_like(img)
            out_img.data = vert_data + disc_data

            # Save the binarized image
            out_img.save(str(output_path))


if __name__ == "__main__":
    main()