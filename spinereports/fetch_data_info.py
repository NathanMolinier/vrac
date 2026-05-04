
import argparse
import json
import nibabel as nib
import pandas as pd
from pathlib import Path
from vrac.data_management.image import Image

def get_parser():
    """
    Creates an argument parser for the script.
    """
    parser = argparse.ArgumentParser(description='Fetch information from BIDS dataset JSON sidecars.')
    parser.add_argument('--path-input', required=True, help='If path endswith .json, will be treated as config else treated as an input folder path to a BIDS dataset.')
    parser.add_argument('--file-pattern', default='*', help='Pattern to match files (e.g., "*_acq-sag_T2w*").')
    parser.add_argument('--output-csv', help='Path to save the output CSV file.')
    return parser

def main():
    """
    Main function to fetch and process the data.
    """
    parser = get_parser()
    args = parser.parse_args()

    dataset_path = Path(args.path_input)
    if dataset_path.suffix == '.json':
        with open(dataset_path, 'r') as f:
            config = json.load(f)
        split = 'TRAINING'
        dataset_path = Path(config['DATASETS_PATH'])
        image_files = [dataset_path /d['IMAGE'] for d in config[split]]
    else:
        file_pattern = args.file_pattern
        image_files = sorted(list(dataset_path.glob(f'**/{file_pattern}.nii.gz')))
    
    output_csv = args.output_csv
    
    data_list = []

    for image_file in image_files:
        json_file = image_file.with_name(image_file.name.replace('.nii.gz', '.json'))
        
        if not json_file.exists():
            print(f"JSON sidecar not found for {image_file}, skipping.")
            continue

        with open(json_file, 'r') as f:
            json_data = json.load(f)

        # Extracting information
        sequence = json_data.get('SeriesDescription', json_data.get('ProtocolName', json_data.get('SequenceName', 'N/A')))
        manufacturer = json_data.get('Manufacturer', 'N/A')
        field_strength = json_data.get('MagneticFieldStrength', 'N/A')

        # Get resolution from nifti header
        try:
            img = Image(str(image_file)).change_orientation('RPI')
            resolution = [str(round(val, 2)) for val in img.hdr.get_zooms()[:3]]
        except Exception as e:
            print(f"Could not read resolution for {image_file}: {e}")
            resolution = "N/A"

        data_list.append({
            'filename': image_file.name,
            'sequence': sequence,
            'manufacturer': manufacturer,
            'field_strength': field_strength,
            'rx': resolution[0] if resolution != "N/A" else "N/A",
            'ry': resolution[1] if resolution != "N/A" else "N/A",
            'rz': resolution[2] if resolution != "N/A" else "N/A",
        })

    df = pd.DataFrame(data_list)

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Data saved to {output_csv}")
    else:
        print(df.to_string())

if __name__ == '__main__':
    main()
