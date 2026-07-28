from vrac.data_management.image import Image
import os
import numpy as np

def main():
    # Set paths
    spineps_folder = '/home/ge.polymtl.ca/p118739/data/datasets/article-totalspineseg/spider-comparison/spineps'
    ground_truth_folder = '/home/ge.polymtl.ca/p118739/data/datasets/article-totalspineseg/spider-comparison/gt'

    for file in os.listdir(spineps_folder):
        if file.endswith(".nii.gz"):
            spineps_path = os.path.join(spineps_folder, file)
            gt_path = os.path.join(ground_truth_folder, file)

            # Load images
            spineps_img = Image(spineps_path)
            gt_img = Image(gt_path)

            # Compare labels
            gt_labels = set(np.unique(gt_img.data))

            for label in gt_labels:
                mask = (gt_img.data == label)
                spineps_values = spineps_img.data[mask]

                print()

                
            print(f"File: {file}")
            print(f"SpinePS Labels: {spineps_labels}")
            print(f"Ground Truth Labels: {gt_labels}")
            print("-" * 40)
    


if __name__ == "__main__":
    main()