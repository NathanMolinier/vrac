from vrac.data_management.image import Image, zeros_like
import os
import numpy as np

def main():
    # Set paths
    spineps_folder = '/home/ge.polymtl.ca/p118739/data/datasets/article-totalspineseg/totalsegXtotalspinesegXspineps/spineps/vert-disc/spineps-pred'
    ground_truth_folder = '/home/ge.polymtl.ca/p118739/data/datasets/article-totalspineseg/totalsegXtotalspinesegXspineps/spineps/vert-disc/gt-spider'
    output_folder = '/home/ge.polymtl.ca/p118739/data/datasets/article-totalspineseg/totalsegXtotalspinesegXspineps/spineps/vert-disc/spineps-pred-newlabel'

    for file in os.listdir(spineps_folder):
        if file.endswith(".nii.gz"):
            spineps_path = os.path.join(spineps_folder, file)
            gt_path = os.path.join(ground_truth_folder, file)
            output_path = os.path.join(output_folder, file)

            # Load images
            spineps_img = Image(spineps_path)
            gt_img = Image(gt_path)
            out_img = zeros_like(gt_img)

            # Compare labels
            gt_labels = set(np.unique(gt_img.data))

            for label in gt_labels:
                mask = (gt_img.data == label)
                spineps_values = spineps_img.data[mask]

                print(label)
                print(spineps_values)
                break
            break

                
            print(f"File: {file}")
            print(f"SpinePS Labels: {spineps_labels}")
            print(f"Ground Truth Labels: {gt_labels}")
            print("-" * 40)
    


if __name__ == "__main__":
    main()