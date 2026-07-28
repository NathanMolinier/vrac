from vrac.data_management.image import Image, zeros_like
import os
import numpy as np

def main():
    # Set paths
    spineps_folder = '/home/ge.polymtl.ca/p118739/data/datasets/article-totalspineseg/spineps-spider/spineps-pred-label'
    ground_truth_folder = '/home/ge.polymtl.ca/p118739/data/datasets/article-totalspineseg/spineps-spider/gt-spider'
    output_folder = '/home/ge.polymtl.ca/p118739/data/datasets/article-totalspineseg/spineps-spider/spineps-pred-newlabels'

    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(spineps_folder):
        if file.endswith(".nii.gz"):
            spineps_path = os.path.join(spineps_folder, file)
            gt_path = os.path.join(ground_truth_folder, file)
            output_path = os.path.join(output_folder, file)

            # Load images
            spineps_img = Image(spineps_path).change_orientation('RPI')
            gt_img = Image(gt_path).change_orientation('RPI')
            out_img = zeros_like(gt_img)

            # Compare labels
            gt_labels = set([label for label in np.unique(gt_img.data) if label > 10])

            for label in gt_labels:
                mask = (gt_img.data == label)
                spineps_values = spineps_img.data[mask]
                unique_spineps_values = [value for value in np.unique(spineps_values) if value != 0]
                count_pixels = [np.sum(spineps_values == value) for value in unique_spineps_values]
                max_idx = np.argmax(count_pixels)
                spineps_label = unique_spineps_values[max_idx]
                spineps_mask = (spineps_img.data == spineps_label)
                out_img.data[spineps_mask] = label

            # Save the new image
            out_img.save(output_path)
    

if __name__ == "__main__":
    main()