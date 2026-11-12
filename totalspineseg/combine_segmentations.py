import os
from vrac.data_management.image import Image

def main():
    canal_seg_path = "/home/ge.polymtl.ca/p118739/data/totalspineseg_data/sc-crop/labels_canal"
    cord_seg_path = "/home/ge.polymtl.ca/p118739/data/totalspineseg_data/sc-crop/labels_cord"
    spine_seg_path = "/home/ge.polymtl.ca/p118739/data/totalspineseg_data/sc-crop/labels_spine"

    for canal_name in os.listdir(canal_seg_path):
        if not canal_name.endswith(".nii.gz"):
            raise ValueError(f"Unexpected file format: {canal_name}")
        stem = canal_name.replace("_label-canal_seg", "")
        canal_path = os.path.join(canal_seg_path, canal_name)
        cord_path = os.path.join(cord_seg_path, stem.replace(".nii.gz", "_label-cord_seg.nii.gz"))
        spine_path = os.path.join(spine_seg_path, stem.replace(".nii.gz", "_label-spine_dseg.nii.gz"))

        # Load images
        canal = Image(canal_path).change_orientation('LPI')
        cord = Image(cord_path).change_orientation('LPI')
        spine = Image(spine_path).change_orientation('LPI')

        if not canal.data.shape == cord.data.shape and canal.data.shape == spine.data.shape:
            raise ValueError(f"Shape error for image: {stem}")

        # Combine the segmentations
        spine.data[canal.data > 0] = 2  # Canal
        spine.data[cord.data > 0] = 1  # Cord

        # Save the combined segmentation
        output_path = os.path.join("/home/ge.polymtl.ca/p118739/data/totalspineseg_data/sc-crop/labels", stem)
        spine.save(output_path)
        print(f"Saved combined segmentation to {output_path}")

if __name__ == "__main__":
    main()