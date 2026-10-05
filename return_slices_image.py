from vrac.data_management.image import Image
import cv2
import numpy as np
import os

def main():
    img_path = "/Users/nathan/Desktop/test-spineps/sub-amu03_T2w.nii.gz"
    out_folder = "/Users/nathan/Desktop/test-spineps/slices"

    os.makedirs(out_folder, exist_ok=True)

    img = Image(img_path).change_orientation('RSP')

    shape = img.data.shape

    ma = np.percentile(img.data, 95)
    mi = np.percentile(img.data, 5)

    img_norm = normalize_image(img.data, mi, ma)

    for slice_idx in range(0, shape[0], 5):
        sl = img_norm[slice_idx, :, :]
        sl = sl.astype(np.uint8)

        sl = cv2.cvtColor(sl, cv2.COLOR_GRAY2BGR)
        h = sl.shape[0]
        w = sl.shape[1]
        cv2.putText(sl, f"slice {slice_idx}", (w - 80, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(f"/Users/nathan/Desktop/test-spineps/slices/slice_{slice_idx:03d}.jpg", sl)

def normalize_image(img, mi, ma):
    im = (img - mi) / (ma - mi)
    im = np.clip(im, 0, 1)
    return (im * 255).astype(np.uint8)

if __name__ == "__main__":
    main()