import argparse
import math
from pathlib import Path
from typing import Optional, Sequence
import logging
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import center_of_mass

from spinalcordtoolbox.centerline.core import get_centerline, ParamCenterline
from spinalcordtoolbox.resampling import resample_nib
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils.sys import __version__, LazyLoader, list2cmdline
from spinalcordtoolbox.reports.qc2 import inf_nan_fill, equalize_histogram, add_orientation_labels, assign_label_colors_by_groups, plot_outlines, add_segmentation_labels, create_qc_entry

pd = LazyLoader("pd", globals(), "pandas")
mpl_plt = LazyLoader("mpl_plt", globals(), "matplotlib.pyplot")
mpl_figure = LazyLoader("mpl_figure", globals(), "matplotlib.figure")
mpl_axes = LazyLoader("mpl_axes", globals(), "matplotlib.axes")
mpl_cm = LazyLoader("mpl_cm", globals(), "matplotlib.cm")
mpl_colors = LazyLoader("mpl_colors", globals(), "matplotlib.colors")
mpl_backend_agg = LazyLoader("mpl_backend_agg", globals(), "matplotlib.backends.backend_agg")
mpl_patheffects = LazyLoader("mpl_patheffects", globals(), "matplotlib.patheffects")
mpl_collections = LazyLoader("mpl_collections", globals(), "matplotlib.collections")


logger = logging.getLogger(__name__)


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Refine segmentation using nnInteractive.')
    parser.add_argument('--img', '-i', required=True, help='Path to the input image (Required)')
    parser.add_argument('--seg', '-s', required=True, help='Path to the input segmentation (Required)')
    parser.add_argument('--ofolder', '-o', required=True, help='Path to the qc folder (Required)')
    return parser


def main():
    # Load parser
    parser = get_parser()
    args = parser.parse_args()

    # Load variables
    img_path = args.img
    seg_path = args.seg
    ofolder = args.ofolder

    # Run SCT QC
    command = 'sct_deepseg'
    cmdline = [command, 'totalspineseg']
    with create_qc_entry(
        path_input=Path(img_path).absolute(),
        path_qc=Path(ofolder),
        command=command,
        cmdline=list2cmdline(cmdline),
        plane='Axial',
        dataset=None,
        subject=None,
    ) as imgs_to_generate:
        sct_deepseg_spinal_rootlets_t2w(
                imgs_to_generate, img_path, seg_path, None, 'human',
                radius=(80, 50))
        


def sct_deepseg_spinal_rootlets_t2w(
    imgs_to_generate: dict[str, Path],
    fname_input: str,
    fname_seg_sc: str,
    fname_seg_lesion: Optional[str],
    species: str,
    radius: Sequence[int],
    outline: bool = True
):
    """
    Generate a QC report for `sct_deepseg rootlets_t2`.

    This refactor is based off of the `listed_seg` method in qc.py, adapted to support multiple images.
    """
    # Axial orientation, switch between one anat image and 1-2 seg images
    # FIXME: This code is more or less duplicated with the 'sct_register_multimodal' report, because both reports
    #        use the old qc.py method "_make_QC_image_for_3d_volumes" for generating the background img.

    # Load the input images
    img_input = Image(fname_input).change_orientation('SAL')
    img_seg_sc = Image(fname_seg_sc).change_orientation('SAL')
    img_seg_lesion = Image(fname_seg_lesion).change_orientation('SAL') if fname_seg_lesion else None

    # - Normally, we would apply isotropic resampling to the image to a specific mm resolution (based on the species).
    p_resample = {'human': 0.6, 'mouse': 0.1}[species]

    # Resample images
    S_I_resolution = img_input.dim[4]*3
    logger.info('Resample images to %fx%fx%f vox', S_I_resolution, p_resample, p_resample)
    img_input = resample_nib(
        image=img_input,
        new_size=[S_I_resolution, p_resample, p_resample],
        new_size_type='mm',
        interpolation='spline',
    )

    logger.info('Resample images to %fx%fx%f vox', S_I_resolution, p_resample, p_resample)
    img_seg_sc = resample_nib(
        image=img_seg_sc,
        new_size=[S_I_resolution, p_resample, p_resample],
        new_size_type='mm',
        interpolation='nn',
    )

    img_seg_lesion = resample_nib(
        image=img_seg_lesion,
        new_size=[S_I_resolution, p_resample, p_resample],
        new_size_type='mm',
        interpolation='nn',
    ) if fname_seg_lesion else None
    #   Choosing a fixed resolution allows us to crop the image around the spinal cord at a fixed radius that matches the chosen resolution,
    #   while also handling anisotropic images (so that they display correctly on an isotropic grid).
    # - However, we cannot apply resampling here because rootlets labels are often small (~1vox wide), and so resampling might
    #   corrupt the labels and cause them to be displayed unfaithfully.
    # - So, instead of resampling the image to fit the default crop radius, we scale the crop radius to suit the original resolution.
    p_original = (img_seg_sc.dim[5], img_seg_sc.dim[6])  # Image may be anisotropic, so use both resolutions (H,W)
    p_ratio = tuple(p_resample / p for p in p_original)
    radius = tuple(int(r * p) for r, p in zip(radius, p_ratio))
    # - One problem with this, however, is that if the crop radius ends up being smaller than the default, the QC will in turn be smaller as well.
    #   So, to ensure that the QC is still readable, we scale up by an integer factor whenever the p_ratio is < 1
    scale = int(math.ceil(1 / max(p_ratio)))  # e.g. 0.8mm human => p_ratio == 0.6/0.8 == 0.75; scale == 1/p_ratio == 1/0.75 == 1.33 => 2x scale
    # - One other problem is that for anisotropic images, the aspect ratio won't be 1:1 between width/height.
    #   So, we use `aspect` to adjust the image via imshow, and `radius` to know where to place the text in x/y coords
    aspect = p_ratio[1] / p_ratio[0]

    # Each slice is centered on the segmentation
    logger.info('Find the center of each slice')
    centerline_param = ParamCenterline(algo_fitting="optic", contrast="t2")
    img_centerline, _, _, _ = get_centerline(img_input, param=centerline_param)
    centers = np.array([center_of_mass(slice) for slice in img_centerline.data])
    inf_nan_fill(centers[:, 0])
    inf_nan_fill(centers[:, 1])

    # Generate the first QC report image
    img = equalize_histogram(mosaic(img_input, centers, radius, scale))

    # For QC reports, axial mosaics will often have smaller height than width
    # (e.g. WxH = 20x3 slice images). So, we want to reduce the fig height to match this.
    # `size_fig` is in inches. So, dpi=300 --> 1500px, dpi=100 --> 500px, etc.
    size_fig = [5, 5 * (img.shape[0] / img.shape[1]) * aspect]

    fig = mpl_figure.Figure()
    fig.set_size_inches(*size_fig, forward=True)
    mpl_backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(img, cmap='gray', interpolation='none', aspect=aspect)
    add_orientation_labels(ax, radius=tuple(r*scale for r in radius))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    img_path = str(imgs_to_generate['path_background_img'])
    logger.debug('Save image %s', img_path)
    fig.savefig(img_path, format='png', transparent=True, dpi=300)

    # Generate the second QC report image
    fig = mpl_figure.Figure()
    fig.set_size_inches(*size_fig, forward=True)
    mpl_backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    # get available labels
    img = np.rint(np.ma.masked_where(img_seg_sc.data < 1, img_seg_sc.data))
    labels = np.unique(img[np.where(~img.mask)]).astype(int)
    colormaps = [mpl_colors.ListedColormap(assign_label_colors_by_groups(labels))]
    for i, image in enumerate([img_seg_sc, img_seg_lesion]):
        if not image:
            continue
        img = mosaic(image, centers, radius, scale)
        img = np.ma.masked_less_equal(img, 0)
        img.set_fill_value(0)
        ax.imshow(img,
                  cmap=colormaps[i],
                  norm=None,
                  alpha=1.0,
                  interpolation='none',
                  aspect=aspect)
        if outline:
            # linewidth 0.5 is too thick, 0.25 is too thin
            plot_outlines(img, ax=ax, facecolor='none', edgecolor='black', linewidth=0.3)
        add_segmentation_labels(ax, img, colors=colormaps[i].colors, radius=tuple(r*scale for r in radius))

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    img_path = str(imgs_to_generate['path_overlay_img'])
    logger.debug('Save image %s', img_path)
    fig.savefig(img_path, format='png', transparent=True, dpi=300)


def mosaic(img: Image, centers: np.ndarray, radius: tuple[int, int] = (15, 15), scale: int = 1):
    """
    Arrange the slices of `img` into a grid of images.

    Each slice is centered at the approximate coordinates given in `centers`,
    and cropped to `radius` pixels in each direction (horizontal, vertical).

    If `img` has N slices, then `centers` should have shape (N, 2).
    """
    # Fit as many slices as possible in each row of 600 pixels
    num_col = math.floor(4000 / (2*radius[0]*scale))

    # Center and crop each axial slice
    cropped = []
    for center, slice in zip(centers.astype(int), img.data):
        # If the `center` coordinate is close to the edge, then move it away from the edge to capture more of the image
        # In other words, make sure the `center` coordinate is at least `radius` pixels away from the edge
        for i in [0, 1]:
            center[i] = min(slice.shape[i] - radius[i], center[i])  # Check far edge first
            center[i] = max(radius[i],                  center[i])  # Then check 0 edge last
        # Add a margin before cropping, in case the center is still too close to one of the edges
        # Also, use Kronecker product to scale each block in multiples
        cropped.append(np.kron(np.pad(slice, [[r] for r in radius])[
            center[0]:center[0] + 2*radius[0],
            center[1]:center[1] + 2*radius[1],
        ], np.ones((scale, scale))))

    # Pad the list with empty arrays, to get complete rows of num_col
    empty = np.zeros((2*radius[0]*scale, 2*radius[1]*scale))
    cropped.extend([empty] * (-len(cropped) % num_col))

    # Arrange the images into a grid
    return np.block([cropped[i:i+num_col] for i in range(0, len(cropped), num_col)])


if __name__ == '__main__':
    main()