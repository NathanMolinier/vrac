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
                radius=(500, 150))
        


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
    img_input = Image(fname_input).change_orientation('LSA')
    img_seg_sc = Image(fname_seg_sc).change_orientation('LSA')
    img_seg_lesion = Image(fname_seg_lesion).change_orientation('LSA') if fname_seg_lesion else None

    # Resample images
    S_I_resolution = 1
    R_L_resolution = 5
    A_P_resolution = 1
    logger.info('Resample images to %fx%fx%f vox', R_L_resolution, S_I_resolution, A_P_resolution)
    img_input = resample_nib(
        image=img_input,
        new_size=[R_L_resolution, S_I_resolution, A_P_resolution],
        new_size_type='mm',
        interpolation='spline',
    )

    logger.info('Resample images to %fx%fx%f vox', R_L_resolution, S_I_resolution, A_P_resolution)
    img_seg_sc = resample_nib(
        image=img_seg_sc,
        new_size=[R_L_resolution, S_I_resolution, A_P_resolution],
        new_size_type='mm',
        interpolation='nn',
    )

    img_seg_lesion = resample_nib(
        image=img_seg_lesion,
        new_size=[R_L_resolution, S_I_resolution, A_P_resolution],
        new_size_type='mm',
        interpolation='nn',
    ) if fname_seg_lesion else None
    #   Choosing a fixed resolution allows us to crop the image around the spinal cord at a fixed radius that matches the chosen resolution,
    #   while also handling anisotropic images (so that they display correctly on an isotropic grid).
    # - However, we cannot apply resampling here because rootlets labels are often small (~1vox wide), and so resampling might
    #   corrupt the labels and cause them to be displayed unfaithfully.
    # - So, instead of resampling the image to fit the default crop radius, we scale the crop radius to suit the original resolution.
    p_ratio = tuple([1, 1])
    radius = tuple(int(r * p) for r, p in zip(radius, p_ratio))
    # - One problem with this, however, is that if the crop radius ends up being smaller than the default, the QC will in turn be smaller as well.
    #   So, to ensure that the QC is still readable, we scale up by an integer factor whenever the p_ratio is < 1
    scale = int(math.ceil(1 / max(p_ratio)))  # e.g. 0.8mm human => p_ratio == 0.6/0.8 == 0.75; scale == 1/p_ratio == 1/0.75 == 1.33 => 2x scale
    # - One other problem is that for anisotropic images, the aspect ratio won't be 1:1 between width/height.
    #   So, we use `aspect` to adjust the image via imshow, and `radius` to know where to place the text in x/y coords
    aspect = p_ratio[1] / p_ratio[0]

    # get available labels
    seg = np.rint(np.ma.masked_where(img_seg_sc.data < 1, img_seg_sc.data))
    labels = np.unique(seg[np.where(~seg.mask)]).astype(int)
    colormaps = [mpl_colors.ListedColormap(assign_label_colors_by_groups(labels))]
    for i, image in enumerate([img_seg_sc, img_seg_lesion]):
        if not image:
            continue
        img, seg = mosaic(img_input, seg, radius, scale)
        # Generate the first QC report image
        img = equalize_histogram(img)
        seg = np.ma.masked_less_equal(seg, 0)
        seg.set_fill_value(0)
        # For QC reports, axial mosaics will often have smaller height than width
        # (e.g. WxH = 20x3 slice images). So, we want to reduce the fig height to match this.
        # `size_fig` is in inches. So, dpi=300 --> 1500px, dpi=100 --> 500px, etc.
        size_fig = [5, 5 * (img.shape[0] / img.shape[1]) * aspect]

        # Generate the second QC report image
        fig = mpl_figure.Figure()
        fig.set_size_inches(*size_fig, forward=True)
        mpl_backend_agg.FigureCanvasAgg(fig)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(seg,
                  cmap=colormaps[i],
                  norm=None,
                  alpha=1.0,
                  interpolation='none',
                  aspect=aspect)
        if outline:
            # linewidth 0.5 is too thick, 0.25 is too thin
            plot_outlines(seg, ax=ax, facecolor='none', edgecolor='black', linewidth=0.1)
        # add_segmentation_labels(ax, seg, colors=colormaps[i].colors, radius=tuple(r*scale for r in radius))

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    img_path = str(imgs_to_generate['path_overlay_img'])
    logger.debug('Save image %s', img_path)
    fig.savefig(img_path, format='png', transparent=True, dpi=300)

    fig = mpl_figure.Figure()
    fig.set_size_inches(*size_fig, forward=True)
    mpl_backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(img, cmap='gray', interpolation='none', aspect=aspect)
    add_orientation_labels(ax, radius=tuple(r*scale for r in radius), letters=['S', 'I', 'P', 'A'])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    img_path = str(imgs_to_generate['path_background_img'])
    logger.debug('Save image %s', img_path)
    fig.savefig(img_path, format='png', transparent=True, dpi=300)


def mosaic(img: Image, seg: Image, radius: tuple[int, int] = (15, 15), scale: int = 1):
    """
    Arrange the slices of `img` into a grid of images.

    Each slice is centered at the approximate coordinates given in `centers`,
    and cropped to `radius` pixels in each direction (horizontal, vertical).

    If `img` has N slices, then `centers` should have shape (N, 2).
    """
    # Fit as many slices as possible in each row of 600 pixels
    num_col = math.floor(12000 / (2*radius[0]*scale))

    # Center and crop each axial slice
    cropped_img = []
    cropped_seg = []
    for sliceImg, sliceSeg in zip(img.data, seg.data):
        if sliceSeg.any():
            # Add a margin before cropping, in case the center is still too close to one of the edges
            # Also, use Kronecker product to scale each block in multiples
            cropped_img.append(np.kron(np.pad(sliceImg, [[r] for r in radius])[
                radius[0]:radius[0] + 2*radius[0],
                radius[1]:radius[1] + 2*radius[1],
            ], np.ones((scale, scale))))
            cropped_seg.append(np.kron(np.pad(sliceSeg, [[r] for r in radius])[
                radius[0]:radius[0] + 2*radius[0],
                radius[1]:radius[1] + 2*radius[1],
            ], np.ones((scale, scale))))

    # Pad the list with empty arrays, to get complete rows of num_col
    empty = np.zeros((2*radius[0]*scale, 2*radius[1]*scale))
    cropped_img.extend([empty] * (-len(cropped_img) % num_col))
    cropped_seg.extend([empty] * (-len(cropped_seg) % num_col))

    # Arrange the images into a grid
    out_img = np.block([cropped_img[i:i+num_col] for i in range(0, len(cropped_img), num_col)])
    out_seg = np.block([cropped_seg[i:i+num_col] for i in range(0, len(cropped_seg), num_col)])
    return out_img, out_seg


if __name__ == '__main__':
    main()