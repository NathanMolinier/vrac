import torch
import torch.nn.functional as F

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform, BasicTransform
from typing import Tuple

import numpy as np
import torchio as tio
import gryds
import scipy.ndimage as ndi
from scipy.stats import norm

### Contrast transform (Laplace, Gamma, Histogram Equalization, Log, Sqrt, Exp, Sin, Sig, Inverse)

class ConvTransform(ImageOnlyTransform):
    '''
    Based on https://github.com/spinalcordtoolbox/disc-labeling-playground/blob/main/src/ply/models/transform.py
    '''
    def __init__(self, kernel_type: str = 'Laplace', absolute: bool = False):
        super().__init__()
        if kernel_type not in  ["Laplace","Scharr"]:
            raise NotImplementedError('Currently only "Laplace" and "Scharr" are supported.')
        else:
            self.kernel_type = kernel_type
        self.absolute = absolute

    def get_parameters(self, **data_dict) -> dict:
        spatial_dims = len(data_dict['image'].shape) - 1
        if spatial_dims == 2:
            if self.kernel_type == "Laplace":
                kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
            elif self.kernel_type == "Scharr":
                kernel_x = torch.tensor([[-3, 0, 3], [-10, 0, -10], [-3, 0, 3]], dtype=torch.float32)
                kernel_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32)
                kernel = [kernel_x, kernel_y]
        elif spatial_dims == 3:
            if self.kernel_type == "Laplace":
                kernel = -1.0 * torch.ones(3, 3, 3, dtype=torch.float32)
                kernel[1, 1, 1] = 26.0
            elif self.kernel_type == "Scharr":
                kernel_x = torch.tensor([[[  9,    0,    -9],
                                          [ 30,    0,   -30],
                                          [  9,    0,    -9]],

                                          [[ 30,    0,   -30],
                                           [100,    0,  -100],
                                           [ 30,    0,   -30]],

                                          [[  9,    0,    -9],
                                           [ 30,    0,   -30],
                                           [  9,    0,    -9]]], dtype=torch.float32)
                
                kernel_y = torch.tensor([[[    9,   30,    9],
                                          [    0,    0,    0],
                                          [   -9,  -30,   -9]],

                                         [[  30,  100,   30],
                                          [   0,    0,    0],
                                          [ -30, -100,  -30]],

                                         [[   9,   30,    9],
                                          [   0,    0,    0],
                                          [  -9,  -30,   -9]]], dtype=torch.float32)
                
                kernel_z = torch.tensor([[[   9,   30,   9],
                                          [  30,  100,  30],
                                          [   9,   30,   9]],

                                         [[   0,    0,   0],
                                          [   0,    0,   0],
                                          [   0,    0,   0]],

                                         [[   -9,  -30,  -9],
                                          [  -30, -100, -30],
                                          [   -9,  -30,  -9]]], dtype=torch.float32)
                kernel = [kernel_x, kernel_y, kernel_z]
        else:
            raise ValueError(f"{self.__class__} can only handle 2D or 3D images.")

        return {
            'kernel_type': self.kernel_type,
            'kernel': kernel,
            'absolute': self.absolute
        }
    
    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        '''
        We expect (C, X, Y) or (C, X, Y, Z) shaped inputs for image and seg
        '''
        for c in range(img.shape[0]):
            orig_mean = torch.mean(img[c])
            orig_std = torch.std(img[c])
            img_ = img[c].unsqueeze(0).unsqueeze(0)  # adds temp batch and channel dim
            if params['kernel_type'] == 'Laplace':
                tot_ = apply_filter(img_, params['kernel'])
            elif params['kernel_type'] == 'Scharr':
                tot_ = torch.zeros_like(img_)
                for kernel in params['kernel']:
                    if params['absolute']:
                        tot_ += torch.abs(apply_filter(img_, kernel))
                    else:
                        tot_ += apply_filter(img_, kernel)
            mean = torch.mean(tot_[0,0])
            std = torch.std(tot_[0,0])
            img[c] = (tot_[0,0] - mean)/torch.clamp(std, min=1e-7)
            img[c] = img[c]*orig_std + orig_mean # return to original distribution
        return img


class HistogramEqualTransform(ImageOnlyTransform):
    def __init__(self):
        super().__init__()
    
    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        for c in range(img.shape[0]): 
            orig_mean = torch.mean(img[c])
            orig_std = torch.std(img[c])
            img_min, img_max = img[c].min(), img[c].max()

            # Flatten the image and compute the histogram
            img_flattened = img[c].flatten().to(torch.float32)
            hist, bins = torch.histogram(img_flattened, bins=256)

            # Compute bin edges
            bin_edges = torch.linspace(img_min, img_max, steps=257)  # 256 bins -> 257 edges

            # Compute the normalized cumulative distribution function (CDF)
            cdf = hist.cumsum(dim=0)
            cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())  # Normalize to [0,1]
            cdf = cdf * (img_max - img_min) + img_min  # Scale back to image range

            # Perform histogram equalization
            indices = torch.searchsorted(bin_edges[:-1], img_flattened)
            img_eq = torch.index_select(cdf, dim=0, index=torch.clamp(indices, 0, 255))
            img[c] = img_eq.reshape(img[c].shape)

            # Return to original distribution
            mean = torch.mean(img[c])
            std = torch.std(img[c])
            img[c] = (img[c] - mean)/torch.clamp(std, min=1e-7)
            img[c] = img[c]*orig_std + orig_mean
        return img

### Redistribute segmentation values
    

### Artifacts augmentation (Motion, Ghosting, Spike, Bias Field, Blur, Noise)


### Spatial augmentation (Flip, BSpline, Affine, Elastic)


### Anisotropy augmentation


class GammaTransform(ImageOnlyTransform):
    def __init__(self, gamma: RandomScalar, p_invert_image: float, synchronize_channels: bool, p_per_channel: float,
                 p_retain_stats: float):
        super().__init__()
        self.gamma = gamma
        self.p_invert_image = p_invert_image
        self.synchronize_channels = synchronize_channels
        self.p_per_channel = p_per_channel
        self.p_retain_stats = p_retain_stats

    def get_parameters(self, **data_dict) -> dict:
        shape = data_dict['image'].shape
        apply_to_channel = torch.where(torch.rand(shape[0]) < self.p_per_channel)[0]
        retain_stats = torch.rand(len(apply_to_channel)) < self.p_retain_stats
        invert_image = torch.rand(len(apply_to_channel)) < self.p_invert_image

        if self.synchronize_channels:
            gamma = torch.Tensor([sample_scalar(self.gamma, image=data_dict['image'], channel=None)] * len(apply_to_channel))
        else:
            gamma = torch.Tensor([sample_scalar(self.gamma, image=data_dict['image'], channel=c) for c in apply_to_channel])
        return {
            'apply_to_channel': apply_to_channel,
            'retain_stats': retain_stats,
            'invert_image': invert_image,
            'gamma': gamma
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        for c, r, i, g in zip(params['apply_to_channel'], params['retain_stats'], params['invert_image'], params['gamma']):
            if i:
                img[c] *= -1
            if r:
                # std_mean is for whatever reason slower than doing the computations separately!?
                # std, mean = torch.std_mean(img[c])
                mean = torch.mean(img[c])
                std = torch.std(img[c])
            minm = torch.min(img[c])
            rnge = torch.max(img[c]) - minm
            img[c] = torch.pow(((img[c] - minm) / torch.clamp(rnge, min=1e-7)), g) * rnge + minm
            if r:
                # std_here, mn_here = torch.std_mean(img[c])
                mn_here = torch.mean(img[c])
                std_here = torch.std(img[c])
                img[c] -= mn_here
                img[c] *= (std / torch.clamp(std_here, min=1e-7))
                img[c] += mean

            if i:
                img[c] *= -1
        return img

class MirrorTransform(BasicTransform):
    def __init__(self, allowed_axes: Tuple[int, ...]):
        super().__init__()
        self.allowed_axes = allowed_axes

    def get_parameters(self, **data_dict) -> dict:
        axes = [i for i in self.allowed_axes if torch.rand(1) < 0.5]
        return {
            'axes': axes
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if len(params['axes']) == 0:
            return img
        axes = [i + 1 for i in params['axes']]
        return torch.flip(img, axes)

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        if len(params['axes']) == 0:
            return segmentation
        axes = [i + 1 for i in params['axes']]
        return torch.flip(segmentation, axes)

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        if len(params['axes']) == 0:
            return regression_target
        axes = [i + 1 for i in params['axes']]
        return torch.flip(regression_target, axes)

    def _apply_to_bbox(self, bbox, **params):
        raise NotImplementedError

    def _apply_to_keypoints(self, keypoints, **params):
        raise NotImplementedError


### Utils function

def apply_filter(x: torch.Tensor, kernel: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Codpied from https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/layers/simplelayers.py

    Filtering `x` with `kernel` independently for each batch and channel respectively.

    Args:
        x: the input image, must have shape (batch, channels, H[, W, D]).
        kernel: `kernel` must at least have the spatial shape (H_k[, W_k, D_k]).
            `kernel` shape must be broadcastable to the `batch` and `channels` dimensions of `x`.
        kwargs: keyword arguments passed to `conv*d()` functions.

    Returns:
        The filtered `x`.

    Examples:

    .. code-block:: python

        >>> import torch
        >>> from monai.networks.layers import apply_filter
        >>> img = torch.rand(2, 5, 10, 10)  # batch_size 2, channels 5, 10x10 2D images
        >>> out = apply_filter(img, torch.rand(3, 3))   # spatial kernel
        >>> out = apply_filter(img, torch.rand(5, 3, 3))  # channel-wise kernels
        >>> out = apply_filter(img, torch.rand(2, 5, 3, 3))  # batch-, channel-wise kernels

    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor but is {type(x).__name__}.")
    batch, chns, *spatials = x.shape
    n_spatial = len(spatials)
    if n_spatial > 3:
        raise NotImplementedError(f"Only spatial dimensions up to 3 are supported but got {n_spatial}.")
    k_size = len(kernel.shape)
    if k_size < n_spatial or k_size > n_spatial + 2:
        raise ValueError(
            f"kernel must have {n_spatial} ~ {n_spatial + 2} dimensions to match the input shape {x.shape}."
        )
    kernel = kernel.to(x)
    # broadcast kernel size to (batch chns, spatial_kernel_size)
    kernel = kernel.expand(batch, chns, *kernel.shape[(k_size - n_spatial) :])
    kernel = kernel.reshape(-1, 1, *kernel.shape[2:])  # group=1
    x = x.view(1, kernel.shape[0], *spatials)
    conv = [F.conv1d, F.conv2d, F.conv3d][n_spatial - 1]
    if "padding" not in kwargs:
        kwargs["padding"] = "same"

    if "stride" not in kwargs:
        kwargs["stride"] = 1
    output = conv(x, kernel, groups=kernel.shape[0], bias=None, **kwargs)
    return output.view(batch, chns, *output.shape[2:])

