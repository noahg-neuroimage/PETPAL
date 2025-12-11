"""
Module for functions calculating standard uptake value (SUV) and related measures, such as standard
uptake value ratio (SUVR).
"""
from ..utils.stats import mean_value_in_region
import numpy as np
from petpal.utils import image_io


import ants


def suvr(input_image_path: str,
         out_image_path: str | None,
         segmentation_image_path: str,
         ref_region: int | list[int]) -> ants.ANTsImage:
    """
    Computes an ``SUVR`` (Standard Uptake Value Ratio) by taking the average of
    an input image within a reference region, and dividing the input image by
    said average value.

    Args:
        input_image_path (str): Path to 3D weighted series sum or other
            parametric image on which we compute SUVR.
        out_image_path (str): Path to output image file which is written to. If None, no output is written.
        segmentation_image_path (str): Path to segmentation image, which we use
            to compute average uptake value in the reference region.
        ref_region (int): Region or list of region mappings over which to compute average SUV. If a
            list is provided, combines all regions in the list as one reference region.

    Returns:
        ants.ANTsImage: SUVR parametric image
    """
    suv_img = ants.image_read(filename=input_image_path)
    suv_arr = suv_img.numpy()
    segmentation_img = ants.image_read(filename=segmentation_image_path,
                                        pixeltype='unsigned int')

    if len(suv_arr.shape)!=3:
        raise ValueError("SUVR input image is not 3D. If your image is dynamic, try running 'weighted_series_sum'"
                         " first.")

    ref_region_avg = mean_value_in_region(input_img=suv_img,
                                          seg_img=segmentation_img,
                                          mapping=ref_region)

    suvr_arr = suv_arr / ref_region_avg

    out_img = ants.from_numpy_like(data=suvr_arr,
                                   image=suv_img)

    if out_image_path is not None:
        ants.image_write(image=out_img,
                         filename=out_image_path)
        image_io.safe_copy_meta(input_image_path=input_image_path,
                                out_image_path=out_image_path)

    return out_img