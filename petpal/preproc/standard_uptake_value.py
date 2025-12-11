"""
Module for functions calculating standard uptake value (SUV) and related measures, such as standard
uptake value ratio (SUVR).
"""
from petpal.preproc.image_operations_4d import extract_mean_roi_tac_from_nifti_using_segmentation
from petpal.utils import image_io


import ants


def suvr(input_image_path: str,
         out_image_path: str | None,
         segmentation_image_path: str,
         ref_region: int,
         verbose: bool=False) -> ants.ANTsImage:
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
        ref_region (int): Region number mapping to the reference region in the
            segmentation image.
        verbose (bool): Set to ``True`` to output processing information. Default is False.

    Returns:
        ants.ANTsImage: SUVR parametric image
    """
    pet_img = ants.image_read(filename=input_image_path)
    pet_arr = pet_img.numpy()
    segmentation_img = ants.image_read(filename=segmentation_image_path,
                                        pixeltype='unsigned int')
    segmentation_arr = segmentation_img.numpy()

    if len(pet_arr.shape)!=3:
        raise ValueError("SUVR input image is not 3D. If your image is dynamic, try running 'weighted_series_sum'"
                         " first.")

    ref_region_avg = extract_mean_roi_tac_from_nifti_using_segmentation(input_image_4d_numpy=pet_arr,
                                                                        segmentation_image_numpy=segmentation_arr,
                                                                        region=ref_region,
                                                                        verbose=verbose)

    suvr_arr = pet_arr / ref_region_avg[0]

    out_img = ants.from_numpy_like(data=suvr_arr,
                                     image=pet_img)

    if out_image_path is not None:
        ants.image_write(image=out_img,
                         filename=out_image_path)
        image_io.safe_copy_meta(input_image_path=input_image_path,
                                out_image_path=out_image_path)

    return out_img