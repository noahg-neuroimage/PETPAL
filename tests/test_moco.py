# pylint: skip-file
import tempfile
import os
import numpy as np
import pandas as pd

import ants
from ants.contrib.affine3d import Rotate3D, Translate3D
from petpal.utils.image_io import safe_copy_meta
from petpal.utils.useful_functions import coerce_outpath_extension
from petpal.utils.timeseries_from_img_list import timeseries_from_img_list
from petpal.preproc.motion_corr import MotionCorrect


def apply_random_xfms(static_img: ants.ANTsImage, center: list, nframes: int=4):
    sim_imgs = []
    by_frame = []

    for _i in range(nframes):
        rotation_pars = np.random.rand((3))*2-1
        translation_pars = np.random.rand((3))*6-3
        by_frame.append(np.hstack((rotation_pars,translation_pars)))

        ants_xfm_rot = Rotate3D(rotation=tuple(rotation_pars)).transform().parameters
        ants_xfm_tra = Translate3D(translation=tuple(translation_pars)).transform().parameters
        ants_xfm_rigid_matrix = np.zeros((12))
        ants_xfm_rigid_matrix[:9] = ants_xfm_rot[:9]
        ants_xfm_rigid_matrix[9:] = ants_xfm_tra[9:]

        ants_xfm = ants.ANTsTransform()
        ants_xfm.set_fixed_parameters(center)
        ants_xfm.set_parameters(ants_xfm_rigid_matrix)

        sim_imgs.append(ants_xfm.apply_to_image(static_img))

    sim_img = timeseries_from_img_list(sim_imgs)
    return sim_img, by_frame


def test_motion():
    """Read static PET data. Apply random translations and rotations. Run motion correction.
    Read in result. Check if it matches the applied transforms up to a tolerance of 0.1 mm or 
    degree for all parameters."""

    static_img_path = 'static.nii.gz' # need a static test image
    static_img = ants.image_read(static_img_path)
    center = [0,0,0] # insert center of the test image

    sim_img, by_frame = apply_random_xfms(static_img=static_img, center=center, nframes=4)
    sim_img_path = tempfile.mkstemp(suffix='.nii.gz',prefix='MocoSim_')[1]
    sim_json_path = coerce_outpath_extension(path=sim_img_path,ext='.json')

    ants.image_write(sim_img,sim_img_path)
    safe_copy_meta(static_img_path,sim_img_path)

    moco_img_path = tempfile.mkstemp(suffix='.nii.gz',prefix='MocoResult_')[1]
    moco_json_path = coerce_outpath_extension(path=moco_img_path,ext='.json')
    mc = MotionCorrect()
    mc(input_image_path=sim_img_path,
       output_image_path=moco_img_path,
       motion_target_option=static_img_path,
       window_duration=300,
       transform_type='DenseRigid')

    xfms_path = coerce_outpath_extension(path=moco_img_path, ext='.csv')
    xfms = pd.read_csv(xfms_path,index_col='frame')
    xfm_applied = -np.asarray(xfms)[:,:6]

    os.remove(sim_img_path)
    os.remove(sim_json_path)
    os.remove(moco_img_path)
    os.remove(moco_json_path)
    os.remove(xfms_path)

    np.testing.assert_allclose(xfm_applied,np.asarray(by_frame),rtol=0.1,atol=0.1)
