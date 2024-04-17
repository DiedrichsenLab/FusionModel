#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproducibility test across individual parcellations

Created on 4/17/2024 at 5:02 PM
Author: dzhi
"""
import numpy as np
import torch as pt
import nibabel as nb
import nitools as nt
import matplotlib.pyplot as plt
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.full_model as fm

from set_globals import base_dir, model_dir

if __name__ == "__main__":
    ## Step 1: Loading a pre-trained group model
    atlas, _ = am.get_atlas('MNISymC3')
    model_name = f'/Models/Models_03/asym_PoNiIbWmDeSo_space-MNISymC3_K-34'
    U, minfo = ar.load_group_parcellation(model_dir + model_name, device='cuda')
    ar_model = ar.build_arrangement_model(U, prior_type='logpi', atlas=atlas,
                                          sym_type='asym')
    # Step 2: Get data set and train the individual maps
    data, info, tds = ds.get_dataset(base_dir, 'MDTB', atlas=atlas.name, subj=None)
    tdata, cond_v, part_v, sub_ind = fm.prep_datasets(data, info.half,
                                                      info['cond_num_uni'].values,
                                                      info['half'].values,
                                                      join_sess=False,
                                                      join_sess_part=False)

    # Get indiv parcellation from first half
    indiv_par_1, _, M = fm.get_indiv_parcellation(ar_model, atlas, [tdata[0]],
                                                  [cond_v[0]], [part_v[0]],
                                                  [sub_ind[0]], sym_type='asym',
                                                  em_params={'num_subj': tdata[0].shape[0],
                                                             'uniform_kappa': False,
                                                             'subjects_equal_weight': True,
                                                             'subject_specific_kappa': False,
                                                             'parcel_specific_kappa': False})
    # Get indiv parcellation from second half
    indiv_par_1, _, M = fm.get_indiv_parcellation(ar_model, atlas, [tdata[1]],
                                                  [cond_v[1]], [part_v[1]],
                                                  [sub_ind[1]], sym_type='asym',
                                                  em_params={'num_subj': tdata[1].shape[0],
                                                             'uniform_kappa': False,
                                                             'subjects_equal_weight': True,
                                                             'subject_specific_kappa': False,
                                                             'parcel_specific_kappa': False})


    ## Step 3: Run dice coefficient between indivpar1 and indivpar2

