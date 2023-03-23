#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation on the parcellations learned by smoothed data (cortex)

Created on 3/21/2023 at 10:59 AM
Author: dzhi
"""
# System import
import sys
from pathlib import Path

# Basic libraries import
import pickle
import numpy as np
import torch as pt
import seaborn as sb
import pandas as pd
import nibabel as nb
import matplotlib.pyplot as plt
import scipy.io as spio

# Modeling import
import generativeMRF.emissions as em
import generativeMRF.arrangements as ar
import generativeMRF.full_model as fm
import generativeMRF.evaluation as ev

# Dataset fusion import
import Functional_Fusion.atlas_map as am
from Functional_Fusion.dataset import *
import FusionModel.util as ut
from FusionModel.evaluate import *
from FusionModel.learn_fusion_gpu import *

# pytorch cuda global flag
# pt.cuda.is_available = lambda : False
pt.set_default_tensor_type(pt.cuda.FloatTensor
                           if pt.cuda.is_available() else
                           pt.FloatTensor)

# Find model directory to save model fitting results
model_dir = 'Y:/data/Cortex/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/srv/diedrichsen/data/Cortex/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/Volumes/diedrichsen_data$/data/Cortex/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    raise (NameError('Could not find model_dir'))

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = 'Y:/data/FunctionalFusion'
if not Path(base_dir).exists():
    raise(NameError('Could not find base_dir'))

atlas_dir = base_dir + f'/Atlases'
res_dir = model_dir + f'/Results'

def fit_smooth(K=[10, 17, 20, 34, 40, 68, 100], smooth=[0,3,7], model_type='03',
               sym_type=['asym'], datasets_list=[0], space='fs32k'):
    # datasets = np.array(['MDTB', 'Pontine', 'Nishimoto', 'IBC', 'WMFS',
    #                      'Demand', 'Somatotopic', 'HCP'], dtype=object)
    _, _, my_dataset = get_dataset(ut.base_dir, 'MDTB')
    sess = my_dataset.sessions
    for indv_sess in sess:
        for k in K:
            for s in smooth:
                wdir, fname, info, models = fit_all(set_ind=datasets_list, K=k,
                                                    repeats=50,
                                                    model_type=model_type,
                                                    sym_type=sym_type,
                                                    this_sess=[[indv_sess]],
                                                    space=space, smooth=s)

                if s is not None:
                    wdir = wdir + '/smoothed'
                    fname = fname + f'_smooth-{s}_{indv_sess}'
                else:
                    fname = fname + f'_{indv_sess}'

                # Write in fitted files
                print(f'Write in {wdir + fname}...')
                info.to_csv(wdir + fname + '.tsv', sep='\t')
                with open(wdir + fname + '.pickle', 'wb') as file:
                    pickle.dump(models, file)

def eval_smoothed(model_name, space='fs32k', t_datasets=['MDTB'], train_ses='ses-s1',
                  test_ses='ses-s2', train_smooth=None, test_smooth=None):
    """Evaluate group and individual DCBC and coserr of IBC single
       sessions on all other test datasets.
    Args:
        K: the number of parcels
    Returns:
        Write in evaluation file
    """
    if not isinstance(model_name, list):
        model_name = [model_name]

    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    atlas, info = am.get_atlas(space, atlas_dir=base_dir + '/Atlases')
    results = pd.DataFrame()
    # Evaluate all single sessions on other datasets
    for ds in t_datasets:
        print(f'Testdata: {ds}\n')
        # Preparing atlas, cond_vec, part_vec
        this_type = T.loc[T.name == ds]['default_type'].item()
        train_dat, train_inf, train_tds = get_dataset(base_dir, ds, atlas=space,
                                                      sess=[train_ses], type=this_type,
                                                      smooth=None if train_smooth==0 else
                                                      train_smooth)
        test_dat, _, _ = get_dataset(base_dir, ds, atlas=space, sess=[test_ses],
                                     type=this_type, smooth=None if test_smooth==0 else test_smooth)

        cond_vec = train_inf[train_tds.cond_ind].values.reshape(-1, ) # default from dataset class
        part_vec = train_inf['half'].values

        # Calculate distance metric given by input atlas
        dist = ut.load_fs32k_dist(file_type='distGOD_sp', hemis='full',
                                  device='cuda' if pt.cuda.is_available() else 'cpu')
        res_dcbc = run_dcbc(model_name, train_dat, test_dat, dist, cond_vec, part_vec,
                            device='cuda' if pt.cuda.is_available() else 'cpu',
                            same_subj=True)
        res_dcbc['test_data'] = ds
        res_dcbc['train_ses'] = train_ses
        res_dcbc['test_ses'] = test_ses
        res_dcbc['train_smooth'] = train_smooth
        res_dcbc['test_smooth'] = test_smooth
        results = pd.concat([results, res_dcbc], ignore_index=True)

    return results

def eval_smoothed_models(K=[100], model_type=['03','04'],
                         smooth=[0,1,2,3,7], outname='K-10to100_Md_on_Sess_smooth'):
    CV_setting = [('ses-s1', 'ses-s2'), ('ses-s2', 'ses-s1')]
    D = pd.DataFrame()
    for (train_ses, test_ses) in CV_setting:
        for t in smooth:
            for s in smooth:
                model_name = []
                if s != 0:
                    model_name += [f'Models_{mt}/sym_Md_space-fs32k_K-{this_k}_smooth-{s}_' \
                                   f'{train_ses}' for this_k in K for mt in model_type]
                else:
                    model_name += [f'Models_{mt}/sym_Md_space-fs32k_K-{this_k}_{train_ses}'
                                   for this_k in K for mt in model_type]

                results = eval_smoothed(model_name, t_datasets=['MDTB'], train_ses=train_ses,
                                        test_ses=test_ses, train_smooth=s, test_smooth=t)
                D = pd.concat([D, results], ignore_index=True)

    # Save file
    wdir = model_dir + f'/Models/Evaluation'
    fname = f'/eval_all_asym_{outname}.tsv'
    D.to_csv(wdir + fname, index=False, sep='\t')


if __name__ == "__main__":
    ############# Fitting cortical models #############
    # 1. fit whole cortex (using symmetric arrangement)
    fit_smooth(K=[100], smooth=[1], model_type='03',sym_type=['sym'], space='fs32k')
    # fit_smooth(K=[100], smooth=[None], model_type='04',sym_type=['sym'], space='fs32k')

    # 2. fit single hemisphere (using asymmetric arrangement)
    # for mt in ['03','04']:
    #     fit_smooth(K=[50], smooth=[None], model_type=mt, sym_type=['asym'],
    #                space='fs32k_L')
    #     fit_smooth(K=[50], smooth=[None], model_type=mt, sym_type=['asym'],
    #                space='fs32k_R')

    ############# Convert fitted model to label cifti #############
    # fname = 'Models_03/sym_Md_space-fs32k_K-10_ses-s1'
    # ut.write_model_to_labelcifti(fname, load_best=True, device='cpu')

    ############# Evaluating models #############
    eval_smoothed_models(model_type=['03'], smooth=[0],
                         outname='K-10to100_Md_on_Sess_smooth')

    ############# Plotting comparison #############
    # fname = f'/Models/Evaluation/eval_all_asym_K-10to100_Md_on_Sess_smooth.tsv'
    # D = pd.read_csv(model_dir + fname, delimiter='\t')
    # # plot_smooth_vs_unsmooth(D, test_s=7)
    # compare_diff_smooth(D)

    ############# Plot fusion atlas #############
    # Making color map
    # plot_smooth_map(K=40, model_type='03', sess='ses-s1')