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
pt.cuda.is_available = lambda : False
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

    space_sp = space.split('_')
    if len(space_sp) == 1:
        hemis = 'full'
        atlas, _ = am.get_atlas(space, atlas_dir=base_dir + '/Atlases')
        vert_indx = np.arange(0,atlas.P)
    elif len(space_sp) == 2:
        hemis = 'half'
        space = space_sp[0]
        hem = space_sp[1]
        hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
        atlas, _ = am.get_atlas(space, atlas_dir=base_dir + '/Atlases')
        stru_idx = atlas.structure.index(hemis_dict[hem])
        vert_indx = atlas.indx_full[stru_idx]
    else:
        raise NameError('Unrecognized `space` for atlasing!')

    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    results = pd.DataFrame()
    for ds in t_datasets:
        # Evaluate all single sessions on other datasets
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
        dist = ut.get_fs32k_weights(file_type='distAvrg_sp', hemis=hemis,
                                  device='cuda' if pt.cuda.is_available() else 'cpu')
        res_dcbc, corrs = run_dcbc(model_name, train_dat[:,:,vert_indx], test_dat[:,:,vert_indx],
                                   dist, cond_vec, part_vec,
                                   device='cuda' if pt.cuda.is_available() else 'cpu',
                                   verbose=False, return_wb=True, same_subj=False)
        res_dcbc['test_data'] = ds
        res_dcbc['train_ses'] = train_ses
        res_dcbc['test_ses'] = test_ses
        res_dcbc['train_smooth'] = train_smooth
        res_dcbc['test_smooth'] = test_smooth
        results = pd.concat([results, res_dcbc], ignore_index=True)

    return results, corrs

def eval_smoothed_models(K=[100], model_type=['03','04'], space='fs32k', sym='asym',
                         smooth=[0,1,2,3,7], save=False, plot_wb=True,
                         outname='asym_K-10to100_Md_on_Sess_smooth'):
    CV_setting = [('ses-s1', 'ses-s2'), ('ses-s2', 'ses-s1')]
    D = pd.DataFrame()
    for (train_ses, test_ses) in CV_setting:
        dict = {}
        for s in smooth:
            dict_row = {}
            for t in smooth:
                #### Option 1: the group prior was trained all from unsmoothed data
                model_name = [f'Models_{mt}/{sym}_Md_space-{space}_K-{this_k}_{train_ses}'
                              for this_k in K for mt in model_type]
                #### Option 2: the group prior was trained on the same smoothing level
                #### that we used for individual training
                # model_name = []
                # if s != 0:
                #     model_name += [f'Models_{mt}/smoothed/{sym}_Md_space-{space}_K-{this_k}_smooth' \
                #                    f'-{s}_{train_ses}' for this_k in K for mt in model_type]
                # else:
                #     model_name += [f'Models_{mt}/{sym}_Md_space-{space}_K-{this_k}_{train_ses}'
                #                    for this_k in K for mt in model_type]

                results, corrs = eval_smoothed(model_name, space=space, t_datasets=['MDTB'],
                                               train_ses=train_ses, test_ses=test_ses,
                                               train_smooth=s, test_smooth=t)
                dict_row[f'test_{t}'] = corrs
                D = pd.concat([D, results], ignore_index=True)

            dict[f'train_{s}'] = dict_row

        if plot_wb:
            plot_corr_wb(dict, 0, title=model_name)

    if save:
        # Save file
        wdir = model_dir + f'/Models/Evaluation'
        fname = f'/eval_all_{outname}.tsv'
        D.to_csv(wdir + fname, index=False, sep='\t')

def plot_smooth_vs_unsmooth(D, test_s=0):
    D = D.loc[D.test_smooth==test_s]
    plt.figure(figsize=(10, 15))
    crits = ['dcbc_group', 'dcbc_indiv', 'dcbc_indiv_em']
    for i, c in enumerate(crits):
        plt.subplot(3, 2, i*2 + 1)
        sb.barplot(data=D, x='model_type', y=c, hue='train_smooth', errorbar="se")

        # if 'coserr' in c:
        #     plt.ylim(0.4, 1)
        plt.subplot(3, 2, i*2 + 2)
        sb.lineplot(data=D, x='K', y=c, hue='model_type', style="train_smooth",
                    errorbar=None, err_style="bars", markers=False)

    plt.suptitle(f'All datasets fusion')
    plt.tight_layout()
    plt.show()

def compare_diff_smooth(D, mt='03', outname='MDTB_cortex', save=False):
    res_plot = D.loc[D.model_type==f'Models_{mt}']

    # 1. Plot evaluation results
    plt.figure(figsize=(15, 10))
    crits = ['dcbc_group', 'dcbc_indiv', 'dcbc_indiv_em']
    for i, c in enumerate(crits):
        plt.subplot(2, 3, i + 1)
        result = pd.pivot_table(res_plot, values=c, index='train_smooth', columns='test_smooth')
        rdgn = sb.color_palette("vlag", as_cmap=True)
        # rdgn = sb.color_palette("Spectral", as_cmap=True)
        sb.heatmap(result, annot=True, cmap=rdgn, fmt='.2g')
        plt.title(c)

        plt.subplot(2, 3, i + 4)
        sb.lineplot(data=res_plot, x='train_smooth', y=c, hue='test_smooth',
                    errorbar='se', err_style="bars", markers=False)

    plt.suptitle(f'Spatial smoothness - model type {mt} - {outname}')
    plt.tight_layout()

    if save:
        plt.savefig('diff_Ktrue20_Kfit5to40.pdf', format='pdf')
    plt.show()

if __name__ == "__main__":
    ############# Fitting cortical models #############
    # 1. fit whole cortex (using symmetric arrangement)
    # fit_smooth(K=[100], smooth=[1], model_type='03',sym_type=['sym'], space='fs32k')
    # fit_smooth(K=[100], smooth=[None], model_type='04',sym_type=['sym'], space='fs32k')

    # 2. fit single hemisphere (using asymmetric arrangement)
    # for mt in ['03','04']:
    #     fit_smooth(K=[50], smooth=[None], model_type=mt, sym_type=['asym'],
    #                space='fs32k_L')
    #     fit_smooth(K=[50], smooth=[None], model_type=mt, sym_type=['asym'],
    #                space='fs32k_R')

    ############# Convert fitted model to label cifti #############
    # fname = 'Models_03/asym_Md_space-fs32k_L_K-17_ses-s1'
    # ut.write_model_to_labelcifti(fname, load_best=True, sym='asym', device='cpu')

    ############# Evaluating models / plot wb-curves #############
    eval_smoothed_models(K=[17], model_type=['03'], space='fs32k_L', sym='asym',
                         smooth=[0,1,2,3], save=False, plot_wb=True,
                         outname='asym_K-17_Md_on_Sess_smooth_groupTrain0')

    ############# Plotting comparison #############
    fname = f'/Models/Evaluation/eval_all_asym_K-17_Md_on_Sess_smooth_groupTrain0.tsv'
    D = pd.read_csv(model_dir + fname, delimiter='\t')
    # plot_smooth_vs_unsmooth(D, test_s=3)
    compare_diff_smooth(D, mt='03')