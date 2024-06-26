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
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.evaluation as ev

# Dataset fusion import
import Functional_Fusion.atlas_map as am
from Functional_Fusion.dataset import *
import FusionModel.util as ut
from FusionModel.evaluate import *
from FusionModel.learn_fusion_gpu import *

# pytorch cuda global flag
# pt.cuda.is_available = lambda : False
DEVICE = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
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
               sym_type=['asym'], datasets_list=[3], space='fs32k', arrange='cRBM_Wc',
               thetas=[1,2,3], part_num=None):
    # datasets = np.array(['MDTB', 'Pontine', 'Nishimoto', 'IBC', 'WMFS',
    #                      'Demand', 'Somatotopic', 'HCP'], dtype=object)
    _, _, my_dataset = get_dataset(ut.base_dir, 'IBC')
    sess = my_dataset.sessions
    for indv_sess in sess:
        for k in K:
            for s in smooth:
                for theta in thetas:
                    wdir, fname, info, models = fit_all(set_ind=datasets_list, K=k,
                                                        repeats=5,
                                                        model_type=model_type,
                                                        sym_type=sym_type,
                                                        arrange=arrange,
                                                        space=space, smooth=s,
                                                        sc=False, Wc_theta=theta,
                                                        part_num=part_num)

                    if s is not None:
                        wdir = wdir + '/smoothed'
                        fname = fname + f'_smooth-{s}_{indv_sess}_{arrange}-{theta}_step-0.5'
                    else:
                        fname = fname + f'_{indv_sess}_{arrange}-{theta}_step-0.5'

                    if part_num is not None:
                        fname = fname + f'_part-{part_num}'

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
        dat, train_inf, train_tds = get_dataset(base_dir, ds, atlas=space,
                                                      type=this_type,
                                                      smooth=None)
        # test_dat, _, _ = get_dataset(base_dir, ds, atlas=space, sess=[test_ses],
        #                              type=this_type, smooth=None if test_smooth==0 else test_smooth)

        cond_vec = train_inf[train_tds.cond_ind].values.reshape(-1, ) # default from dataset class
        part_vec = train_inf['half'].values
        train_idx = part_vec == train_ses
        test_idx = part_vec == test_ses
        train_dat = dat[:, train_idx, :]
        test_dat = dat[:, test_idx, :]
        # Calculate distance metric given by input atlas
        dist = ut.load_fs32k_dist(file_type='distAvrg_sp', hemis=hemis,
                                  device='cuda' if pt.cuda.is_available() else 'cpu')
        res_dcbc, corrs = run_dcbc(model_name, train_dat[:,:,vert_indx], test_dat[:,:,vert_indx],
                                   dist, cond_vec[train_idx], part_vec[train_idx],
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
    CV_setting = [(1, 2), (2, 1)]
    D = pd.DataFrame()
    for (train_ses, test_ses) in CV_setting:
        dict = {}
        for s in smooth:
            dict_row = {}
            for t in [0]:
                #### Option 1: the group prior was trained all from unsmoothed data
                # model_name = [f'Models_{mt}/{sym}_Ib_space-{space}_K-{this_k}' \
                #               f'_cRBM_Wc-{s}' for this_k in K for mt in model_type]
                model_name = [f'Models_{mt}/{sym}_Ib_space-{space}_K-{this_k}_independent'
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

                results, corrs = eval_smoothed(model_name, space=space, t_datasets=['IBC'],
                                               train_ses=train_ses, test_ses=test_ses,
                                               train_smooth=s, test_smooth=t)
                dict_row[f'test_{t}'] = corrs
                D = pd.concat([D, results], ignore_index=True)

            dict[f'train_{s}'] = dict_row

        if plot_wb:
            plot_corr_wb(dict, 0, type_2='wb', title=model_name)
            plot_corr_wb(dict, 0, type_2='nums', title=model_name)
            plot_corr_wb(dict, 0, type_2='weight', title=model_name)

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
        sb.lineplot(data=D, x='train_smooth', y=c, errorbar="se",err_style="bars")
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

def F_test(D, mt='03', value='dcbc_group'):
    res_plot = D.loc[D.model_type==f'Models_{mt}']
    arvg_dcbc = res_plot.groupby(['train_smooth','subj_num'])['dcbc_group','dcbc_indiv',
                                                              'dcbc_indiv_em'].mean().reset_index()

    new_DD = res_plot.loc[res_plot.test_sess=='s1']
    new_DD[['dcbc_group','dcbc_indiv','dcbc_indiv_em']] = arvg_dcbc[['dcbc_group','dcbc_indiv',
                                                                     'dcbc_indiv_em']]

    # F-test
    from statsmodels.stats.anova import AnovaRM
    model = AnovaRM(new_DD, value, 'subj_num', within=['train_smooth'])
    res = model.fit()
    print(res)

if __name__ == "__main__":
    ############# Fitting cortical models #############
    # 1. fit whole cortex (using symmetric arrangement)
    # fit_smooth(K=[100], smooth=[1], model_type='03',sym_type=['sym'], space='fs32k')
    # fit_smooth(K=[100], smooth=[None], model_type='04',sym_type=['sym'], space='fs32k')
    for p in [1,2]:
        for tt in [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
            wdir, fname, info, models = fit_all(set_ind=[3], K=17, repeats=5,
                                                model_type='03', sym_type=['asym'],
                                                arrange='cRBM_Wc', space='fs32k_L',
                                                smooth=None, sc=False, Wc_theta=tt,
                                                part_num=p)

            fname = fname + f'_part-{p}'
            # Write in fitted files
            print(f'Write in {wdir + fname}...')
            info.to_csv(wdir + fname + '.tsv', sep='\t')
            with open(wdir + fname + '.pickle', 'wb') as file:
                pickle.dump(models, file)

    # 2. fit single hemisphere (using asymmetric arrangement)
    # for mt in ['03']:
    #     fit_smooth(K=[17], smooth=[None], model_type=mt, sym_type=['asym'],
    #                space='fs32k_L', thetas=[0.1, 0.5, 1.0, 1.5, 2.0], part_num=1)
        # fit_smooth(K=[17], smooth=[None], model_type=mt, sym_type=['asym'],
        #            space='fs32k_R')

    # ############# Convert fitted model to label cifti #############
    # fname = [f'Models_03/asym_Md_space-fs32k_L_K-17_ses-s1_cRBM_Wc-{s}_step-0.5'
    #          for s in np.linspace(0.5,4,10).round(2).tolist()]
    # col_name = [f'theta_{s}' for s in np.linspace(0.5,4,10).round(2).tolist()]
    # ut.write_model_to_labelcifti(fname, align=True, col_names=col_name, load='best',
    #                              oname='Models_03/asym_Md_space-fs32k_L_K-17_ses-s1_cRBM_Wc_'
    #                                    'step-0.5_theta_0.1-1', device='cpu')

    ############# Evaluating models / plot wb-curves #############
    eval_smoothed_models(K=[17], model_type=['03'], space='fs32k_L', sym='asym',
                         smooth=[0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
                         save=True, plot_wb=True,
                         outname='asym_fs32k-L_K-17_Ib_cRBM_Wc_0.1to5.0_group0_independent')

    ############# Repeated measure ANOVA #############
    # fname = f'/Models/Evaluation/eval_all_asym_fs32k-L_K-17_Md_on_Sess_smooth_group0.tsv'
    # D = pd.read_csv(model_dir + fname, delimiter='\t')
    # F_test(D, mt='03',value='dcbc_indiv')

    ############# Plotting comparison #############
    fname = f'/Models/Evaluation/eval_all_asym_fs32k-L_K-17_Md_on_Sess_cRBM_Wc_0.1-4.0_group0.tsv'
    D = pd.read_csv(model_dir + fname, delimiter='\t')
    plot_smooth_vs_unsmooth(D, test_s=0)
    compare_diff_smooth(D, mt='03')
