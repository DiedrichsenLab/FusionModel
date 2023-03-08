#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation on the parcellations learned by smoothed data

Created on 2/24/2023 at 1:25 PM
Author: dzhi
"""
from time import gmtime
from pathlib import Path
import pandas as pd
import numpy as np
import Functional_Fusion.atlas_map as am
import Functional_Fusion.matrix as matrix
from Functional_Fusion.dataset import *
import generativeMRF.emissions as em
import generativeMRF.arrangements as ar
import generativeMRF.full_model as fm
import generativeMRF.evaluation as ev

from scipy.linalg import block_diag
import nibabel as nb
import SUITPy as suit
import torch as pt
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sb
import sys
import time
import pickle
from copy import copy,deepcopy
from itertools import combinations
from FusionModel.util import *
from FusionModel.evaluate import *
from FusionModel.learn_fusion_gpu import *

# pytorch cuda global flag
# pt.cuda.is_available = lambda : False
pt.set_default_tensor_type(pt.cuda.FloatTensor
                           if pt.cuda.is_available() else
                           pt.FloatTensor)

# Find model directory to save model fitting results
model_dir = 'Y:/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/srv/diedrichsen/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel'
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
               datasets_list=[0]):
    datasets = np.array(['MDTB', 'Pontine', 'Nishimoto', 'IBC', 'WMFS',
                         'Demand', 'Somatotopic', 'HCP'], dtype=object)
    _, _, my_dataset = get_dataset(ut.base_dir, 'MDTB')
    sess = my_dataset.sessions
    for indv_sess in sess:
        for k in K:
            for s in smooth:
                wdir, fname, info, models = fit_all(set_ind=datasets_list, K=k,
                                                    repeats=2,
                                                    model_type=model_type,
                                                    sym_type=['asym'],
                                                    this_sess=[[indv_sess]],
                                                    space='MNISymC3', smooth=s)

                # wdir = wdir + '/smoothed'
                # fname = fname + f'_smooth-{s}_{indv_sess}'
                fname = fname + f'_{indv_sess}'
                info.to_csv(wdir + fname + '.tsv', sep='\t')
                with open(wdir + fname + '.pickle', 'wb') as file:
                    pickle.dump(models, file)

def eval_smoothed(model_name, t_datasets=['MDTB','Pontine','Nishimoto'],
                  train_ses='ses-s1', test_ses='ses-s2', train_smooth=None):
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
    atlas, _ = am.get_atlas('MNISymC3', atlas_dir=base_dir + '/Atlases')
    results = pd.DataFrame()
    # Evaluate all single sessions on other datasets
    for ds in t_datasets:
        print(f'Testdata: {ds}\n')
        # Preparing atlas, cond_vec, part_vec
        this_type = T.loc[T.name == ds]['default_type'].item()
        train_dat, train_inf, train_tds = get_dataset(base_dir, ds, atlas='MNISymC3',
                                                      sess=[train_ses], type=this_type,
                                                      smooth=None if train_smooth==2 else train_smooth)
        test_dat, _, _ = get_dataset(base_dir, ds, atlas='MNISymC3',
                                     sess=[test_ses], type=this_type)

        cond_vec = train_inf[train_tds.cond_ind].values.reshape(-1, ) # default from dataset class
        part_vec = train_inf['half'].values

        # 1. Run DCBC individual
        res_dcbc = run_dcbc(model_name, train_dat, test_dat, atlas,
                            cond_vec, part_vec, device='cuda')
        res_dcbc['test_data'] = ds
        res_dcbc['train_ses'] = train_ses
        res_dcbc['test_ses'] = test_ses
        res_dcbc['smooth'] = train_smooth
        results = pd.concat([results, res_dcbc], ignore_index=True)

    return results

def result_1_plot_curve(fname, crits=['coserr', 'dcbc'], oname=None, save=False):
    D = pd.read_csv(fname, delimiter='\t')
    num_plot = len(crits)
    plt.figure(figsize=(5*num_plot, 5))
    for i, c in enumerate(crits):
        plt.subplot(1, num_plot, i + 1)
        gm = D[c][D.type == 'group'].mean()
        sb.lineplot(data=D.loc[(D.type != 'group')&(D.type != 'floor')],
                    y=c, x='runs', hue='type', markers=True, dashes=False)
        plt.xticks(ticks=np.arange(16) + 1)
        plt.axhline(gm, color='k', ls=':')
        if c == 'coserr':
            fl = D.coserr[D.type == 'floor'].mean()
            plt.axhline(fl, color='r', ls=':')

    plt.suptitle(f'Individual vs. group, {oname}')
    plt.tight_layout()
    if (oname is not None) and save:
        plt.savefig(res_dir + f'/1.indiv_vs_group/{oname}', format='pdf')

    plt.show()

def result_1_plot_flatmap(Us, sub=0, cmap='tab20', save_folder=None):
    group, U_em, U_comp = Us[0], Us[1], Us[2]

    oneRun_em = U_em[0][sub]
    allRun_em = U_em[-1][sub]
    oneRun_comp = U_comp[0][sub]
    maps = pt.stack([oneRun_em, allRun_em, oneRun_comp, group])

    # plt.figure(figsize=(25, 25))
    plot_multi_flat(maps.cpu().numpy(), 'MNISymC3', grid=(2, 2), cmap=cmap,
                    dtype = 'prob', titles=['One run data only',
                                             '16 runs data only',
                                             'One run data + group probability map',
                                             'group probability map'])

    plt.suptitle(f'Individual vs. group, MDTB - sub {sub}')
    if save_folder is not None:
        sdir = res_dir + f'/1.indiv_vs_group' + save_folder
        plt.savefig(sdir + f'/indiv_group_plot_sub_{sub}.png', format='png')

    plt.show()

def plot_smooth_vs_unsmooth(D, model_type='Models_01'):
    plt.figure(figsize=(10, 10))
    crits = ['dcbc_group', 'dcbc_indiv']
    for i, c in enumerate(crits):
        plt.subplot(2, 2, i*2 + 1)
        sb.barplot(data=D, x='model_type', y=c, hue='smooth', errorbar="se")

        # if 'coserr' in c:
        #     plt.ylim(0.4, 1)
        plt.subplot(2, 2, i*2 + 2)
        sb.lineplot(data=D, x='K', y=c, hue='model_type', style="smooth",
                    errorbar=None, err_style="bars", markers=False)

    plt.suptitle(f'All datasets fusion')
    plt.tight_layout()
    plt.show()

def make_all_in_one_tsv(path, out_name):
    """Making all-in-one tsv file of evaluation
    Args:
        path: the path of the folder that contains
              all tsv files will be integrated
        out_name: output file name
    Returns:
        None
    """
    files = os.listdir(path)

    if not any(".tsv" in x for x in files):
        raise Exception('Input data file type must be .tsv file!')
    else:
        D = pd.DataFrame()
        for fname in files:
            res = pd.read_csv(path + f'/{fname}', delimiter='\t')

            # Making sure <PandasArray> mistakes are well-handled
            trains = res["train_data"].unique()
            print(trains)
            D = pd.concat([D, res], ignore_index=True)

        D.to_csv(out_name, sep='\t', index=False)

if __name__ == "__main__":
    ############# Fitting models #############
    # fit_smooth(model_type='03')
    fit_smooth(smooth=[None], model_type='04')

    ############# Evaluating models #############
    # model_name = []
    # K = [10,17,20,34,40,68,100]
    # model_type = ['03']
    # smooth = [0,2,3,7]
    # CV_setting = [('ses-s1', 'ses-s2'), ('ses-s2', 'ses-s1')]
    # D = pd.DataFrame()
    #
    # for (train_ses, test_ses) in CV_setting:
    #     for s in smooth:
    #         for mt in model_type:
    #             if s != 2:
    #                 model_name += [f'Models_{mt}/smoothed/asym_Md_space-MNISymC3_K-' \
    #                                f'{this_k}_smooth-{s}_{train_ses}'
    #                                for this_k in K]
    #             else:
    #                 model_name += [f'Models_{mt}/asym_Md_space-MNISymC3_K-{this_k}_{train_ses}'
    #                                for this_k in K]
    #
    #         results = eval_smoothed(model_name, t_datasets=['MDTB'], train_ses=train_ses,
    #                                 test_ses=test_ses, train_smooth=s)
    #         D = pd.concat([D, results], ignore_index=True)
    #
    # # Save file
    # wdir = model_dir + f'/Models/Evaluation'
    # fname = f'/eval_all_asym_K-10to100_Md_on_Sess_smooth.tsv'
    # D.to_csv(wdir + fname, index=False, sep='\t')

    ############# Plotting comparison #############
    # fname1 = f'/Models/Evaluation/eval_all_asym_K-10to100_MdUnSmoothed_on_otherDatasets.tsv'
    # fname2 = f'/Models/Evaluation/eval_all_asym_K-10to100_MdSmoothed_on_otherDatasets.tsv'
    # D1 = pd.read_csv(model_dir + fname1, delimiter='\t')
    # D2 = pd.read_csv(model_dir + fname2, delimiter='\t')
    #
    # D1['smooth'] = 2
    # D2['smooth'] = 7
    # D = pd.concat([D1, D2], ignore_index=True)
    # plot_smooth_vs_unsmooth(D)