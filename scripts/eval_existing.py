#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation on existing parcellations

Created on 3/1/2023 at 11:58 AM
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
import seaborn as sb
import sys
import time
import pickle
import json
from copy import copy,deepcopy
from itertools import combinations
from FusionModel.util import *
from FusionModel.evaluate import *
import FusionModel.similarity_colormap as sc
import FusionModel.hierarchical_clustering as cl

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
res_dir = model_dir + f'/Results' + '/5.all_datasets_fusion'

def _check_corr(dataset='HCP', type='Tseries', space='MNISymC3', sess=None,
                smooth=None, maxDist=50, binWidth=2):

    if sess is None:
        sess = 'all'

    # Calculate distance metric given by input atlas
    atlas, ainf = am.get_atlas(space, atlas_dir=base_dir + '/Atlases')
    dist = compute_dist(atlas.world.T, resolution=1)

    # Load dataset - all subjects
    tic = time.perf_counter()
    tdata, tinfo, tds = get_dataset(base_dir, dataset, atlas=space,
                                    sess=sess, type=type, subj=None,
                                    smooth=None if smooth==2 else smooth)
    toc = time.perf_counter()
    print(f'Done loading. Used {toc - tic:0.4f} seconds!')

    corr_values = []
    for sub in range(tdata.shape[0]):
        print(f'Subject {sub}', end=':')
        tic = time.perf_counter()
        _, corr = compute_rawCorr(maxDist=maxDist, binWidth=binWidth, dist=dist,
                                  func=tdata[sub].T)
        toc = time.perf_counter()
        print(f"{toc - tic:0.4f}s")
        corr_values.append(corr.cpu().numpy())

    # Calculate the mean and standard deviation across subjects for each timestamp
    mean_data = np.nanmean(np.stack(corr_values), axis=0)
    std_data = np.nanstd(np.stack(corr_values), axis=0)
    std_data = std_data / np.sqrt(np.stack(corr_values).shape[0])

    # Plot the mean data as a line and show the standard deviation with error bars
    plt.errorbar(np.arange(0, maxDist, binWidth)+1, mean_data, yerr=std_data, fmt='-',
                    capsize=4, capthick=1.5, elinewidth=1.5)
    plt.xlabel('Spatial distance (mm)')
    plt.ylabel('Correlation')
    plt.title(dataset)

def run_dcbc_existing(model_names, tdata, space, device=None, load_best=True, verbose=True):
    """ Calculates DCBC using a test_data set. The test data splitted into
        individual training and test set given by `train_indx` and `test_indx`.
        First we use individual training data to derive an individual
        parcellations (using the model) and evaluate it on test data.
        By calling function `calc_test_dcbc`, the Means of the parcels are
        always estimated on N-1 subjects and evaluated on the Nth left-out
        subject.
    Args:
        model_names (list or str): Name of model fit (tsv/pickle file)
        tdata (pt.Tensor or np.ndarray): test data set
        atlas (atlas_map): The atlas map object for calculating voxel distance
        train_indx (ndarray of index or boolean mask): index of individual
            training data
        test_indx (ndarray or index boolean mask): index of individual test
            data
        cond_vec (1d array): the condition vector in test-data info
        part_vec (1d array): partition vector in test-data info
        device (str): the device name to load trained model
        load_best (str): I don't know
    Returns:
        data-frame with model evalution of both group and individual DCBC
    Notes:
        This function is modified for DCBC group and individual evaluation
        in general case (not include IBC two sessions evaluation senario)
        requested by Jorn.
    """
    # Calculate distance metric given by input atlas
    atlas, ainf = am.get_atlas(space, atlas_dir=base_dir + '/Atlases')
    dist = compute_dist(atlas.world.T, resolution=1)
    # convert tdata to tensor
    if type(tdata) is np.ndarray:
        tdata = pt.tensor(tdata, dtype=pt.get_default_dtype())

    if not isinstance(model_names, list):
        model_names = [model_names]

    # Load atlas description json
    with open(atlas_dir + '/atlas_description.json', 'r') as f:
        T = json.load(f)

    space_dir = T[space]['dir']
    space_name = T[space]['space']
    num_subj = tdata.shape[0]
    results = pd.DataFrame()
    # Now loop over possible models we want to evaluate
    cw, cb = [], []
    for i, model_name in enumerate(model_names):
        print(f"Doing model {model_name}\n")
        if verbose:
            ut.report_cuda_memory()
        # load existing parcellation
        par = nb.load(atlas_dir +
                      f'/{space_dir}/atl-{model_name}_space-{space_name}_dseg.nii')
        Pgroup = pt.tensor(atlas.read_data(par, 0),
                           dtype=pt.get_default_dtype())
        Pgroup = pt.where(Pgroup==0, pt.tensor(float('nan')), Pgroup)
        this_res = pd.DataFrame()
        # ------------------------------------------
        # Now run the DCBC evaluation fo the group only
        dcbc_group, corr_w, corr_b = calc_test_dcbc(Pgroup, tdata, dist,
                                                    max_dist=110, bin_width=5,
                                                    trim_nan=True)
        cw.append(pt.stack(corr_w))
        cb.append(pt.stack(corr_b))
        # ------------------------------------------
        # Collect the information from the evaluation
        # in a data frame
        ev_df = pd.DataFrame({'model_name': [model_name] * num_subj,
                              'atlas': [space] * num_subj,
                              'K': [Pgroup.unique().shape[0]-1] * num_subj,
                              'train_data': [model_name] * num_subj,
                              'train_loglik': [np.nan] * num_subj,
                              'subj_num': np.arange(num_subj),
                              'common_kappa': [np.nan] * num_subj})
        # Add all the evaluations to the data frame
        ev_df['dcbc_group'] = dcbc_group.cpu()
        ev_df['dcbc_indiv'] = np.nan
        this_res = pd.concat([this_res, ev_df], ignore_index=True)

        # Concate model type
        this_res['model_type'] = model_name.split('/')[0]
        # Add a column it's session fit
        if len(model_name.split('ses-')) >= 2:
            this_res['test_sess'] = model_name.split('ses-')[1]
        else:
            this_res['test_sess'] = 'all'
        results = pd.concat([results, this_res], ignore_index=True)

    return results, cw, cb

def eval_existing(model_name, t_datasets=['MDTB','Pontine','Nishimoto'],
                  type=None, subj=None, out_name=None, save=True, plot_wb=True):
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
    results = pd.DataFrame()
    # Evaluate all single sessions on other datasets
    corrW, corrB = [], []
    for i, ds in enumerate(t_datasets):
        print(f'Testdata: {ds}\n')
        # Preparing atlas, cond_vec, part_vec
        tic = time.perf_counter()
        tdata, tinfo, tds = get_dataset(base_dir, ds, atlas='MNISymC3',
                                        sess='all', type=type[i], subj=subj[i])
        toc = time.perf_counter()
        print(f'Done loading. Used {toc - tic:0.4f} seconds!')

        if type[i] == 'Tseries':
            tds.cond_ind = 'time_id'

        res_dcbc, corr_w, corr_b = run_dcbc_existing(model_name, tdata, 'MNISymC3',
                                                     device='cuda')

        corrW.append(corr_w)
        corrB.append(corr_b)
        res_dcbc['test_data'] = ds
        results = pd.concat([results, res_dcbc], ignore_index=True)

    if save:
        # Save file
        wdir = model_dir + f'/Models/Evaluation'
        if out_name is None:
            fname = f'/eval_all_5existing_on_otherdatasets.tsv'
        else:
            fname = f'/eval_all_5existing_on_{out_name}.tsv'
        results.to_csv(wdir + fname, index=False, sep='\t')

    if plot_wb:
        return corrW, corrB

def plot_existing(D, t_data='MDTB', outName='7Tasks'):
    if t_data is not None:
        D = D.loc[D.test_data == t_data]

    plt.figure(figsize=(10, 5))
    crits = ['dcbc_group']
    for i, c in enumerate(crits):
        plt.subplot(1, 2, i + 1)
        sb.barplot(data=D, x='model_name', y=c, errorbar="se")

        plt.subplot(1, 2, i + 2)
        sb.barplot(data=D, x='test_data', y=c, hue='model_name',
                   errorbar="se")

    plt.suptitle(f'Existing parcellations, t_data={outName}')
    plt.tight_layout()
    plt.show()

def plot_existing_corr_wb(corr_w, corr_b, par_name=['Anatom'], datasets=['HCP'],
                          max_D=110, binwidth=5):
    x = np.arange(0, max_D, binwidth) + binwidth/2
    num_row = len(datasets)
    num_col = len(par_name)

    plt.figure(figsize=(5*num_col, 5*num_row))
    for i in range(num_row):
        this_cw = corr_w[i]
        this_cb = corr_b[i]
        for j in range(num_col):
            cw = this_cw[j].cpu().numpy()
            cb = this_cb[j].cpu().numpy()

            plt.subplot(num_row, num_col, i*num_col + j+1)
            # Calculate the mean and standard deviation across subjects for each timestamp
            se_w = np.nanstd(np.stack(cw), axis=0) / np.sqrt(np.stack(cw).shape[0])
            se_b = np.nanstd(np.stack(cb), axis=0) / np.sqrt(np.stack(cb).shape[0])

            # Plot the mean data as a line and show the standard deviation with error bars
            plt.errorbar(x, np.nanmean(np.stack(cw), axis=0), yerr=se_w, fmt='-', c='k',
                         capsize=1, capthick=0.5, elinewidth=0.8, label='within')
            plt.errorbar(x, np.nanmean(np.stack(cb), axis=0), yerr=se_b, fmt='-', c='r',
                         capsize=1, capthick=0.5, elinewidth=0.8, label='between')

            plt.legend()
            plt.xlabel('Spatial distance (mm)')
            plt.ylabel('Correlation')
            plt.title(par_name[j] + datasets[i])

    plt.suptitle(f'Existing parcellations, t_data={datasets}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ############# Checking raw correlation #############
    # test_datasets_list = [0,1,2,3,4,5,6,7]
    # T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    # d_names = T.name.to_numpy()[test_datasets_list]
    # types = np.append(T.default_type.to_numpy()[test_datasets_list][:-1], 'Tseries')
    # plt.figure(figsize=(20,10))
    # for i in test_datasets_list:
    #     plt.subplot(2, 4, i+1)
    #     _check_corr(dataset=d_names[i], type=types[i], space='MNISymC3', sess=None,
    #                 smooth=None, maxDist=110, binWidth=5)
    #
    # plt.tight_layout()
    # plt.show()

    ############# Evaluating models #############
    test_datasets_list = [7,7]
    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    num_subj = T.return_nsubj.to_numpy()[test_datasets_list]
    sub_list = [np.arange(c) for c in num_subj]

    # types = T.default_type.to_numpy()[test_datasets_list]
    types = ['Tseries', 'Tseries']
    sub_list = [np.arange(0, 100, 2), np.arange(0, 100, 2)+1]

    # Making half hcp resting subjects data
    # sub_list = [np.arange(c) for c in num_subj[:-2]]
    # hcp_train = np.arange(0, num_subj[-1], 2) + 1
    # hcp_test = np.arange(0, num_subj[-1], 2)
    # sub_list += [hcp_test, hcp_train]

    # 1. Save as tsv
    # eval_existing(['Anatom', 'Buckner7', 'Buckner17', 'Ji10', 'MDTB10'],
    #               t_datasets=T.name.to_numpy()[test_datasets_list],
    #               type=types, subj=sub_list, out_name='hcpTs')

    # 2. Check within and between curve
    corrW, corrB = eval_existing(['Anatom', 'Buckner7', 'Buckner17', 'Ji10', 'MDTB10'],
                                 t_datasets=T.name.to_numpy()[test_datasets_list],
                                 type=types, subj=sub_list, out_name='hcpTs',
                                 save=False, plot_wb=True)
    plot_existing_corr_wb(corrW, corrB, par_name=['Anatom', 'Buckner7',
                                                  'Buckner17', 'Ji10', 'MDTB10'],
                          datasets=['HCP', 'HCP'])

    # ############# Plot evaluation #############
    fname = f'/Models/Evaluation/eval_all_5existing_on_hcpTs.tsv'
    D = pd.read_csv(model_dir + fname, delimiter='\t')
    plot_existing(D, t_data=None, outName='hcpTs')

    ############# Plot fusion atlas #############
    # Making color map
    K = 34
    # fname = [f'/Models_03/asym_PoNiIbWmDeSo_space-MNISymC3_K-{K}',
    #          f'/Models_03/leaveNout/asym_Hc_space-MNISymC3_K-{K}_hcpOdd',
    #          f'/Models_03/leaveNout/asym_PoNiIbWmDeSoHc_space-MNISymC3_K-{K}_hcpOdd']
    # colors = get_cmap(f'/Models_03/asym_PoNiIbWmDeSo_space-MNISymC3_K-{K}')
    #
    # plt.figure(figsize=(20, 10))
    # plot_model_parcel(fname, [1, 3], cmap=colors, align=True, device='cuda')
    # plt.show()