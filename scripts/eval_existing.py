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
        dcbc_group = calc_test_dcbc(Pgroup, tdata, dist, trim_nan=True)

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

    return results


def eval_existing(model_name, t_datasets=['MDTB','Pontine','Nishimoto'],
                  type=None, subj=None, out_name=None):
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

        res_dcbc = run_dcbc_existing(model_name, tdata, 'MNISymC3',
                                     device='cuda')
        res_dcbc['test_data'] = ds
        results = pd.concat([results, res_dcbc], ignore_index=True)

    # Save file
    wdir = model_dir + f'/Models/Evaluation'
    if out_name is None:
        fname = f'/eval_all_5existing_on_otherdatasets.tsv'
    else:
        fname = f'/eval_all_5existing_on_{out_name}.tsv'
    results.to_csv(wdir + fname, index=False, sep='\t')

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


if __name__ == "__main__":
    ############# Evaluating models #############
    # test_datasets_list = [7,7,7,7]
    # T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    # num_subj = T.return_nsubj.to_numpy()[test_datasets_list]
    # sub_list = [np.arange(c) for c in num_subj]
    #
    # # types = T.default_type.to_numpy()[test_datasets_list]
    # types = ['Tseries','Tseries','Tseries','Tseries']
    # sub_list = [np.arange(0, 100, 4), np.arange(0, 100, 4)+1,
    #             np.arange(0, 100, 4)+2, np.arange(0, 100, 4)+3]
    #
    # # Making half hcp resting subjects data
    # # sub_list = [np.arange(c) for c in num_subj[:-2]]
    # # hcp_train = np.arange(0, num_subj[-1], 2) + 1
    # # hcp_test = np.arange(0, num_subj[-1], 2)
    # # sub_list += [hcp_test, hcp_train]
    #
    # eval_existing(['Anatom', 'Buckner7', 'Buckner17', 'Ji10', 'MDTB10'],
    #               t_datasets=T.name.to_numpy()[test_datasets_list],
    #               type=types, subj=sub_list, out_name='hcpTs')

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