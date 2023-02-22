#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Additional result 6 integrating resting state vs. purely task

Created on 2/17/2023 at 11:40 AM
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
from copy import copy,deepcopy
from itertools import combinations
import ProbabilisticParcellation.util as ut
from ProbabilisticParcellation.evaluate import *
import ProbabilisticParcellation.similarity_colormap as sc
import ProbabilisticParcellation.hierarchical_clustering as cl
from ProbabilisticParcellation.learn_fusion_gpu import *

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

def get_cmap(mname, load_best=True, sym=False):
    # Get model and atlas.
    fileparts = mname.split('/')
    split_mn = fileparts[-1].split('_')
    if load_best:
        info, model = load_batch_best(mname)
    else:
        info, model = load_batch_fit(mname)
    atlas, ainf = am.get_atlas(info.atlas, atlas_dir)

    # Get winner-take all parcels
    Prob = np.array(model.marginal_prob())
    parcel = Prob.argmax(axis=0) + 1

    # Get parcel similarity:
    w_cos_sim, _, _ = cl.parcel_similarity(model, plot=False, sym=sym)
    W = sc.calc_mds(w_cos_sim, center=True)

    # Define color anchors
    m, regions, colors = sc.get_target_points(atlas, parcel)
    cmap = sc.colormap_mds(W, target=(m, regions, colors), clusters=None, gamma=0.3)

    return cmap.colors

def result_6_eval(K=[10], model_type=['03','04'], t_datasets=['MDTB','Pontine','Nishimoto'],
                  test_ses=None):
    """Evaluate group and individual DCBC and coserr of IBC single
       sessions on all other test datasets.
    Args:
        K: the number of parcels
    Returns:
        Write in evaluation file
    """
    sess = DataSetIBC(base_dir + '/IBC').sessions
    if test_ses is not None:
        sess = [test_ses]

    model_name = []
    # Making all IBC indiv sessions list
    for mt in model_type:
        model_name += [f'Models_{mt}/asym_Ib_space-MNISymC3_K-{this_k}_{s}'
                       for this_k in K for s in sess]

        # Additionally, add all IBC sessions fusion to list
        model_name += [f'Models_{mt}/asym_Ib_space-MNISymC3_K-{this_k}'
                       for this_k in K]

    results = pd.DataFrame()
    # Evaluate all single sessions on other datasets
    for ds in t_datasets:
        print(f'Testdata: {ds}\n')

        # Preparing atlas, cond_vec, part_vec
        atlas, _ = am.get_atlas('MNISymC3', atlas_dir=base_dir + '/Atlases')
        tdata, tinfo, tds = get_dataset(base_dir, ds, atlas='MNISymC3', sess='all')
        cond_vec = tinfo[tds.cond_ind].values.reshape(-1, ) # default from dataset class
        part_vec = tinfo['half'].values
        # part_vec = np.ones((tinfo.shape[0],), dtype=int)
        CV_setting = [('half', 1), ('half', 2)]

        ################ CV starts here ################
        for (indivtrain_ind, indivtrain_values) in CV_setting:
            # get train/test index for cross validation
            train_indx = tinfo[indivtrain_ind] == indivtrain_values
            test_indx = tinfo[indivtrain_ind] != indivtrain_values
            # 1. Run DCBC individual
            res_dcbc = run_dcbc(model_name, tdata, atlas,
                               train_indx=train_indx,
                               test_indx=test_indx,
                               cond_vec=cond_vec,
                               part_vec=part_vec,
                               device='cuda')
            res_dcbc['indivtrain_ind'] = indivtrain_ind
            res_dcbc['indivtrain_val'] = indivtrain_values
            res_dcbc['test_data'] = ds
            # 2. Run coserr individual
            # res_coserr = run_prederror(model_name, ds, 'all', cond_ind=None,
            #                            part_ind='half', eval_types=['group', 'floor'],
            #                            indivtrain_ind='half', indivtrain_values=[1,2],
            #                            device='cuda')
            # 3. Merge the two dataframe
            # res = pd.merge(res_dcbc, res_coserr, how='outer')
            results = pd.concat([results, res_dcbc], ignore_index=True)

    # Save file
    wdir = model_dir + f'/Models/Evaluation_01'
    if test_ses is not None:
        fname = f'/eval_all_asym_Ib_K-{K}_{test_ses}_on_otherDatasets.tsv'
    else:
        fname = f'/eval_all_asym_Ib_K-10_to_100_indivSess_on_otherDatasets.tsv'
    results.to_csv(wdir + fname, index=False, sep='\t')

def fit_rest_vs_task(datasets_list = [1,7], K=[34], sym_type=['asym'],
                     model_type=['03','04'], space='MNISymC3'):
    """Fitting model of task-datasets (MDTB out) + HCP (half subjects)

    Args:
        datasets_list: the dataset indices list
        K: number of parcels
        sym_type: atlas type
        model_type: fitting model types
        space: atlas space
    Returns:
        write in fitted model in .tsv and .pickle
    """
    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    num_subj = T.return_nsubj.to_numpy()[datasets_list]
    sub_list = [np.arange(c) for c in num_subj[:-1]]

    # Odd indices for training, Even for testing
    hcp_train = np.arange(0, num_subj[-1], 2) + 1
    hcp_test = np.arange(0, num_subj[-1], 2)

    sub_list += [hcp_train]
    for t in model_type:
        for k in K:
            wdir, fname, info, models = fit_all(set_ind=datasets_list, K=k,
                                                repeats=100, model_type=t,
                                                sym_type=sym_type,
                                                subj_list=sub_list,
                                                space=space)
            fname = fname + f'_hcpOdd'
            info.to_csv(wdir + fname + '.tsv', sep='\t')
            with open(wdir + fname + '.pickle', 'wb') as file:
                pickle.dump(models, file)


if __name__ == "__main__":
    ############# Fitting models #############
    # fit_rest_vs_task(datasets_list=[1,2,3,4,5,6,7], K=[10,17,20,34,40,68,100],
    #                  sym_type=['asym'], model_type=['03','04'], space='MNISymC3')

    ############# Plot fusion atlas #############
    # Making color map
    fname = f'/Models_03/leaveNout/asym_PoNiIbWmDeSoHc_space-MNISymC3_K-10_hcpOdd'
    colors = get_cmap(fname)

    # plt.figure(figsize=(20, 10))
    plot_model_parcel([fname], [1, 1], cmap=colors, align=True, device='cuda')
    plt.show()