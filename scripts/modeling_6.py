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
import FusionModel.util as ut
from FusionModel.evaluate import *
import FusionModel.similarity_colormap as sc
import FusionModel.hierarchical_clustering as cl
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

def result_6_eval(model_name, K='10', t_datasets=['MDTB','Pontine','Nishimoto'],
                  out_name=None):
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
    for i, this_ds in enumerate(t_datasets):
        print(f'Testdata: {this_ds}\n')
        # Preparing atlas, cond_vec, part_vec
        if this_ds.startswith('HCP'):
            ds = this_ds.split("_")[0]
            this_type = this_ds.split("_")[1]
            subj = np.arange(0, 100, 2)
        else:
            this_type = T.loc[T.name == ds]['default_type'].item()
            subj = None

        # Load testing data
        tic = time.perf_counter()
        tdata, tinfo, tds = get_dataset(base_dir, ds, atlas='MNISymC3',
                                        sess='all', type=this_type, subj=subj)
        toc = time.perf_counter()
        print(f'Done loading. Used {toc - tic:0.4f} seconds!')

        if this_type == 'Tseries':
            cond_vec = tinfo['time_id'].values.reshape(-1, )
        else:
            # default from dataset class
            cond_vec = tinfo[tds.cond_ind].values.reshape(-1, )

        part_vec = tinfo['half'].values
        # part_vec = np.ones((tinfo.shape[0],), dtype=int)
        CV_setting = [('half', 1), ('half', 2)]

        ################ CV starts here ################
        atlas, _ = am.get_atlas('MNISymC3', atlas_dir=base_dir + '/Atlases')
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
            res_dcbc['test_data'] = t_datasets[i]
            results = pd.concat([results, res_dcbc], ignore_index=True)

    # Save file
    wdir = model_dir + f'/Models/Evaluation'
    if out_name is None:
        return results
    else:
        fname = f'/eval_all_asym_K-{K}_{out_name}_on_HcEven.tsv'
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
            writein_dir = ut.model_dir + f'/Models/Models_{t}/leaveNout'
            dataname = ''.join(T.two_letter_code[datasets_list])
            nam = f'/asym_{dataname}_space-MNISymC3_K-{k}_hcpOdd'
            if Path(writein_dir + nam + '.tsv').exists():
                wdir, fname, info, models = fit_all(set_ind=datasets_list, K=k,
                                                    repeats=100, model_type=t,
                                                    sym_type=sym_type,
                                                    subj_list=sub_list,
                                                    space=space)
                fname = fname + f'_hcpOdd'
                info.to_csv(wdir + fname + '.tsv', sep='\t')
                with open(wdir + fname + '.pickle', 'wb') as file:
                    pickle.dump(models, file)
            else:
                print(f"Already fitted {dataname}, K={k}, Type={t}...")

def plot_result_6(D, t_data='MDTB'):
    D = D.replace(["['Pontine' 'Nishimoto' 'IBC' 'WMFS' 'Demand' 'Somatotopic' 'HCP']",
                   "['Pontine' 'Nishimoto' 'IBC' 'WMFS' 'Demand' 'Somatotopic']",
                   "['HCP']"], ['task+rest', 'task', 'rest'])
    D = D.loc[D.test_data == t_data]

    plt.figure(figsize=(10, 10))
    crits = ['dcbc_group', 'dcbc_indiv']
    for i, c in enumerate(crits):
        plt.subplot(2, 2, i*2 + 1)
        sb.barplot(data=D, x='model_type', y=c, hue='train_data',
                   hue_order=['task','rest','task+rest'], errorbar="se")

        # if 'coserr' in c:
        #     plt.ylim(0.4, 1)
        plt.subplot(2, 2, i*2 + 2)
        sb.lineplot(data=D, x='K', y=c, hue='train_data',
                    hue_order=['task','rest','task+rest'],
                    style="model_type", errorbar='se', markers=False)

    plt.suptitle(f'Task, rest, task+rest, test_data={t_data}')
    plt.tight_layout()
    plt.show()

def plot_loo_task(D, t_data=['MDTB'], model_type='03'):
    num_td = len(t_data)
    D = D.loc[D.model_type == f'Models_{model_type}']
    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')

    new_D = pd.DataFrame()
    plt.figure(figsize=(5*num_td, 10))
    for i in t_data:
        this_D = D.loc[D.test_data == T.name[i]]

        # Replace training dataset name to task, rest, task+rest
        datasets_list = [0, 1, 2, 3, 4, 5, 6]
        datasets_list.remove(i)
        tasks_name = T.name.to_numpy()[datasets_list]
        all_name = T.name.to_numpy()[datasets_list+[7]]
        rest_name = T.name.to_numpy()[[7]]

        strings = this_D.train_data.unique()
        for st in strings:
            if 'PandasArray' in st:
                this_D = this_D.replace([st], [str(tasks_name)])

        this_D = this_D.replace([str(all_name), str(tasks_name), str(rest_name)],
                                ['task+rest', 'task', 'rest'])


        plt.subplot(2, num_td, i+1)
        sb.lineplot(data=this_D, x='K', y='dcbc_group', hue='train_data',
                    hue_order=['task', 'rest', 'task+rest'], errorbar='se', markers=False)
        plt.title(T.name[i])

        plt.subplot(2, num_td, i+num_td+1)
        sb.lineplot(data=this_D, x='K', y='dcbc_indiv', hue='train_data',
                    hue_order=['task', 'rest', 'task+rest'], errorbar='se', markers=False)
        plt.title(T.name[i])

        new_D = pd.concat([new_D, this_D], ignore_index=True)

    plt.suptitle(f'Model {model_type}: Task, rest, task+rest, test_data=Tasks')
    plt.tight_layout()
    plt.show()

def plot_loo_rest(D, t_data=['HCP_Net69Run','HCP_Ico162Run'], model_type='03'):
    num_td = len(t_data)
    D = D.loc[D.model_type == f'Models_{model_type}']
    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')

    D = D.replace(["['MDTB' 'Pontine' 'Nishimoto' 'IBC' 'WMFS' 'Demand' 'Somatotopic' 'HCP']",
                   "['MDTB' 'Pontine' 'Nishimoto' 'IBC' 'WMFS' 'Demand' 'Somatotopic']",
                   "['HCP']"], ['task+rest', 'task', 'rest'])

    plt.figure(figsize=(10*num_td, 20))
    for i, td in enumerate(t_data):
        this_D = D.loc[D.test_data == td]

        plt.subplot(2, num_td, i+1)
        sb.lineplot(data=this_D, x='K', y='dcbc_group', hue='train_data',
                    hue_order=['task', 'rest', 'task+rest'], errorbar='se', markers=False)
        plt.title(td)

        plt.subplot(2, num_td, i+num_td+1)
        sb.lineplot(data=this_D, x='K', y='dcbc_indiv', hue='train_data',
                    hue_order=['task', 'rest', 'task+rest'], errorbar='se', markers=False)
        plt.title(td)

    plt.suptitle(f'Model {model_type}: Task, rest, task+rest, test_data=rest')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ############# Fitting models #############
    for i in range(0,7):
        datasets_list = [0, 1, 2, 3, 4, 5, 6, 7]
        datasets_list.remove(i)
        print(datasets_list)
        fit_rest_vs_task(datasets_list=datasets_list, K=[10,17,20,34,40,68,100],
                         sym_type=['asym'], model_type=['03','04'], space='MNISymC3')

    ############# Evaluating models (on task) #############
    # model_type = ['03', '04']
    # K = [10,17,20,34,40,68,100]
    #
    # T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    # results = pd.DataFrame()
    # for i in range(0, 7):
    #     model_name = []
    #     datasets_list = [0, 1, 2, 3, 4, 5, 6]
    #     datasets_list.remove(i)
    #     dataname = ''.join(T.two_letter_code[datasets_list])
    #     # Pure Task
    #     model_name += [f'Models_{mt}/asym_{dataname}_space-MNISymC3_K-{this_k}'
    #                    for this_k in K for mt in model_type]
    #     # Task+rest
    #     model_name += [f'Models_{mt}/leaveNout/asym_{dataname}Hc_space-MNISymC3_K-{this_k}_hcpOdd'
    #                    for this_k in K for mt in model_type]
    #
    #     # Pure Rest
    #     model_name += [f'Models_{mt}/leaveNout/asym_Hc_space-MNISymC3_K-{this_k}_hcpOdd'
    #                    for this_k in K for mt in model_type]
    #
    #     res = result_6_eval(model_name, K='10to100', t_datasets=[T.name[i]], out_name=None)
    #     results = pd.concat([results, res], ignore_index=True)
    #
    # # Save file
    # wdir = model_dir + f'/Models/Evaluation'
    # results.to_csv(wdir + '/eval_all_asym_K-10to100_7taskHcOdd_on_looTask.tsv', index=False,
    #                sep='\t')

    ############# Evaluating models (on rest) #############
    # model_type = ['03', '04']
    # K = [10, 17, 20, 34, 40, 68, 100]
    #
    # model_name = []
    # T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    # datasets_list = [0, 1, 2, 3, 4, 5, 6]
    #
    # dataname = ''.join(T.two_letter_code[datasets_list])
    # # Pure Task
    # model_name += [f'Models_{mt}/asym_{dataname}_space-MNISymC3_K-{this_k}'
    #                for this_k in K for mt in model_type]
    # # Task+rest
    # model_name += [f'Models_{mt}/leaveNout/asym_{dataname}Hc_space-MNISymC3_K-{this_k}_hcpOdd'
    #                for this_k in K for mt in model_type]
    #
    # # Pure Rest
    # model_name += [f'Models_{mt}/leaveNout/asym_Hc_space-MNISymC3_K-{this_k}_hcpOdd'
    #                for this_k in K for mt in model_type]
    #
    # result_6_eval(model_name, K='10to100', t_datasets=['HCP_Ico162Run','HCP_Net69Run'],
    #               out_name='7taskHcOdd')

    ############# Plot evaluation #############
    # 1. evaluation on Task
    fname = f'/Models/Evaluation/eval_all_asym_K-10to100_7taskHcOdd_on_looTask.tsv'
    D = pd.read_csv(model_dir + fname, delimiter='\t')
    # plot_result_6(D, t_data='HCP')
    plot_loo_task(D, t_data=[0,1,2,3,4,5,6], model_type='03')

    # 2. evaluation on rest
    fname = f'/Models/Evaluation/eval_all_asym_K-10to100_7taskHcOdd_on_HcEven.tsv'
    D = pd.read_csv(model_dir + fname, delimiter='\t')
    plot_loo_rest(D, model_type='03')


    ############# Plot fusion atlas #############
    # Making color map
    # K = 34
    # fname = [f'/Models_03/asym_PoNiIbWmDeSo_space-MNISymC3_K-{K}',
    #          f'/Models_03/leaveNout/asym_Hc_space-MNISymC3_K-{K}_hcpOdd',
    #          f'/Models_03/leaveNout/asym_PoNiIbWmDeSoHc_space-MNISymC3_K-{K}_hcpOdd']
    # colors = get_cmap(f'/Models_03/asym_PoNiIbWmDeSo_space-MNISymC3_K-{K}')
    #
    # plt.figure(figsize=(20, 10))
    # plot_model_parcel(fname, [1, 3], cmap=colors, align=True, device='cuda')
    # plt.show()