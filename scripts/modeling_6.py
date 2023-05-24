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
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.evaluation as ev

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

def result_6_eval(model_name, K='10', t_datasets=['MDTB','Pontine','Nishimoto'],
                  out_name=None, add_zero=False):
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
            ds = this_ds
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
            # Split tdata into individual training / testing
            train_dat = tdata[:,train_indx,:]
            test_dat = tdata[:,test_indx,:]

            if add_zero and this_ds in ['Demand','WMFS','Somatotopic']:
                # Add zero as rest to the data
                train_dat = np.concatenate((train_dat,
                                            np.zeros((train_dat.shape[0], 1, train_dat.shape[2]))),
                                           axis=1)
                test_dat = np.concatenate((test_dat,
                                            np.zeros((test_dat.shape[0], 1, test_dat.shape[2]))),
                                           axis=1)

                # condition / partition vector
                train_cond_vec = np.append(cond_vec[train_indx], cond_vec[train_indx][-1] + 1)
                train_part_vec = np.append(part_vec[train_indx], part_vec[train_indx][-1])
            else:
                # condition / partition vector
                train_cond_vec = cond_vec[train_indx]
                train_part_vec = part_vec[train_indx]

            # 1. Run DCBC individual
            dist = compute_dist(atlas.world.T, resolution=1)
            res_dcbc = run_dcbc(model_name, train_dat,
                                test_dat, dist,
                                cond_vec=train_cond_vec,
                                part_vec=train_part_vec,
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

def fit_rest_vs_task(datasets_list=[1,7], K=[34], sym_type=['asym'],
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
            writein_dir = ut.model_dir + f'/Models/Models_{t}'
            dataname = ''.join(T.two_letter_code[datasets_list])
            nam = f'/asym_{dataname}_space-MNISymC3_K-{k}'
            if not Path(writein_dir + nam + '.tsv').exists():
                # wdir, fname, info, models = fit_all(set_ind=datasets_list, K=k,
                #                                     repeats=100, model_type=t,
                #                                     sym_type=sym_type,
                #                                     subj_list=sub_list,
                #                                     space=space)
                fit_all(set_ind=datasets_list, K=k, repeats=100, model_type=t,
                        sym_type=sym_type, space=space)
                # fname = fname + f'_hcpOdd'
                # info.to_csv(wdir + fname + '.tsv', sep='\t')
                # with open(wdir + fname + '.pickle', 'wb') as file:
                #     pickle.dump(models, file)
            else:
                print(f"Already fitted {dataname}, K={k}, Type={t}...")

def plot_result_6(D, t_data='MDTB'):
    """Leave MDTB as a clean benchmark test set
    Args:
        D: dataframe
        t_data: name of the test dataset

    Returns:
        plot
    """
    D = D.loc[D.test_data == t_data]
    D = D.loc[D.train_data != "['MDTB' 'Pontine' 'Nishimoto' 'IBC' 'WMFS' 'Demand' 'Somatotopic']"]
    D = D.replace(["['Pontine' 'Nishimoto' 'IBC' 'WMFS' 'Demand' 'Somatotopic' 'HCP']",
                   "['Pontine' 'Nishimoto' 'IBC' 'WMFS' 'Demand' 'Somatotopic']",
                   "['HCP']"], ['tasks+rest', 'tasks', 'HCP'])

    plt.figure(figsize=(10, 10))
    crits = ['dcbc_group', 'dcbc_indiv']
    for i, c in enumerate(crits):
        plt.subplot(2, 2, i*2 + 1)
        sb.barplot(data=D, x='train_data', y=c, hue='model_type', errorbar="se")

        if c == 'dcbc_group':
            plt.ylim(0.02, 0.1)
        elif c == 'dcbc_indiv':
            plt.ylim(0.1, 0.18)
        plt.xticks(rotation=45)

        plt.subplot(2, 2, i*2 + 2)
        sb.lineplot(data=D, x='K', y=c, hue='train_data',
                    style="model_type", errorbar='se', markers=False)

    plt.suptitle(f'Task, rest, task+rest, test_data={t_data}')
    plt.tight_layout()
    plt.show()

def plot_loo_task_avrg(D, t_data=[0,1,2,3,4,5,6], save=False):
    num_td = len(t_data)
    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')

    new_D = pd.DataFrame()
    # fig, axes = plt.subplots(nrows=2, ncols=num_td, figsize=(5 * num_td, 10), sharey='row')
    for i, d_idx in enumerate(t_data):
        this_D = D.loc[D.test_data == T.name[d_idx]]

        # Replace training dataset name to task, rest, task+rest
        datasets_list = [0, 1, 2, 3, 4, 5, 6]
        datasets_list.remove(d_idx)
        tasks_name = T.name.to_numpy()[datasets_list]
        all_name = T.name.to_numpy()[datasets_list + [7]]
        rest_name = T.name.to_numpy()[[7]]

        for st in this_D.train_data.unique():
            if 'PandasArray' in st:
                this_D = this_D.replace([st], [str(tasks_name)])

        this_D = this_D.replace([str(all_name), str(tasks_name),
                                 str(all_name).replace(" ", ", "),
                                 str(tasks_name).replace(" ", ", "), str(rest_name)],
                                ['task+rest', 'task', 'task+rest', 'task', 'rest'])

        new_D = pd.concat([new_D, this_D], ignore_index=True)

    df1 = new_D.loc[new_D.model_type == 'Models_03'].reset_index()
    df2 = new_D.loc[new_D.model_type == 'Models_04'].reset_index()
    df_toadd = df1.copy()

    df_toadd.loc[:, 'dcbc_group'] = (df1.dcbc_group.to_numpy() + df2.dcbc_group.to_numpy())/2
    df_toadd.loc[:, 'dcbc_indiv'] = (df1.dcbc_indiv.to_numpy() + df2.dcbc_indiv.to_numpy())/2
    df_toadd['model_type'].replace(['Models_03'], 'Average', inplace=True)
    new_D = pd.concat([new_D, df_toadd], ignore_index=True)

    nums = [('Pontine',24), ('Nishimoto',32), ('IBC',38), ('WMFS',50), ('Demand',66),
            ('Somatotopic', 103)]
    for (td, i) in nums:
        new_D.loc[new_D.test_data == td, 'subj_num'] += i

    plt.figure(figsize=(10, 10))
    crits = ['dcbc_group', 'dcbc_indiv']
    for i, c in enumerate(crits):
        plt.subplot(2, 2, i * 2 + 1)
        sb.barplot(data=new_D, x='train_data', order=['rest','task','task+rest'],
                   y=c, hue='model_type', errorbar="se", palette=sb.color_palette()[1:4])
        if c == 'dcbc_group':
            plt.ylim(0.06, 0.11)
        elif c == 'dcbc_indiv':
            plt.ylim(0.19, 0.225)

        plt.subplot(2, 2, i * 2 + 2)
        sb.lineplot(data=new_D, x='K', y=c, hue='train_data',hue_order=['rest','task','task+rest'],
                    style="model_type", errorbar='se', markers=False,
                    palette=sb.color_palette()[1:4])

    plt.suptitle(f'Averaged leaveOneOut: Task, rest, task+rest, test_data=Tasks')
    plt.tight_layout()

    if save:
        plt.savefig('task_vs_rest_loo.pdf', format='pdf')
    plt.show()


def plot_loo_task(D, t_data=['MDTB'], model_type='03', average=False):
    num_td = len(t_data)
    D = D.loc[D.model_type == f'Models_{model_type}']
    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')

    new_D = pd.DataFrame()
    fig, axes = plt.subplots(nrows=2, ncols=num_td, figsize=(5 * num_td, 10), sharey='row')
    for i, d_idx in enumerate(t_data):
        this_D = D.loc[D.test_data == T.name[d_idx]]

        # Replace training dataset name to task, rest, task+rest
        datasets_list = [0, 1, 2, 3, 4, 5, 6]
        datasets_list.remove(d_idx)
        tasks_name = T.name.to_numpy()[datasets_list]
        all_name = T.name.to_numpy()[datasets_list + [7]]
        rest_name = T.name.to_numpy()[[7]]

        for st in this_D.train_data.unique():
            if 'PandasArray' in st:
                this_D = this_D.replace([st], [str(tasks_name)])

        this_D = this_D.replace([str(all_name), str(tasks_name),
                                 str(all_name).replace(" ", ", "),
                                 str(tasks_name).replace(" ", ", "), str(rest_name)],
                                ['task+rest', 'task', 'task+rest', 'task', 'rest'])

        # Set y-axis limits
        # y_max = max(this_D[['dcbc_group', 'dcbc_indiv']].max())
        # y_min = min(this_D[['dcbc_group', 'dcbc_indiv']].min())

        # Create line plot for group and individual DCBC scores
        sb.lineplot(ax=axes[0, i], data=this_D, x='K', y='dcbc_group', hue='train_data',
                    hue_order=['rest', 'task+rest', 'task'], errorbar='se', markers=False)
        axes[0, i].set_title(T.name[d_idx])
        # axes[0, i].set_ylim(y_min, y_max)

        sb.lineplot(ax=axes[1, i], data=this_D, x='K', y='dcbc_indiv', hue='train_data',
                    hue_order=['rest', 'task+rest', 'task'], errorbar='se', markers=False)
        axes[1, i].set_title(T.name[d_idx])
        # axes[1, i].set_ylim(y_min, y_max)

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
                    hue_order=['rest', 'task+rest', 'task'], errorbar='se', markers=False)
        plt.title(td)

        plt.subplot(2, num_td, i+num_td+1)
        sb.lineplot(data=this_D, x='K', y='dcbc_indiv', hue='train_data',
                    hue_order=['rest', 'task+rest', 'task'], errorbar='se', markers=False)
        plt.title(td)

    plt.suptitle(f'Model {model_type}: Task, rest, task+rest, test_data=rest')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ############# Fitting models #############
    # for i in [6,5,4,3,2,1,0]:
    #     datasets_list = [0, 1, 2, 3, 4, 5, 6, 7]
    #     datasets_list.remove(i)
    #     print(datasets_list)
    #     fit_rest_vs_task(datasets_list=datasets_list, K=[10,17,20,34,40,68,100],
    #                      sym_type=['asym'], model_type=['03'], space='MNISymC3')

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
    # fname = f'/Models/Evaluation/eval_all_asym_K-10to100_7taskHcOdd_on_looTask.tsv'
    # # fname = f'/Models/Evaluation/eval_all_asym_K-10to100_MdPoNiIbWmSoHcOdd_on_De.tsv'
    # D = pd.read_csv(model_dir + fname, delimiter='\t')
    # # plot_loo_task(D, t_data=[0,1,2,3,4,5,6], model_type='03')
    # plot_loo_task_avrg(D, save=True)

    # 2. evaluation on rest
    # fname = f'/Models/Evaluation/eval_all_asym_K-10to100_7taskHcOdd_on_HcEven.tsv'
    # D = pd.read_csv(model_dir + fname, delimiter='\t')
    # plot_loo_rest(D, model_type='03')

    # 3. Plot evaluation of result 6
    # fname1 = f'/Models/Evaluation/eval_all_asym_K-10to100_7taskHcOdd_on_looTask.tsv'
    # fname2 = f'/Models/Evaluation/eval_dataset7_asym.tsv'
    # D1 = pd.read_csv(model_dir + fname1, delimiter='\t')
    # D2 = pd.read_csv(model_dir + fname2, delimiter='\t')
    # D2 = D2.drop(['coserr_group', 'coserr_floor', 'coserr_ind2', 'coserr_ind3'], axis=1)
    # D2.rename(columns={'session': 'test_sess'}, inplace=True)
    # D1 = D1[D2.columns]
    # D = pd.concat([D2, D1])
    # D = D.reindex(columns=D2.columns)
    # plot_result_6(D, t_data='MDTB')

    ############# Plot fusion atlas #############
    # Making color map
    K = [34]
    model_type = ['03']
    fname = [f'/Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC3_K-34',
             f'/Models_03/leaveNout/asym_MdPoNiIbWmDeSoHc_space-MNISymC3_K-34_hcpOdd',
             f'/Models_03/leaveNout/asym_Hc_space-MNISymC3_K-34_hcpOdd']
    colors = ut.get_cmap(f'/Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC3_K-34')
    # T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    # results = pd.DataFrame()
    # model_name = []
    # for i in range(0, 7):
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
    # # Pure Rest
    # model_name += [f'Models_{mt}/leaveNout/asym_Hc_space-MNISymC3_K-{this_k}_hcpOdd'
    #                for this_k in K for mt in model_type]
    #
    plt.figure(figsize=(10, 10))
    plot_model_parcel(fname, [1, 3], cmap=colors, align=True, device='cuda')
    plt.show()