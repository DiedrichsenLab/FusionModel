#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result 5: Individual datasets vs. all datasets fusion

Created on 1/5/2023 at 11:14 AM
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
from FusionModel.util import *
from FusionModel.evaluate import *
import FusionModel.similarity_colormap as sc
import FusionModel.depreciated.hierarchical_clustering as cl

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

def result_5_eval(K=10, symmetric='asym', model_type=None, model_name=None,
                  t_datasets=None, subj=None, return_df=False):
    """Evaluate group and individual DCBC and coserr of all dataset fusion
       and any dataset training standalone on each of the datasets.
    Args:
        K: the number of parcels
        t_datasets (list): a list of test datasets
    Returns:
        Write in evaluation file
    """
    # Preparing model type, name, and test set is not given
    if model_type is None:
        model_type = ['01','02','03','04','05']

    if model_name is None:
        model_name = ['Md','Po','Ni','Ib','Wm','De','So','MdPoNiIbWmDeSo']

    if t_datasets is None:
        t_datasets = ['MDTB','Pontine','Nishimoto','IBC',
                      'WMFS','Demand','Somatotopic']

    results = pd.DataFrame()
    # Evaluate all single sessions on other datasets
    m_name = []
    for t in model_type:
        print(f'- Start evaluating Model_{t} - {model_name}...')
        m_name += [f'Models_{t}/{symmetric}_{nam}_space-MNISymC3_K-{K}' for nam in model_name]

    for ds in t_datasets:
        print(f'Testdata: {ds}\n')
        # 1. Run DCBC individual
        res_dcbc = run_dcbc_individual(m_name, ds, 'all', cond_ind=None,
                                       part_ind='half', indivtrain_ind='half',
                                       indivtrain_values=[1,2], subj=subj, device='cuda')
        # 2. Run coserr individual
        # res_coserr = run_prederror(m_name, ds, 'all', cond_ind=None,
        #                            part_ind='half', eval_types=['group', 'floor'],
        #                            indivtrain_ind='half', indivtrain_values=[1,2],
        #                            device='cuda')
        # 3. Merge the two dataframe
        # res = pd.merge(res_dcbc, res_coserr, how='outer')
        results = pd.concat([results, res_dcbc], ignore_index=True)

    if return_df:
        return results
    else:
        # Save file
        wdir = model_dir + f'/Models/Evaluation'
        fname = f'/eval_all_{symmetric}_K-{K}_datasetFusion.tsv'
        results.to_csv(wdir + fname, index=False, sep='\t')

def result_5_dice(K=10, symmetric='asym', model_type=None, model_name=None,
                  t_datasets='MDTB', subj=None):
    """Evaluate group and individual DCBC and coserr of all dataset fusion
       and any dataset training standalone on each of the datasets.
    Args:
        K: the number of parcels
        t_datasets (list): a list of test datasets
    Returns:
        Write in evaluation file
    """
    # Preparing model type, name, and test set is not given
    if model_type is None:
        model_type = ['01','02','03','04','05']
    if model_name is None:
        model_name = ['Md','Po','Ni','Ib','Wm','De','So','MdPoNiIbWmDeSo']

    this_type = 'Ico162Run' if t_datasets =='HCP' else None
    # this_type = 'CondAll' if t_datasets == 'MDTB' else None

    tdata, info, tds = ds.get_dataset(base_dir, t_datasets, atlas='MNISymC3',
                                       sess='all', subj=subj, type=this_type)

    tdata, cond_v, part_v, sub_ind = fm.prep_datasets(tdata, info.sess,
                                                      info['cond_num_uni'].values,
                                                      info['half'].values,
                                                      join_sess=False,
                                                      join_sess_part=False)

    # convert tdata to tensor
    atlas, _ = am.get_atlas('MNISymC3', atlas_dir=base_dir + '/Atlases')

    results = pd.DataFrame()
    # Evaluate all single sessions on other datasets
    for t in model_type:
        U_indiv = []
        print(f'- Start evaluating Model_{t} - {model_name}...')
        # Get all individual parcellations per model
        for mn in model_name:
            print(f'Testdata: {ds}\n')
            m_name = f'/Models/Models_{t}/{symmetric}_{mn}_space-MNISymC3_K-{K}'
            # 1. Run DCBC individual
            U, _ = ar.load_group_parcellation(model_dir + m_name, device='cuda')
            ar_model = ar.build_arrangement_model(U, prior_type='logpi', atlas=atlas,
                                                  sym_type='asym')

            indiv_par, _, M = fm.get_indiv_parcellation(ar_model, atlas, tdata,
                                                        cond_v, part_v,
                                                        [np.arange(0,24),np.arange(0,24)],
                                                        em_params={'uniform_kappa': False})
            U_indiv.append(indiv_par)

            results = pd.concat([results, res_dcbc], ignore_index=True)

    return results

def result_5_plot(fname, model_type='Models_01'):
    D = pd.read_csv(model_dir + fname, delimiter='\t')
    df = D.loc[(D['model_type'] == model_type)]
    crits = ['dcbc_group','dcbc_indiv','coserr_group',
             'coserr_floor','coserr_ind2','coserr_ind3']

    plt.figure(figsize=(15, 10))
    for i, c in enumerate(crits):
        plt.subplot(6, 1, i + 1)
        sb.barplot(data=df, x='test_data', y=c, hue='model_name', errorbar="se")
        plt.legend('')
        # if 'coserr' in c:
        #     plt.ylim(0.4, 1)

    plt.suptitle(f'All datasets fusion, {model_type}')
    plt.show()

def result_5_benchmark(fname, benchmark='MDTB', save=False):
    D = pd.read_csv(model_dir + fname, delimiter='\t')
    D = D.loc[(D['train_data']!="['MDTB' 'Pontine' 'Nishimoto' 'IBC' 'WMFS' 'Demand' "
                                "'Somatotopic']")]
    D = D.replace("['Pontine' 'Nishimoto' 'IBC' 'WMFS' 'Demand' 'Somatotopic']", 'all')

    nums = [('Pontine',24), ('Nishimoto',32), ('WMFS',38), ('Demand',54), ('Somatotopic', 91)]
    for (td, i) in nums:
        D.loc[D.test_data == td, 'subj_num'] += i

    df = D.loc[(D['test_data'] == benchmark)]
    crits = ['dcbc_group','dcbc_indiv']
    plt.figure(figsize=(10, 5))
    for i, c in enumerate(crits):
        plt.subplot(2, 1, i + 1)
        sb.barplot(data=df, x='train_data', y=c, order=["['Pontine']", "['Nishimoto']",
                                                        "['IBC']", "['WMFS']", "['Demand']",
                                                        "['Somatotopic']", 'all'],
                   hue='model_type', hue_order=['Models_03', 'Models_04'], errorbar="se",
                   palette=sb.color_palette()[1:3], width=0.7)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='small')
        # plt.xticks(rotation=45)
        if c == 'dcbc_group':
            plt.ylim(0.02, 0.13)
        elif c == 'dcbc_indiv':
            plt.ylim(0.1, 0.18)

    plt.suptitle(f'All datasets fusion, benchmark_dataset={benchmark}')
    plt.tight_layout()

    if save:
        plt.savefig(f'all_datasets_fusion.pdf', format='pdf')
    plt.show()

def plot_diffK(fname, hue="test_data", style="common_kappa"):
    D = pd.read_csv(model_dir + fname, delimiter='\t')

    df = D.loc[(D['model_type'] == 'Models_03')|(D['model_type'] == 'Models_04')]

    plt.figure(figsize=(12,15))
    crits = ['dcbc_group','dcbc_indiv','coserr_group',
             'coserr_floor','coserr_ind2','coserr_ind3']
    for i, c in enumerate(crits):
        plt.subplot(3, 2, i + 1)
        sb.lineplot(data=df, x="K", y=c, hue=style,
                    style_order=D[style].unique(),markers=True)
        # if i == len(crits)-1:
        #     plt.legend(loc='upper left')
        # else:
        #     plt.legend('')
        # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='small')

    plt.suptitle(f'all datasets fusion, diff K = {df.K.unique()}')
    plt.tight_layout()
    plt.show()

def plot_diffK_benchmark(fname, benchmark='MDTB', save=False):
    D = pd.read_csv(model_dir + fname, delimiter='\t')
    D = D.loc[(D['train_data']!="['MDTB' 'Pontine' 'Nishimoto' 'IBC' 'WMFS' 'Demand' "
                                "'Somatotopic']")]
    D = D.replace("['Pontine' 'Nishimoto' 'IBC' 'WMFS' 'Demand' 'Somatotopic']", 'all')
    df = D.loc[(D['test_data'] == benchmark)]
    # df1 = df.loc[df['indiv_train_kappa'] == True]
    # df2 = df.loc[df['indiv_train_kappa'] == False]

    plt.figure(figsize=(8, 5))
    crits = ['dcbc_group','dcbc_indiv']
    for i, c in enumerate(crits):
        plt.subplot(1, 2, i + 1)
        sb.lineplot(data=df, x='K', y=c,
                    hue='model_type', hue_order=['Models_03', 'Models_04'],
                    style="train_data",
                    style_order=["all", "['Pontine']", "['Nishimoto']", "['IBC']",
                                 "['WMFS']", "['Demand']", "['Somatotopic']"],
                    palette=sb.color_palette()[1:3],
                    errorbar=None, err_style="bars", markers=False)
        if c == 'dcbc_group':
            plt.ylim(0.01, 0.16)
        elif c == 'dcbc_indiv':
            plt.ylim(0.09, 0.20)

        if i == len(crits)-1:
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='small')
        else:
            plt.legend('')

    plt.suptitle(f'All datasets fusion, diff K = {df.K.unique()}')
    plt.tight_layout()

    if save:
        plt.savefig('all_datasets_fusion_diffK.pdf', format='pdf')
    plt.show()

def result_5_benchmark_ext(fname, K=17, benchmark='MDTB', save=False):
    D = pd.read_csv(model_dir + fname, delimiter='\t')

    D = D.loc[D.K == K]
    D = D.loc[(D['train_data']!="['MDTB' 'Pontine' 'Nishimoto' 'IBC' 'WMFS' 'Demand' "
                                "'Somatotopic']")]
    D = D.replace("['Pontine' 'Nishimoto' 'IBC' 'WMFS' 'Demand' 'Somatotopic']", 'all')

    nums = [('Pontine',24), ('Nishimoto',32), ('WMFS',38), ('Demand',54), ('Somatotopic', 91)]
    for (td, i) in nums:
        D.loc[D.test_data == td, 'subj_num'] += i

    df = D.loc[(D['test_data'] == benchmark)]
    df1 = df.loc[df['indiv_train_kappa'] == True]
    df2 = df.loc[df['indiv_train_kappa'] == False]

    crits = ['dcbc_indiv','dcbc_indiv','coserr_ind3','coserr_ind3']
    plt.figure(figsize=(10, 10))
    for i, c in enumerate(crits):
        plt.subplot(2, 2, i + 1)
        if i == 0 or i == 2:
            this_df = df1
            plt.title('indiv_train_kappa=True')
        else:
            this_df = df2
            plt.title('indiv_train_kappa=False')

        sb.barplot(data=this_df, x='train_data', y=c, order=["['Pontine']", "['Nishimoto']",
                                                        "['IBC']", "['WMFS']", "['Demand']",
                                                        "['Somatotopic']", 'all'],
                   hue='model_type', hue_order=['Models_03', 'Models_04'], errorbar="se",
                   palette=sb.color_palette()[1:3], width=0.7)
        # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='small')
        plt.legend(loc='upper right', fontsize='small')
        plt.xticks(rotation=45)
        if c == 'dcbc_indiv':
            plt.ylim(0.12, 0.22)
        elif c == 'coserr_ind3':
            plt.ylim(0.6, 0.72)

    plt.suptitle(f'All datasets fusion, benchmark_dataset={benchmark}, K={K}')
    plt.tight_layout()

    if save:
        plt.savefig(f'all_datasets_fusion.pdf', format='pdf')
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

def gmm_vs_vmf(m_name, save=True):
    t_datasets = ['Pontine', 'Nishimoto', 'IBC',
                  'WMFS', 'Demand', 'Somatotopic']

    results = pd.DataFrame()
    for ds in t_datasets:
        print(f'Testdata: {ds}\n')
        # 1. Run DCBC individual
        res_dcbc = run_dcbc_IBC(m_name, ds, 'all', cond_ind=None,
                                part_ind='half', indivtrain_ind='half',
                                indivtrain_values=[1,2], device='cuda')

        results = pd.concat([results, res_dcbc], ignore_index=True)

    if save:
        wdir = model_dir + f'/Models/Evaluation'
        fname = f'/eval_all_asym_K-10to100_VMFvsGMM.tsv'
        results.to_csv(wdir + fname, index=False, sep='\t')

def plot_gmm_vs_vmf(fname, benchmark=None, save=False):
    D = pd.read_csv(model_dir + fname, delimiter='\t')

    if benchmark is not None:
        D = D.loc[(D['test_data'] == benchmark)]

    crits = ['dcbc_group','dcbc_indiv']
    plt.figure(figsize=(10, 8))
    for i, c in enumerate(crits):
        plt.subplot(2, 1, i + 1)
        sb.barplot(data=D, x='test_data', y=c, hue='emission', errorbar="se",
                   palette=sb.color_palette()[1:2] + sb.color_palette()[9:10],
                   width=0.7)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='small')
        # plt.xticks(rotation=45)
        # if c == 'dcbc_group':
        #     plt.ylim(0.02, 0.12)
        # elif c == 'dcbc_indiv':
        #     plt.ylim(0.1, 0.18)

    plt.suptitle(f'GMM vs. VMF, tdata={benchmark}')
    plt.tight_layout()

    if save:
        plt.savefig(f'all_datasets_fusion.pdf', format='pdf')
    plt.show()

def eval_single_vs_fusion_on_rest():
    T = pd.read_csv(base_dir + '/dataset_description.tsv', sep='\t')
    D = pd.DataFrame()
    # datasets_list = [[0], [1], [2], [3], [4], [5], [6], [0, 1, 2, 3, 4, 5, 6]]
    datasets = [1,2,3,4,5,6]
    for k in [10,17,20,34,40,68,100]:
        for sub in [0,25,50,75]:
            datanames = T.two_letter_code[datasets].to_list()
            datanames += [''.join(T.two_letter_code[datasets])]
            res = result_5_eval(K=k, symmetric='asym', model_type=['03','04'],
                                model_name=datanames, t_datasets=[T.name[7]],
                                subj=np.arange(sub,sub+25), return_df=True)
            D = pd.concat([D, res], ignore_index=True)

    wdir = model_dir + f'/Models/Evaluation/asym'
    fname = f'/eval_indiv-fusion_asym_dcbc_k-10_to_100_HcIcos.tsv'
    D.to_csv(wdir + fname, index=False, sep='\t')

def eval_single_vs_fusion_dice():
    T = pd.read_csv(base_dir + '/dataset_description.tsv', sep='\t')
    D = pd.DataFrame()
    # datasets_list = [[0], [1], [2], [3], [4], [5], [6], [0, 1, 2, 3, 4, 5, 6]]
    datasets = [1,2,3,4,5,6]
    for k in [10,17,20,34,40,68,100]:
        # for sub in [0,25,50,75]:
        datanames = T.two_letter_code[datasets].to_list()
        datanames += [''.join(T.two_letter_code[datasets])]
        res = result_5_dice(K=k, symmetric='asym', model_type=['03','04'],
                            model_name=datanames, t_datasets='MDTB', subj=None)

        # res = result_5_dice(K=k, symmetric='asym', model_type=['03', '04'],
        #                     model_name=datanames, t_datasets='MDTB',
        #                     subj=np.arange(sub, sub + 25))

        D = pd.concat([D, res], ignore_index=True)

    wdir = model_dir + f'/Models/Evaluation/asym'
    fname = f'/eval_indiv-fusion_asym_dcbc_k-10_to_100_HcIcos.tsv'
    D.to_csv(wdir + fname, index=False, sep='\t')

    result_5_dice(K=10, symmetric='asym', model_type=None, model_name=None,
                  t_datasets='MDTB', subj=None)

if __name__ == "__main__":
    ############# Evaluating indiv datasets vs fusion #############
    # eval_single_vs_fusion_on_rest()
    eval_single_vs_fusion_dice()

    # T = pd.read_csv(base_dir + '/dataset_description.tsv', sep='\t')
    # D = pd.DataFrame()
    # # datasets_list = [[0], [1], [2], [3], [4], [5], [6], [0, 1, 2, 3, 4, 5, 6]]
    # for i in range(1):
    #     datasets = [0, 1, 2, 3, 4, 5, 6]
    #     datasets.remove(i)
    #     for k in [200, 300]:
    #         datanames = T.two_letter_code[datasets].to_list()
    #         datanames += [''.join(T.two_letter_code[datasets])]
    #         res = result_5_eval(K=k, symmetric='asym', model_type=['04'],
    #                             model_name=datanames, t_datasets=[T.name[i]],
    #                             return_df=True)
    #         D = pd.concat([D, res], ignore_index=True)
    # wdir = model_dir + f'/Models/Evaluation/asym'
    # fname = f'/eval_dataset7_asym_dcbc_k-200_to_300.tsv'
    # D.to_csv(wdir + fname, index=False, sep='\t')

    ############# Plot results #############
    fname = f'/Models/Evaluation/asym/eval_indiv-fusion_asym_dcbc_k-10_to_100_HcIcos_type3Indv.tsv'
    # result_5_plot(fname, model_type='Models_03')
    result_5_benchmark(fname, benchmark='HCP', save=False)
    plot_diffK_benchmark(fname, benchmark='HCP', save=False)
    # result_5_benchmark_ext(fname, benchmark='HCP', save=False)
    # plot_diffK_benchmark(fname, save=False)

    ############# Plot fusion atlas #############
    # indiv_dataset = ['Po', 'Ni', 'Ib', 'Wm', 'De', 'So']
    # m_fusion = ''.join(indiv_dataset)
    # datasets = [m_fusion] + indiv_dataset
    # # Making color map
    # colors = get_cmap(f'Models_04/asym_{m_fusion}_space-MNISymC3_K-34')
    # model_names = [f'Models_04/asym_{s}_space-MNISymC3_K-34' for s in datasets]
    #
    # # plt.figure(figsize=(20, 10))
    # plot_model_parcel(model_names, [2, 4], cmap=colors, align=True, device='cuda')
    # plt.show()

    ############# Plot fusion atlas (different K) #############
    for k in [200, 300]:
        fname = f'Models_04/asym_PoNiIbWmDeSo_space-MNISymC3_K-{k}'
        colors = get_cmap(fname)
        plot_model_parcel([fname], [1, 1], cmap=colors, align=True, device='cuda')
        plt.savefig(f'asym_PoNiIbWmDeSo_space-MNISymC3_K-{k}.png', format='png')
        plt.show()

    ############# GMM vs. vMF #############
    # m_name = ['Models_03/asym_Md_space-MNISymC3_K-10',
    #           'Models_03/asym_Md_space-MNISymC3_K-10_GMM',
    #           'Models_03/asym_Md_space-MNISymC3_K-17',
    #           'Models_03/asym_Md_space-MNISymC3_K-17_GMM',
    #           'Models_03/asym_Md_space-MNISymC3_K-20',
    #           'Models_03/asym_Md_space-MNISymC3_K-20_GMM',
    #           'Models_03/asym_Md_space-MNISymC3_K-34',
    #           'Models_03/asym_Md_space-MNISymC3_K-34_GMM',
    #           'Models_03/asym_Md_space-MNISymC3_K-40',
    #           'Models_03/asym_Md_space-MNISymC3_K-40_GMM',
    #           'Models_03/asym_Md_space-MNISymC3_K-68',
    #           'Models_03/asym_Md_space-MNISymC3_K-68_GMM',
    #           'Models_03/asym_Md_space-MNISymC3_K-100',
    #           'Models_03/asym_Md_space-MNISymC3_K-100_GMM']
    # gmm_vs_vmf(m_name, save=True)

    fname = f'/Models/Evaluation/eval_all_asym_K-10to100_VMFvsGMM.tsv'
    plot_gmm_vs_vmf(fname, save=True)
