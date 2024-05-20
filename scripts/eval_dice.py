#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproducibility test across individual parcellations

Created on 4/17/2024 at 5:02 PM
Author: dzhi
"""
import numpy as np
import torch as pt
import nibabel as nb
import nitools as nt
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.evaluation as hev
import util as ut

from set_globals import base_dir, model_dir

def make_network_plot(space='MNISymC3',
                      model_name='Models_03/asym_Md_space-MNISymC3_K-17'):
    atlas, _ = am.get_atlas(space)
    _, M = ut.load_batch_best(model_name, device='cuda')
    U_group_hard = pt.argmax(M.arrange.marginal_prob(), dim=0).cpu().numpy() + 1
    colors = ut.get_cmap(model_name)
    colors[12] = np.array([249 / 255, 178 / 255, 247 / 255, 1.])

    U_parcels = [U_group_hard]
    for i in range(1,18):
        this_U = np.copy(U_group_hard)
        this_U[this_U != i] = 0
        U_parcels.append(this_U)

    U_parcels = np.stack(U_parcels)
    plt.figure(figsize=(50, 25))
    ut.plot_multi_flat(U_parcels, atlas.name, grid=(3, 6),
                    cmap=colors, dtype='label', titles=[f"network_{i}" for i in range(18)])
    plt.savefig('asym_Md_space-MNISymC3_K-17_Networks.png', format='png')
    plt.show()


if __name__ == "__main__":
    ## Step 1: Get data set and train the individual maps
    atlas, _ = am.get_atlas('MNISymC3')
    data, info, tds = ds.get_dataset(base_dir, 'MDTB', atlas=atlas.name, subj=None)
    tdata, cond_v, part_v, sub_ind = fm.prep_datasets(data, info.sess,
                                                      info['cond_num_uni'].values,
                                                      info['half'].values,
                                                      join_sess=False,
                                                      join_sess_part=False)

    ##### Option1: Use full model E_step
    _, M = ut.load_batch_best('Models_03/asym_Md_space-MNISymC3_K-17', device='cuda')
    U_group_hard = pt.argmax(M.arrange.marginal_prob(), dim=0).cpu().numpy() + 1
    M.initialize(tdata)
    # emloglik = M.collect_evidence([e.Estep() for e in M.emissions])
    U_indiv = M.Estep()[0]
    indiv_par_1 = U_indiv[M.subj_ind[0]]
    indiv_par_2 = U_indiv[M.subj_ind[1]]

    ##### Option2: pipeline
    ## Step 1: Loading a pre-trained group model
    # model_name = f'/Models/Models_03/asym_Md_space-MNISymC3_K-17'
    # U, minfo = ar.load_group_parcellation(model_dir + model_name, device='cuda')
    # ar_model = ar.build_arrangement_model(U, prior_type='logpi', atlas=atlas,
    #                                       sym_type='asym')
    # Get indiv parcellation from first half
    # indiv_par_1, _, M1 = fm.get_indiv_parcellation(ar_model, atlas, [tdata[0]],
    #                                               [cond_v[0]], [part_v[0]],
    #                                               [sub_ind[0]], sym_type='asym',
    #                                               em_params={'uniform_kappa': True})
    # # Get indiv parcellation from second half
    # indiv_par_2, _, M2 = fm.get_indiv_parcellation(ar_model, atlas, [tdata[1]],
    #                                               [cond_v[1]], [part_v[1]],
    #                                               [sub_ind[1]], sym_type='asym',
    #                                               em_params={'uniform_kappa': True})
    # em_params = {'num_subj': tdata[0].shape[0],
    #              'uniform_kappa': True,
    #              'subjects_equal_weight': False,
    #              'subject_specific_kappa': False,
    #              'parcel_specific_kappa': False}

    ## Step 3: Run dice coefficient between indivpar1 and indivpar2
    indiv_par_1 = pt.argmax(indiv_par_1, dim=1)+1
    indiv_par_2 = pt.argmax(indiv_par_2, dim=1)+1

    # colors = ut.get_cmap('Models_03/asym_Md_space-MNISymC3_K-17')
    # colors[12] = np.array([249 / 255, 178 / 255, 247 / 255, 1.])
    # plt.figure(figsize=(40, 15))
    # ut.plot_multi_flat(indiv_par_2.cpu().numpy(), atlas.name, grid=(3, 8),
    #                    cmap=colors, dtype='label', titles=[f"sub_{i+1}" for i in range(indiv_par_2.shape[0])])
    # plt.savefig('asym_Md_indiv_ses2.png', format='png')
    # plt.show()

    # 1. Within dice
    within_dice = [hev.dice_coefficient(indiv_par_1[i],
                                        indiv_par_2[i],
                                        label_matching=False).item()
                    for i in range(indiv_par_1.shape[0])]

    within_nmi = [hev.nmi(indiv_par_1[i].cpu().numpy(),
                          indiv_par_2[i].cpu().numpy()).item()
                   for i in range(indiv_par_1.shape[0])]

    within_ari = [hev.ARI(indiv_par_1[i],indiv_par_2[i]).item()
                   for i in range(indiv_par_1.shape[0])]

    num_subj = indiv_par_1.shape[0]
    ev_df = pd.DataFrame({'atlas': [atlas.name] * num_subj,
                          'K': [M.K] * num_subj,
                          'networks': ['all'] * num_subj,
                          'subj': np.arange(0,24)})
    ev_df['dice'] = within_dice
    ev_df['nmi'] = within_nmi
    ev_df['ari'] = within_ari
    ev_df['type'] = 'within'

    # for sub in range(24):
    #     networks_dice = hev.dice_coefficient(indiv_par_1[sub], indiv_par_2[sub],
    #                                          label_matching=False, separate=True)
    #     df = pd.DataFrame({'atlas': [atlas.name] * M.K,
    #                        'K': [M.K] * M.K,
    #                        'networks': np.arange(0,M.K),
    #                        'subj': [sub] * M.K})
    #     df['dice'] = networks_dice.cpu().numpy()
    #     ev_df = pd.concat([ev_df, df], ignore_index=True)

    # Between dice
    # plt.figure(figsize=(16, 8))
    # sb.boxplot(data=ev_df, x="networks", y="dice", width=0.5)

    print("1-Visual, 2-Motor, 3-dorsal attention, 4-ventral attention, 5-limbic, 6-frontoparietal, 7-DMN")
    # 2. Between dice
    between_dice = np.zeros((24,24))
    for i, par1 in enumerate(indiv_par_1):
        for j, par2 in enumerate(indiv_par_2):
            dice1 = hev.dice_coefficient(indiv_par_1[i], indiv_par_1[j],
                                         label_matching=False)
            dice2 = hev.dice_coefficient(indiv_par_2[i], indiv_par_2[j],
                                         label_matching=False)
            dice3 = hev.dice_coefficient(indiv_par_1[i], indiv_par_2[j],
                                         label_matching=False)
            dice4 = hev.dice_coefficient(indiv_par_1[j], indiv_par_2[i],
                                         label_matching=False)
            between_dice[i,j] = (dice1 + dice2 + dice3 + dice4)/4

    y_values = between_dice[np.triu_indices(between_dice.shape[0], k=1)]
    num_subj = y_values.shape[0]
    df = pd.DataFrame({'atlas': [atlas.name] * num_subj,
                          'K': [M.K] * num_subj,
                          'networks': ['all'] * num_subj,
                          'subj': np.arange(0, num_subj)})
    df['dice'] = y_values
    df['type'] = 'between'
    # Calculate mean and standard deviation of the y values
    mean_y = np.mean(y_values)
    std_y = np.std(y_values)
    plt.axhline(y=mean_y, color='r', linestyle='--', label='Mean')
    plt.fill_betweenx(y=[mean_y - std_y, mean_y + std_y],
                      x1=0, x2=17,
                      color='grey', alpha=0.3, label='Standard Deviation')
    plt.savefig('dice.pdf', format='pdf')
    plt.show()
    # Vectorize the upper triangular part
    vectorized_upper_triangle = upper_triangle.flatten()

