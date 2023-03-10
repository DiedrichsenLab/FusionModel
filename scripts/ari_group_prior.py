#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute the adjust rand index between pair-wise
group parcellations of the trained model

Created on 3/6/2023 at 11:34 AM
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
from sklearn.manifold import MDS
import sklearn

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

def get_parcels(model_names):
    if not isinstance(model_names, list):
        model_names = [model_names]

    parcels = []
    for i, model_name in enumerate(model_names):
        try:
            # Load best model
            _, model = load_batch_best(f"{model_name}", device='cuda')
            Prop = model.marginal_prob()
            Pgroup = pt.argmax(Prop, dim=0) + 1
        except:
            try:
                atlas, _ = am.get_atlas('MNISymC3', atlas_dir=base_dir + '/Atlases')
                # load existing parcellation
                par = nb.load(atlas_dir + model_name)
                Pgroup = pt.tensor(atlas.read_data(par, 0) + 1,
                                   dtype=pt.get_default_dtype())
            except NameError:
                print('The input model is neither fitted model or existing parcellations!')
        finally:
            parcels.append(Pgroup)

    return parcels

def run_ari(parcels, titles=[], colors=[], mds=False, ret=False):
    num_parcels = len(parcels)

    corr = np.zeros((num_parcels, num_parcels))
    for i in range(num_parcels):
        for j in range(num_parcels):
            corr[i,j] = ev.ARI(parcels[i], parcels[j])
            # corr[i,j] = sklearn.metrics.adjusted_rand_score(parcels[i].cpu(), parcels[j].cpu())
    if ret:
        return corr
    else:
        if mds:
            # fig = plt.figure()
            # Perform MDS on the correlation matrix
            # mds = MDS(n_components=2, dissimilarity='precomputed')
            # pos = mds.fit_transform(1 - corr)

            # Calculate the eigenvalues and eigenvectors of the correlation coefficient matrix
            eigenvalues, eigenvectors = np.linalg.eig(corr)

            # Choose the two eigenvectors with the highest eigenvalues
            idx = eigenvalues.argsort()[::-1][:2]
            pos = eigenvectors[:, idx]

            # Plot the resulting points
            plt.scatter(pos[:, 0], pos[:, 1], c=colors)
            for j in range(pos.shape[0]):
                plt.text(pos[j, 0]+0.005, pos[j, 1], titles[j],
                         fontdict=dict(color=colors[j], alpha=0.5))

        else:
            # Plot the correlation matrix
            np.fill_diagonal(corr, 0)
            plt.imshow(corr, vmin=corr.min(), vmax=corr.max())

            # Add x and y axis labels
            plt.xticks(range(len(titles)), titles, rotation=45)
            plt.yticks(range(len(titles)), titles, rotation=45)
            plt.colorbar()

def plot_ari(model_type=['03'], K=[10], singleTask=False, singleRest=True, looTask=True,
             looCombined=True, allTask=False, all8=False, existing=False, mds=True,
             average=False, N=4):
    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')

    num_row = len(model_type)
    num_col = len(K)
    corrs = []
    plt.figure(figsize=(5 * num_col, 5 * num_row))
    for row, mt in enumerate(model_type):
        for col, k in enumerate(K):
            model_name, labels, colors = [], [], []
            for i in range(0, 7):
                datasets_list = [0, 1, 2, 3, 4, 5, 6]
                datasets_list.remove(i)
                dataname = ''.join(T.two_letter_code[datasets_list])

                if looTask:
                    # Pure Task
                    model_name += [f'Models_{mt}/asym_{dataname}_space-MNISymC3_K-{k}']
                    labels += [dataname]
                    colors += ['tab:green']

                if looCombined:
                    # Task+rest
                    model_name += [f'Models_{mt}/leaveNout/asym_{dataname}Hc_space-MNISymC3_K-'
                                   f'{k}_hcpOdd']
                    labels += [dataname+'Hc']
                    colors += ['tab:orange']

                if singleTask:
                    ts = T.two_letter_code[i]
                    model_name += [f'Models_{mt}/asym_{ts}_space-MNISymC3_K-{k}']
                    labels += [ts]
                    colors += ['tab:green']

            if allTask:
                model_name += [f'Models_{mt}/asym_MdPoNiIbWmDeSo_space-MNISymC3_K-{k}']
                labels += ['7Tasks']
                colors += ['tab:green']
            if singleRest:
                # Pure Rest
                model_name += [f'Models_{mt}/leaveNout/asym_Hc_space-MNISymC3_K-{k}_hcpOdd']
                labels += ['HCP']
                colors += ['tab:blue']
            if all8:
                # All 8 datasets
                model_name += [f'Models_{mt}/leaveNout/asym_MdPoNiIbWmDeSoHc_space-MNISymC3_K-{k}_hcpOdd']
                labels += ['7tasks+HCP']
                colors += ['tab:orange']
            if existing:
                # Existing
                model_name += ['/tpl-MNI152NLin2009cSymC/atl-Anatom_space-MNI152NLin2009cSymC_dseg.nii',
                               '/tpl-MNI152NLin2009cSymC/atl-Buckner7_space-MNI152NLin2009cSymC_dseg.nii',
                               '/tpl-MNI152NLin2009cSymC/atl-Buckner17_space-MNI152NLin2009cSymC_dseg.nii',
                               '/tpl-MNI152NLin2009cSymC/atl-Ji10_space-MNI152NLin2009cSymC_dseg.nii',
                               '/tpl-MNI152NLin2009cSymC/atl-MDTB10_space-MNI152NLin2009cSymC_dseg.nii']
                labels += ['Anatom', 'Buckner7', 'Buckner17', 'Ji10', 'MDTB10']
                colors += ['black', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:green']

            parcels = get_parcels(model_name)

            if average:
                corrs.append(run_ari(parcels, titles=labels, colors=colors, mds=mds, ret=True))
            else:
                plt.subplot(num_row, num_col, row * num_col + col + 1)
                run_ari(parcels, titles=labels, colors=colors, mds=mds)
                plt.title(f'Model {mt}, K={k}')

    if average:
        plt.clf()
        plt.figure(figsize=(8,8))
        mean_corr = np.stack(corrs[0:N]).mean(axis=0)

        if mds:
            # Calculate the eigenvalues and eigenvectors of the correlation coefficient matrix
            eigenvalues, eigenvectors = np.linalg.eig(mean_corr)

            # Choose the two eigenvectors with the highest eigenvalues
            idx = eigenvalues.argsort()[::-1][:2]
            pos = eigenvectors[:, idx]

            # Plot the resulting points
            plt.scatter(pos[:, 0], pos[:, 1], c=colors)
            for j in range(pos.shape[0]):
                plt.text(pos[j, 0] + 0.005, pos[j, 1], labels[j],
                         fontdict=dict(color=colors[j], alpha=0.5))
        else:
            # Plot the correlation matrix
            np.fill_diagonal(mean_corr, 0)
            plt.imshow(mean_corr, vmin=mean_corr.min(), vmax=mean_corr.max())

            # Add x and y axis labels
            plt.xticks(range(len(labels)), labels, rotation=45)
            plt.yticks(range(len(labels)), labels, rotation=45)
            plt.colorbar()

    else:
        plt.suptitle('ARI - MDS')
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    ############# Visualize ARI-MDS plot #############
    plot_ari(model_type=['03'], K=[10,17,20,34,40,68,100], singleTask=True, singleRest=True,
             looTask=False, looCombined=False, allTask=True, all8=True, existing=True, mds=False,
             average=True, N=7)

    ############# Similarity between datasets #############
    datasets = ['MDTB','Pontine','Nishimoto','IBC','WMFS','Demand','Somatotopic','HCP']
    rel = similarity_between_datasets(base_dir, datasets, atlas='MNISymC3',
                                      subtract_mean=True, voxel_wise=True)

    corr = np.ma.corrcoef(np.ma.masked_array(rel, np.isnan(rel)))
    plt.imshow(corr.data)
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(24,12))
    plot_multi_flat(rel, 'MNISymC3', grid=(2, 4), dtype='func',
                    cscale=None, colorbar=False, titles=datasets)
    plt.show()
