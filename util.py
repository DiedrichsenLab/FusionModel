import numpy as np
import nibabel as nb
import SUITPy as suit
import pickle, psutil
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as dt
import pandas as pd
import torch as pt
import json
import matplotlib.pyplot as plt
import HierarchBayesParcel.evaluation as ev
import HierarchBayesParcel.full_model as fm
from pathlib import Path
import FusionModel.similarity_colormap as sc
# import FusionModel.depreciated.hierarchical_clustering as cl
import nitools as nt
import scipy.io as spio
from scipy.sparse import block_diag, coo_matrix
import scipy.ndimage as snd

# Set directories for the entire project - just set here and import everywhere
# else
model_dir = '/data/tge/dzhi/Indiv_par'
home = str(Path.home())
if not Path(model_dir).exists():
    model_dir = 'Y:/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/home/dzhi/eris_mount/dzhi/Indiv_par'
if not Path(model_dir).exists():
    model_dir = '/srv/diedrichsen/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/cifs/diedrichsen/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/Users/callithrix/Documents/Projects/Functional_Fusion/'
if not Path(model_dir).exists():
    model_dir = '/Users/jdiedrichsen/Data/FunctionalFusion/'
if not Path(model_dir).exists():
    model_dir = str(Path(home, 'diedrichsen_data/data/Cerebellum/ProbabilisticParcellationModel'))
if not Path(model_dir).exists():
    raise (NameError('Could not find model_dir'))

base_dir = '/data/tge/Tian/UKBB_full/imaging'
if not Path(base_dir).exists():
    base_dir = '/home/dzhi/eris_mount/Tian/UKBB_full/imaging'
if not Path(base_dir).exists():
    base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/cifs/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = 'Y:\data\FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/Users/callithrix/Documents/Projects/Functional_Fusion/'
if not Path(base_dir).exists():
    base_dir = '/Users/jdiedrichsen/Data/FunctionalFusion/'
if not Path(base_dir).exists():
    base_dir = str(Path(home, 'diedrichsen_data/data/FunctionalFusion'))
if not Path(base_dir).exists():
    raise (NameError('Could not find base_dir'))

atlas_dir = base_dir + f'/Atlases'

# pytorch cuda global flag
pt.cuda.is_available = lambda : True
if pt.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
pt.set_default_device(DEVICE)
pt.set_default_dtype(pt.float32)

# Keep track of cuda memory
def report_cuda_memory():
    if pt.cuda.is_available():
        ma = pt.cuda.memory_allocated() / 1024 / 1024
        mma = pt.cuda.max_memory_allocated() / 1024 / 1024
        mr = pt.cuda.memory_reserved() / 1024 / 1024
        print(
            f'Allocated:{ma:.2f} MB, MaxAlloc:{mma:.2f} MB, Reserved {mr:.2f} MB')

# print the current CPU memory usage
def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"CPU Memory Usage: {mem_info.rss / (1024 ** 2):.2f} MB")

def cal_corr(Y_target, Y_source):
    """ Matches the rows of two Y_source matrix to Y_target
    Using row-wise correlation and matching the highest pairs
    consecutively
    Args:
        Y_target: Matrix to align to
        Y_source: Matrix that is being aligned
    Returns:
        indx: New indices, so that YSource[indx,:]~=Y_target
    """
    K = Y_target.shape[0]
    # Compute the row x row correlation matrix
    Y_tar = Y_target - Y_target.mean(dim=1, keepdim=True)
    Y_sou = Y_source - Y_source.mean(dim=1, keepdim=True)
    Cov = pt.matmul(Y_tar, Y_sou.t())
    Var1 = pt.sum(Y_tar * Y_tar, dim=1)
    Var2 = pt.sum(Y_sou * Y_sou, dim=1)
    Corr = Cov / pt.sqrt(pt.outer(Var1, Var2))

    return Corr


def load_batch_fit(fname, device=None):
    """ Loads a batch of fits and extracts marginal probability maps
    and mean vectors
    Args:
        fname (str): File name
    Returns:
        info: Data Frame with information
        models: List of models
    """
    wdir = model_dir + '/Models/'
    info = pd.read_csv(wdir + fname + '.tsv', sep='\t')
    with open(wdir + fname + '.pickle', 'rb') as file:
        models = pickle.load(file)

    if device is not None:
        for m in models:
            m.move_to(device)

    return info, models


def clear_batch(fname):
    """Ensures that pickle file does not contain superflous data
    Args:
        fname (): filename
    """
    wdir = base_dir + '/Models/'
    with open(wdir + fname + '.pickle', 'rb') as file:
        models = pickle.load(file)
    # Clear models
    for m in models:
        m.clear()

    with open(wdir + fname + '.pickle', 'wb') as file:
        pickle.dump(models, file)


def move_batch_to_device(fname, device='cpu'):
    """Overwrite all tensors in the batch fitted models
       from torch.cuda to the normal torch.Tensor for
       people who cannot use cuda.
    Args:
        fname (): filename
        device: the target device to store tensors
    """
    wdir = model_dir + '/Models/'
    with open(wdir + fname + '.pickle', 'rb') as file:
        models = pickle.load(file)

    # Recursively tensors to device
    for m in models:
        m.move_to(device=device)

    with open(wdir + fname + '.pickle', 'wb') as file:
        pickle.dump(models, file)

def load_batch_best(fname, device=None):
    """ Loads a batch of model fits and selects the best one
    Args:
        fname (str): File name
    """
    info, models = load_batch_fit(fname)

    j = info.loglik.argmax()

    best_model = models[j]
    if device is not None:
        best_model.move_to(device)

    info_reduced = info.iloc[j]
    return info_reduced, best_model

def get_fs32k_neighbours(include_self=True, remove_mw=True, return_type='pt_csr'):
    '''Compute the neighbouring matrix of cortical mash

    :param file: the cortical mash file (e.g surf.gii)
    :return:     the vertices neighbouring matrix, shape = [N,N],
                 N is the number of vertices

    '''
    atlas, info = am.get_atlas('fs32k', atlas_dir=base_dir + '/Atlases')
    neighbours = []

    for i, hemis in enumerate(['L', 'R']):
        mat = nb.load(atlas_dir + f'/tpl-fs32k/tpl-fs32k_hemi-{hemis}_sphere.surf.gii')
        surf = [x.data for x in mat.darrays]
        surf_vertices = surf[0]
        surf_faces = surf[1]

        neigh = np.zeros([surf_vertices.shape[0], surf_vertices.shape[0]], dtype=int)
        for idx in range(surf_vertices.shape[0]):
            # the surf faces usually shape of [N,3] because a typical mesh is a triangle
            connected_idx = np.where(np.any(surf_faces == idx, axis=1))[0]
            connected = np.unique(surf_faces[connected_idx,:])
            neigh[idx, connected] = 1

        if not include_self:
            np.fill_diagonal(neigh, 0)
        # Remove medial wall if applied
        if remove_mw:
            neigh = neigh[:, atlas.vertex[i]][atlas.vertex[i], :]

        neighbours.append(neigh)
        del neigh
    
    neighbours = block_diag(neighbours).toarray()

    # Return types
    if return_type == 'full':
        return neighbours
    elif return_type == 'sparse_coo':
        return coo_matrix(neighbours)
    elif return_type == 'pt_coo':
        return pt.sparse_coo_tensor(np.argwhere(neighbours != 0).T,
                                    pt.tensor(neighbours[neighbours != 0],
                                              dtype=pt.get_default_dtype()),
                                    neighbours.shape)
    elif return_type == 'pt_csr':
        neighbours = pt.sparse_coo_tensor(np.argwhere(neighbours != 0).T,
                                    pt.tensor(neighbours[neighbours != 0],
                                              dtype=pt.get_default_dtype()),
                                    neighbours.shape)
        return neighbours.to_sparse_csr()


def load_fs32k_dist(file_type='distGOD_sp', hemis='full', remove_mw=True,
                    device='cuda'):
    # Load distance metric of vertices pairs in fs32k template
    atlas, info = am.get_atlas('fs32k', atlas_dir=base_dir + '/Atlases')
    dist = spio.loadmat(base_dir + '/Atlases/{0}'.format(info['dir'])
                        + f'/{file_type}.mat')
    dist = dist['avrgDs']

    # Remove medial wall if applied
    if remove_mw:
        dist = dist.tocsr()[:, atlas.vertex[0]][atlas.vertex[0], :]

    # Concatenate both hemispheres if `full`
    if hemis == 'full':
        dist = block_diag((dist, dist))
    elif hemis == 'half':
        pass
    else:
        raise ValueError('Unknown input `hemis`, please specify whether '
                         'the distances are calculated for single hemisphere'
                         'or whole cortex!')

    # Convert the numpy sparse matrix to a PyTorch sparse tensor
    c_matrix = dist.tocoo()
    indices = pt.LongTensor(np.vstack((c_matrix.row, c_matrix.col)))
    values = pt.FloatTensor(c_matrix.data)
    shape = c_matrix.shape
    sparse_tensor = pt.sparse_coo_tensor(indices, values, pt.Size(shape))

    return sparse_tensor.to(device=device)

def get_fs32k_weights(file_type='distGOD_sp', hemis='full', remove_mw=True,
                      max_dist=None, kernel='gaussian', sigma=10, device='cuda'):
    # Load distance metric of vertices pairs in fs32k template
    atlas, info = am.get_atlas('fs32k', atlas_dir=base_dir + '/Atlases')
    dist = spio.loadmat(base_dir + '/Atlases/{0}'.format(info['dir'])
                        + f'/{file_type}.mat')
    dist = dist['avrgDs']

    # Remove medial wall and larger distances if applied
    if remove_mw:
        dist = dist[:, atlas.vertex[0]][atlas.vertex[0], :]
    if max_dist is not None:
        dist[dist > max_dist] = 0
        dist = coo_matrix(dist.todense())

    # Concatenate both hemispheres if `full`
    if hemis == 'full':
        dist = block_diag((dist, dist))
    elif hemis == 'half':
        pass
    else:
        raise ValueError('Unknown input `hemis`, please specify whether '
                         'the distances are calculated for single hemisphere'
                         'or whole cortex!')

    # Convert the numpy sparse matrix to a PyTorch sparse tensor
    coo_mat = dist.tocoo()
    indices = pt.LongTensor(np.vstack((coo_mat.row, coo_mat.col)))
    sparse_tensor = pt.sparse_coo_tensor(indices,
                                         pt.FloatTensor(coo_mat.data),
                                         pt.Size(coo_mat.shape))

    # create a diagonal tensor - fill by zeros
    N = sparse_tensor.shape[0]
    # diagonal_tensor = pt.sparse_coo_tensor(np.stack([np.arange(N), np.arange(N)]),
    #                                        pt.FloatTensor(np.full((N,), 0)),
    #                                        pt.Size([N, N]))

    # change the values of the matrix by given kernel
    weights = sparse_tensor.coalesce()
    values = weights._values()
    if kernel == 'gaussian':
        values = pt.exp(-values ** 2 / (2*sigma**2))
        values = (values - values.min()) / (values.max() - values.min())
    elif kernel == 'connectivity':
        values = pt.ones_like(values)
    else:
        raise ValueError('Unknown kernel type! '
                         'We currently support gaussina kernel only.')

    weights = pt.sparse_coo_tensor(weights._indices(), values, weights.shape)

    # Coalesce weights to remove the duplicate indices
    return weights.coalesce().to(device=device)

def get_colormap_from_lut(fname=base_dir + '/Atlases/tpl-SUIT/atl-MDTB10.lut'):
    """ Makes a color map from a *.lut file
    Args:
        fname (str): Name of Lut file

    Returns:
        _type_: _description_
    """
    color_info = pd.read_csv(fname, sep=' ', header=None)
    color_map = np.zeros((color_info.shape[0] + 1, 3))
    color_map[1:, :] = color_info.iloc[:, 1:4].to_numpy()
    return color_map

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

def write_model_to_labelcifti(mname, align=True, col_names=None, label_names=None,
                              label_RGBA=None, load='best', oname='', device='cuda'):

    if load == 'best':
        models, infos = [], []
        # Load models and produce titles
        for i, mn in enumerate(mname):
            info, model = load_batch_best(mn, device=device)
            models.append(model)
            infos.append(info)
        at_name = infos[0].atlas
    elif load == 'all':
        assert len(mname) == 1, 'Only one model can be loaded at a time!'
        infos, models = load_batch_fit(mname[0], device=device)
        at_name = infos.atlas[0]
    else:
        raise ValueError('Unknown load type! Please specify whether to load '
                            'the best model or all models.')

    # Align models if requested
    if isinstance(align, np.ndarray or pt.Tensor):
        if type(align) is np.ndarray:
            align = pt.tensor(align, dtype=pt.get_default_dtype(), device=device)
        indx = ev.matching_greedy(align, models[0].marginal_prob())
        Prob = models[0].marginal_prob()[indx,:].T
    elif align:
        Prob = ev.align_models(models, in_place=False)
    else:
        Prob = ev.extract_marginal_prob(models)

    atlas, _ = am.get_atlas(at_name, atlas_dir)
    # Get winner-take all parcels
    parcel = Prob.cpu().numpy().argmax(axis=1) + 1

    # Get the indices of the maximum values in each column
    # Prob = Prob.T
    # _, max_indices = pt.max(Prob, dim=0)
    # result = pt.full_like(Prob, float('nan'))
    # result[max_indices, pt.arange(Prob.size(1))] = Prob[max_indices, pt.arange(Prob.size(1))]


    if col_names is None:
        col_names = [f'col_{i+1}' for i in range(parcel.shape[0])]

    img = nt.make_label_cifti(parcel.T, atlas.get_brain_model_axis(),
                              column_names=col_names, 
                              label_names=label_names,
                              label_RGBA=label_RGBA)
    nb.save(img, model_dir + f'/Models/{oname}.dlabel.nii')

def plot_data_flat(data, atlas,
                   cmap=None,
                   dtype='label',
                   cscale=None,
                   labels=None,
                   render='matplotlib',
                   colorbar=False):
    """ Maps data from an atlas space to a full volume and
    from there onto the surface - then plots it.

    Args:
        data (_type_): _description_
        atlas (str): Atlas code ('SUIT3','MNISymC3',...)
        cmap (_type_, optional): Colormap. Defaults to None.
        dtype (str, optional): 'label' or 'func'
        cscale (_type_, optional): Color scale
        render (str, optional): 'matplotlib','plotly'

    Returns:
        ax: Axis / figure of plot
    """
    # Plot Data from a specific atlas space on the flatmap
    suit_atlas, ainf = am.get_atlas(atlas, base_dir + '/Atlases')
    Nifti = suit_atlas.data_to_nifti(data)

    # Mapping labels directly by the mode
    if dtype == 'label':
        surf_data = suit.flatmap.vol_to_surf(Nifti, stats='mode',
                                             space=ainf['normspace'], ignore_zeros=True)
        ax = suit.flatmap.plot(surf_data,
                               render=render,
                               cmap=cmap,
                               new_figure=False,
                               label_names=labels,
                               overlay_type='label',
                               colorbar=colorbar)
    # Plotting one series of functional data
    elif dtype == 'func':
        surf_data = suit.flatmap.vol_to_surf(Nifti, stats='nanmean',
                                             space=ainf['normspace'])
        ax = suit.flatmap.plot(surf_data,
                               render=render,
                               cmap=cmap,
                               cscale=cscale,
                               new_figure=False,
                               overlay_type='func',
                               colorbar=colorbar)
    # Mapping probabilities on the flatmap and then
    # determining a winner from this (slightly better than label)
    elif dtype == 'prob':
        surf_data = suit.flatmap.vol_to_surf(Nifti, stats='nanmean',
                                             space=ainf['normspace'])
        label = np.argmax(surf_data, axis=1) + 1
        ax = suit.flatmap.plot(label,
                               render=render,
                               cmap=cmap,
                               new_figure=False,
                               label_names=labels,
                               overlay_type='label',
                               colorbar=colorbar)
    else:
        raise (NameError('Unknown data type'))
    return ax


def plot_multi_flat(data, atlas, grid,
                    cmap=None,
                    dtype='label',
                    cscale=None,
                    titles=None,
                    colorbar=False,
                    save_fig=True,
                    save_under=None):
    """Plots a grid of flatmaps with some data

    Args:
        data (array or list): NxP array of data or list of NxP arrays of data (if plotting Probabilities)
        atlas (str): Atlas code ('SUIT3','MNISymC3',...)
        grid (tuple): (rows,cols) grid for subplot
        cmap (colormap or list): Color map or list of color maps. Defaults to None.
        dtype (str, optional):'label' or 'func'
        cscale (_type_, optional): Scale of data (None)
        titles (_type_, optional): _description_. Defaults to None.
    """
    if isinstance(data, np.ndarray):
        n_subplots = data.shape[0]
    elif isinstance(data, list):
        n_subplots = len(data)

    if not isinstance(cmap, list):
        cmap = [cmap] * n_subplots

    for i in np.arange(n_subplots):
        plt.subplot(grid[0], grid[1], i + 1)
        plot_data_flat(data[i], atlas,
                       cmap=cmap[i],
                       dtype=dtype,
                       cscale=cscale,
                       render='matplotlib',
                       colorbar=(i == 0) & colorbar)

        plt.title(titles[i])
        plt.tight_layout()
        # if titles is not None:
        #     # plt.title(titles[i])
            # if save_fig:
                # plt.savefig(f'{titles}.png')
            #     fname = f'rel_{titles[i]}.png'
        #     #     if save_under is not None:
        #     #         fname = save_under
        #     #     plt.savefig(fname, format='png')
        #     plt.savefig(f'rel_{titles[i]}_{i}.png', format='png',
        #                 bbox_inches='tight', pad_inches=0)


def hard_max(Prob):
    K, P = Prob.shape
    parcel = np.argmax(Prob, axis=0)
    U = np.zeros((K, P))
    U[parcel, np.arange(P)] = 1
    return U


def plot_model_pmaps(Prob, atlas, sym=True, labels=None, subset=None, grid=None):
    if isinstance(labels, list):
        labels = np.array(labels)
    K, P = Prob.shape
    if not sym:
        raise (NameError('only for symmetric models right now'))
    else:
        K = int(K / 2)
        PL = Prob[:K, :]
        PR = Prob[K:, :]
        Prob = PL + PR
        Prob[Prob > 1] = 1  # Exclude problems in the vermis
    if subset is None:
        subset = np.arange(K)
    if grid is None:
        a = int(np.ceil(np.sqrt(len(subset))))
        grid = (a, a)
    plot_multi_flat(Prob[subset, :], atlas, grid,
                    dtype='func',
                    cscale=[0, 0.2],
                    titles=labels[subset],
                    colorbar=False,
                    save_fig=False)


def plot_model_parcel(model_names, grid, cmap='tab20b', align=False, device=None):
    """  Load a bunch of model fits, selects the best from
    each of them and plots the flatmap of the parcellation

    Args:
        model_names (list): List of mode names
        grid (tuple): (rows,cols) of matrix
        cmap (str / colormat): Colormap. Defaults to 'tab20b'.
        align (bool): Align the models before plotting. Defaults to False.
    """
    titles = []
    models = []

    # Load models and produce titles
    for i, mn in enumerate(model_names):
        info, model = load_batch_best(mn, device=device)
        models.append(model)
        # Split the name and build titles
        fname = mn.split('/')  # Get filename if directory is given
        split_mn = fname[-1].split('_')
        atlas = split_mn[2][6:]
        titles.append(split_mn[1] + ' ' + split_mn[3])

    # Align models if requested
    if align:
        Prob = ev.align_models(models, in_place=False)
    else:
        Prob = ev.extract_marginal_prob(models)

    if type(Prob) is pt.Tensor:
        if pt.cuda.is_available() or pt.backends.mps.is_built():
            Prob = Prob.cpu().numpy()
        else:
            Prob = Prob.numpy()

    parc = np.argmax(Prob, axis=1) + 1

    plot_multi_flat(Prob, atlas, grid=grid,
                    cmap=cmap, dtype='prob',
                    titles=titles)

def plot_corr_wb(corr, idx, type='group', type_2='wb', max_D=35,
                 binwidth=1, title=[]):
    x = np.arange(0, max_D, binwidth) + binwidth/2
    num_row = len(corr)
    naming = {"wb": ['within', 'between', 'Correlation'],
              "nums": ['numW', 'numB', 'number of voxel pair'],
              "weight": 'weight'}
    # find a maximum row number variable
    max_col = 0
    for row in corr:
        # get the length of the keys in the row's sub-dictionary
        num_col = len(corr[row])
        # update the maximum column number if needed
        if num_col > max_col:
            max_col = num_col

    fig, axes = plt.subplots(nrows=num_row, ncols=max_col, squeeze=False,
                             figsize=(5*max_col, 5*num_row), sharey='row')

    for i, train_smooth in enumerate(corr.keys()):
        for j, (test_smooth, value) in enumerate(corr[train_smooth].items()):
            if type_2 == 'wb' or type_2 == 'nums':
                cw = value[idx][f'{type}_{naming[type_2][0]}']
                cb = value[idx][f'{type}_{naming[type_2][1]}']
                # Calculate the mean and standard deviation across subjects for each timestamp
                se_w = np.nanstd(cw, axis=0) / np.sqrt(cw.shape[0])
                se_b = np.nanstd(cb, axis=0) / np.sqrt(cb.shape[0])

                # Plot the mean data as a line and show the standard deviation with error bars
                axes[i,j].errorbar(x, np.nanmean(cw, axis=0), yerr=se_w,
                                   fmt='-', c='k', capsize=1, capthick=0.5,
                                   elinewidth=0.8, label=f'{naming[type_2][0]}')
                axes[i,j].errorbar(x, np.nanmean(cb, axis=0), yerr=se_b,
                                   fmt='-', c='r', capsize=1, capthick=0.5,
                                   elinewidth=0.8, label=f'{naming[type_2][1]}')

                axes[i,j].legend()
                axes[i,j].set_xlabel('Spatial distance (mm)')
                axes[i,j].set_ylabel(f'{naming[type_2][2]} - {train_smooth}')
                axes[i,j].set_title(test_smooth)
            elif type_2 == 'weight':
                c = value[idx][f'{type}_weight']
                se = np.nanstd(c, axis=0) / np.sqrt(c.shape[0])
                axes[i,j].errorbar(x, np.nanmean(c, axis=0), yerr=se,
                                   fmt='-', c='b', capsize=1, capthick=0.5,
                                   elinewidth=0.8, label='weighting')
                axes[i, j].set_xlabel('Spatial distance (mm)')
                axes[i, j].set_ylabel(f'weighting - {train_smooth}')
                axes[i, j].set_title(test_smooth)
            else:
                raise ValueError('Unrecognized type 2.')

    plt.suptitle(f'DCBC related curves - type:{type} of {title[idx]}')
    plt.tight_layout()
    plt.show()

def compute_var_cov(data, cond='all', mean_centering=True):
    """
        Compute the affinity matrix by given kernel type,
        default to calculate Pearson's correlation between all vertex pairs

        :param data: subject's connectivity profile, shape [N * k]
                     N - the size of vertices (voxel)
                     k - the size of activation conditions
        :param cond: specify the subset of activation conditions to evaluation
                    (e.g condition column [1,2,3,4]),
                     if not given, default to use all conditions
        :param mean_centering: boolean value to determine whether the given subject data
                               should be mean centered

        :return: cov - the covariance matrix of current subject data. shape [N * N]
                 var - the variance matrix of current subject data. shape [N * N]
    """
    if mean_centering:
        data = data - pt.mean(data, dim=1, keepdim=True)  # mean centering

    # specify the condition index used to compute correlation, otherwise use all conditions
    if cond != 'all':
        data = data[:, cond]
    elif cond == 'all':
        data = data
    else:
        raise TypeError("Invalid condition type input! cond must be either 'all'"
                        " or the column indices of expected task conditions")

    k = data.shape[1]
    cov = pt.matmul(data, data.T) / (k - 1)
    # sd = data.std(dim=1).reshape(-1, 1)  # standard deviation
    sd = pt.sqrt(pt.sum(data ** 2, dim=1, keepdim=True) / (k - 1))
    var = pt.matmul(sd, sd.T)

    return cov, var


def compute_dist(coord, resolution=2):
    """
    calculate the distance matrix between each of the voxel pairs by given mask file

    :param coord: the ndarray of all N voxels coordinates x,y,z. Shape N * 3
    :param resolution: the resolution of .nii file. Default 2*2*2 mm

    :return: a distance matrix of N * N, where N represents the number of masked voxels
    """
    if type(coord) is np.ndarray:
        coord = pt.tensor(coord, dtype=pt.get_default_dtype())

    num_points = coord.shape[0]
    D = pt.zeros((num_points, num_points))
    for i in range(3):
        D = D + (coord[:, i].reshape(-1, 1) - coord[:, i]) ** 2
    return pt.sqrt(D) * resolution


def compute_DCBC(maxDist=35, binWidth=1, parcellation=np.empty([]),
                 func=None, dist=None, weighting=True):
    """
    The main entry of DCBC calculation for volume space
    :param hems:        Hemisphere to test. 'L' - left hemisphere; 'R' - right hemisphere; 'all' - both hemispheres
    :param maxDist:     The maximum distance for vertices pairs
    :param binWidth:    The spatial binning width in mm, default 1 mm
    :param parcellation:
    :param dist_file:   The path of distance metric of vertices pairs, for example Dijkstra's distance, GOD distance
                        Euclidean distance. Dijkstra's distance as default
    :param weighting:   Boolean value. True - add weighting scheme to DCBC (default)
                                       False - no weighting scheme to DCBC
    """
    numBins = int(np.floor(maxDist / binWidth))
    cov, var = compute_var_cov(func)
    # cor = np.corrcoef(func)
    if not dist.is_sparse:
        dist = dist.to_sparse()

    row = dist._indices()[0]
    col = dist._indices()[1]
    distance = dist._values()
    # row, col, distance = sp.sparse.find(dist)

    # making parcellation matrix without medial wall and nan value
    par = parcellation
    num_within, num_between, corr_within, corr_between = [], [], [], []
    for i in range(numBins):
        inBin = pt.where((distance > i * binWidth) &
                         (distance <= (i + 1) * binWidth))[0]

        # lookup the row/col index of within and between vertices
        within = pt.where((par[row[inBin]] == par[col[inBin]]) == True)[0]
        between = pt.where((par[row[inBin]] == par[col[inBin]]) == False)[0]

        # retrieve and append the number of vertices for within/between in current bin
        num_within.append(
            pt.tensor(within.numel(), dtype=pt.get_default_dtype()))
        num_between.append(
            pt.tensor(between.numel(), dtype=pt.get_default_dtype()))

        # Compute and append averaged within- and between-parcel correlations in current bin
        this_corr_within = pt.nanmean(cov[row[inBin[within]], col[inBin[within]]]) \
            / pt.nanmean(var[row[inBin[within]], col[inBin[within]]])
        this_corr_between = pt.nanmean(cov[row[inBin[between]], col[inBin[between]]]) \
            / pt.nanmean(var[row[inBin[between]], col[inBin[between]]])

        corr_within.append(this_corr_within)
        corr_between.append(this_corr_between)

        del inBin

    if weighting:
        weight = 1 / (1 / pt.stack(num_within) + 1 / pt.stack(num_between))
        weight = weight / pt.sum(weight)
        DCBC = pt.nansum(pt.multiply(
            (pt.stack(corr_within) - pt.stack(corr_between)), weight))
    else:
        DCBC = pt.nansum(pt.stack(corr_within) - pt.stack(corr_between))
        weight = pt.nan

    D = {
        "binWidth": binWidth,
        "maxDist": maxDist,
        "num_within": num_within,
        "num_between": num_between,
        "corr_within": corr_within,
        "corr_between": corr_between,
        "weight": weight,
        "DCBC": DCBC
    }

    return D

def compute_rawCorr(maxDist=35, binWidth=1, func=None, dist=None):
    """
    The main entry of DCBC calculation for volume space
    :param hems:        Hemisphere to test. 'L' - left hemisphere;
                        'R' - right hemisphere; 'all' - both hemispheres
    :param maxDist:     The maximum distance for vertices pairs
    :param binWidth:    The spatial binning width in mm, default 1 mm
    :param parcellation:
    :param dist_file:   The path of distance metric of vertices pairs,
                        for example Dijkstra's distance, GOD distance
                        Euclidean distance. Dijkstra's distance as default
    """
    if type(func) is np.ndarray:
        func = pt.tensor(func, dtype=pt.get_default_dtype())

    numBins = int(np.floor(maxDist / binWidth))
    cov, var = compute_var_cov(func)

    # remove the nan value and medial wall from dist file
    dist = dist.to_sparse()
    row = dist.indices()[0]
    col = dist.indices()[1]
    distance = dist.values()
    # row, col, distance = sp.sparse.find(dist)

    # making parcellation matrix without medial wall and nan value
    num, corr,= [], []
    for i in range(numBins):
        inBin = pt.where((distance > i * binWidth) &
                         (distance <= (i + 1) * binWidth))[0]

        # Retrieve and append the number of vertices in current bin
        num.append(pt.tensor(inBin.numel(), dtype=pt.get_default_dtype()))

        # Compute and append averaged correlation in current bin
        this_corr = pt.nanmean(cov[row[inBin], col[inBin]]) \
                    / pt.nanmean(var[row[inBin], col[inBin]])
        corr.append(this_corr)

        del inBin

    return pt.stack(num), pt.stack(corr)

def similarity_between_datasets(base_dir, dataset_name, atlas='MNISymC3',
                                subtract_mean=True, voxel_wise=True):
    """Calculates the average within subject reliability
    maps across sessions for a single data

    Args:
        base_dir (str / path): Base directory
        dataset_name (str): Name of data set
        atlas (str): _description_. Defaults to 'MNISymC3'.
        subtract_mean (bool): Remove the mean per voxel before correlation calc?

    Returns:
        _type_: _description_
    """
    n_datasets = len(dataset_name)
    # n_vox = data.shape[2]
    # Rel = np.zeros((n_datasets, n_vox))
    Rel = []
    for i, ds in enumerate(dataset_name):
        data, _, _ = dt.get_dataset(base_dir, ds, atlas=atlas)
        # r = reliability_within_subj(data[:, indx, :],
        #                             part_vec=info[dataset.part_ind][indx],
        #                             cond_vec=info[dataset.cond_ind][indx],
        #                             voxel_wise=voxel_wise,
        #                             subtract_mean=subtract_mean)
        r = dt.reliability_between_subj(data, cond_vec=None,
                                        voxel_wise=voxel_wise,
                                        subtract_mean=subtract_mean)
        Rel.append(np.nanmean(r, axis=0))
    return np.stack(Rel)

# def make_label_cifti(data,
#                      anatomical_struct='Cerebellum',
#                      labels=None,
#                      label_names=None,
#                      column_names=None,
#                      label_RGBA=None):
#     """Generates a label Cifti2Image from a numpy array
#
#     Args:
#         data (np.array):
#              num_vert x num_col data
#         anatomical_struct (string):
#             Anatomical Structure for the Meta-data default= 'Cerebellum'
#         labels (list): Numerical values in data indicating the labels -
#             defaults to np.unique(data)
#         label_names (list):
#             List of strings for names for labels
#         column_names (list):
#             List of strings for names for columns
#         label_RGBA (list):
#             List of rgba vectors for labels
#     Returns:
#         gifti (GiftiImage): Label gifti image
#
#     """
#     num_verts, num_cols = data.shape
#     if labels is None:
#         labels = np.unique(data)
#     num_labels = len(labels)
#
#     # Create naming and coloring if not specified in varargin
#     # Make columnNames if empty
#     if column_names is None:
#         column_names = []
#         for i in range(num_cols):
#             column_names.append("col_{:02d}".format(i+1))
#
#     # Determine color scale if empty
#     if label_RGBA is None:
#         label_RGBA = np.zeros([num_labels,4])
#         hsv = plt.cm.get_cmap('hsv',num_labels)
#         color = hsv(np.linspace(0,1,num_labels))
#         # Shuffle the order so that colors are more visible
#         color = color[np.random.permutation(num_labels)]
#         for i in range(num_labels):
#             label_RGBA[i] = color[i]
#
#     # Create label names from numerical values
#     if label_names is None:
#         label_names = []
#         for i in labels:
#             label_names.append("label-{:02d}".format(i))
#
#     # Create label.gii structure
#     C = nb.gifti.GiftiMetaData.from_dict({
#         'AnatomicalStructurePrimary': anatomical_struct,
#         'encoding': 'XML_BASE64_GZIP'})
#
#     E_all = []
#     for (label, rgba, name) in zip(labels, label_RGBA, label_names):
#         E = nb.gifti.gifti.GiftiLabel()
#         E.key = label
#         E.label= name
#         E.red = rgba[0]
#         E.green = rgba[1]
#         E.blue = rgba[2]
#         E.alpha = rgba[3]
#         E.rgba = rgba[:]
#         E_all.append(E)
#
#     D = list()
#     for i in range(num_cols):
#         d = nb.gifti.GiftiDataArray(
#             data=np.float32(data[:, i]),
#             intent='NIFTI_INTENT_LABEL',
#             datatype='NIFTI_TYPE_UINT8',
#             meta=nb.gifti.GiftiMetaData.from_dict({'Name': column_names[i]})
#         )
#         D.append(d)
#
#     # Make and return the gifti file
#     gifti = nb.gifti.GiftiImage(meta=C, darrays=D)
#     gifti.labeltable.labels.extend(E_all)
#     return gifti
