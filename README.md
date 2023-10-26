FusionModel
====
Diedrichsen Lab, Western University

This repository hosts the scripts to replicate the results reported in the 
[paper](https://www.biorxiv.org/content/10.1101/2023.05.24.542121v1). It uses the functions from 
two repositories, [HierarchBayesParcel](https://github.com/DiedrichsenLab/HierarchBayesParcel)
and [Functional_Fusion](https://github.com/DiedrichsenLab/Functional_Fusion). The former is the 
computational model and the latter is the data preprocessing pipeline. The general workflow is
to first preprocess the data using our pipeline to bring different fMRI datasets into a standard 
template, then the preprocessed data will be sent as input to the computational model to learn 
the probabilistic brain parcellations across datasets. Please find below paper for more details.

Reference
------
* Zhi, D., Shahshahani, L., Nettekovena, C., Pinho, A. L. Bzdok, D., Diedrichsen, J., (2023). 
"A hierarchical Bayesian brain parcellation framework for fusion of functional imaging datasets". 
BioRxiv. [[link]](https://www.biorxiv.org/content/10.1101/2023.05.24.542121v1)

Dependencies
------
### Packages
This project depends on several third party libraries, including: [numpy](https://numpy.org/) 
(version>=1.22.2), [PyTorch](https://pytorch.org/) (version>=1.10.1 + CUDA enabled), 
[nilearn](https://nilearn.github.io/stable/index.html) (version>=0.9.0),
[nibabel](https://nipy.org/nibabel/) (version>=3.2.0), etc.

Please find the `requirements.txt ` for more details and their version.

### Installations
```
pip install -r requirements.txt 
```

Or you can install the package manually from their original binary source as above links.

Once you clone the functional fusion repository, you need to add it to your PYTHONPATH, so you can
import the functionality. Add these lines to your .bash_profile, .bash_rc .zsh_profile file... 

```
PYTHONPATH=<your_repo_absolute_path>:${PYTHONPATH}
export PYTHONPATH
```

Results replication
------
### Usage
In `set_globals.py`, user can set the global variables for the project, including the data path,
the output path, and CUDA GPU device settings. We highly recommend user to use GPU to run the 
learning algorithm as the CPU version can be extremely slow, considering the large number of brain
voxels.

We also offer the separate scripts to replicate the results in the paper with its own data path 
and computational settings. Please find below for more details.

**Note**: By implementing our scripts does not guarantee the user can replicate the results, 
further debug may be needed since the trained model is not openly available yet, and the path of 
the data host may change user by user. Please contact Diedrichsen Lab to request the trained 
model, parcellations and the resultant data or figures.

### 1. Individual parcellations in the scarce data setting
`scripts/modeling_1.py` is the script to replicate the result 1 in the paper. It first find the 
pre-trained parcellation models using MDTB dataset session 1 or session 2, then cross-validate 
using the other session. The results will be saved in the output path using pandas dataframe. 
function `result_1_plot_curve` is used to plot the individual parcellation DCBC values of data only 
vs. the ones with the integrated prior. `result_1_plot_flatmap` is used to plot the individual 
parcellations given the optimally calculated color map.

### 2 and 3. Simulation on synthetic datasets
**Simulation 1: Dataset-specific emission models optimally capture differences in measurement 
noise**

**Simulation 2: Region-specific concentration parameters further improve fusion parcellation**

`scripts/modeling_2.py` is the script to replicate the synthetic datasets simulation in the 
result 2 and 3. It first comparing the model type 1 and 2 on 100 simulations, then comparing the
type 2 and 3 on another 100 simulations with different SNR for different functional regions. 
Lastly, we compare the model type 2 and 3 with increasing number of K for fitting, along with 
the type 1,2,3 comparison in the supplementary material 3.

### 4. Model performance on real data and the choice of atlas resolution K
`scripts/modeling_4.py` is the script to replicate the result 4 in the paper - IBC dataset
individual session vs. all sessions fusion. As well as, the performance comparison 
of trained individual/group parcellations using single session or fusion of the two sessions on 
different atlas resolution K.

### 5. The fusion atlas shows combined strengths across different task-based fMRI datasets
`scripts/modeling_5.py` is the script to replicate the result 5. It has the DCBC evaluation 
routine of the parcellations trained on single dataset or on datasets fusion. It also has the 
plot function for the result curves and the flatmap of the parcellations. This script also host 
the GMM vs. VMF comparison shown in the supplementary material figure 1.

### 6. Integrating resting-state data into the task-based parcellation
`scripts/modeling_6.py` is the script to replicate the result 6. Lastly, we ran the performance
comparison between pure task-based parcellation vs. pure resting-state parcellation vs. the 
fusion of the two types. The script consists of everything in this task from model fitting, 
evaluating, result plotting to the final flatmap visualization.


Parcellation model training
------
`learn_fusion_gpu.py` contains the major functions of how to use the two repositories to train 
the brain parcellation on GPU using PyTorch CUDA enabled tensor. It has the following functions:

* `build_data_list()` - build the data list for the model to read in the data
* `build_model()` - build the model with the given initial parameters, which will be used for 
  further training
* `batch_fit()` - the function to executes a set of fits starting from random starting values
selects the best one from a batch and saves them.
* `fit_all()` - the higher level function to control the model fitting process with the user 
defined settings.

The `main()` function has a minimum training example of how to use the above functions to train 
a model. In this case, the script will train the model to learn individual/group parcellation by 
fusing two functional datasets (one task-based and one resting-state) with the number of parcels
`K` from 10 to 100 using the Type 2 fusion model. Again, this example does not guarantee the 
user can get the trained results and the further debug may be needed. Please contact Diedrichsen 
Lab members if helps needed.

License
------
Please find out our development license (MIT) in `LICENSE` file.

Bug reports
------
Please contact Da Zhi at dzhi@uwo.ca if you have any questions about this repository.