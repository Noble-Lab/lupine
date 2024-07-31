"""
UTILS

This module contains commonly used utility and plotting functions. 
Right now I'm just setting output png paths ad hoc in 
`plt_obs_vs_imputed` and `plot_loss_curves_utils`. At some point it 
would be nice to robustify these plotting functions a bit.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('../../bin/')
from data_loaders import FactorizationDataset
from scalers import STDScaler

# plotting templates
sns.set(context="talk", style="ticks") 
pal = sns.color_palette()

def plt_obs_vs_imputed(
        recon_mat, 
        orig_mat, 
        log_scale=False, 
        name=None,
):
    """ 
    Generate an observed vs imputed peptide abundance plot for a 
    given model. Can be for the training or validation set. 

    Parameters
    ----------
    recon_mat : np.ndarray, 
        The reconstructed matrix
    orig_mat : np.ndarray, 
        The original matrix. Can be either the validation or the
        training matrix
    log_scale : bool, 
        Log scale the plot axes? 
    name : str, 
        The full path to write the output file to

    Returns
    -------
    None
    """
    # get index of nan values in original set
    orig_nans = np.isnan(orig_mat)

    # get non-nan values in both matrices, for the {valid, train} 
    # set only
    orig_set = orig_mat[~orig_nans]
    recon_set = recon_mat[~orig_nans]

    # get Pearson's correlation coefficient
    corr_mat = np.corrcoef(x=orig_set, y=recon_set)
    pearson_r = np.around(corr_mat[0][1], 2)

    # initialize the figure
    plt.figure(figsize=(5,5))
    ax = sns.scatterplot(x=orig_set, y=recon_set, alpha=0.5)

    ax.set_xlabel('Observed Abundance')
    ax.set_ylabel('Imputed Abundance')

    if log_scale:
        plt.xscale("log")
        plt.yscale("log")

    # add the correlation coefficient
    ax.text(0.95, 0.05, "R: %s"%(pearson_r),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=20)

    set_min = min(np.min(orig_set), np.min(recon_set))
    set_max = max(np.max(orig_set), np.max(recon_set))

    if set_min < 1:
        set_min = 1

    ax.set_xlim(left=set_min, right=set_max)
    ax.set_ylim(bottom=set_min, top=set_max)

    # add diagonal line
    x = np.linspace(set_min, set_max, 100)
    y = x
    plt.plot(x, y, '-r', label='y=x', alpha=0.6)

    plt.minorticks_off()

    if name is not None:
        plt.savefig(
            name + ".png", 
            dpi=250, 
            bbox_inches="tight",
        )
    else:
        plt.show()

    plt.close()
    return

def mcar_partition(
			matrix, 
			val_frac=0.1, 
			test_frac=0.1, 
			min_present=5, 
			random_state=42
):
    """
    Use MCAR procedure to split a data matrix into training, 
    validation, test sets. Note that the fractions of data in the 
    validation and tests sets is only approximate due to the need 
    to drop rows with too much missing data.

    Also returning the discard boolean array so that we can 
    keep track of which proteins were removed.
    
    Parameters
    ----------
    matrix : array-like,
        The data matrix to split.
    val_frac : float, optional
        The fraction of data to assign to the validation set.
    test_frac : float, optional
        The fraction of data to assign to the test set.
    min_present : int, optional
        The minimum number of non-missing values required in each 
        row of the training set.
    random_state : int or numpy.random.Generator
        The random state for reproducibility.
    
    Returns
    -------
    train_set, val_set, test_set : numpy.ndarray,
        The training set, validation and test sets. In the case
        of validation and test, the non-valid/test set values
        are NaNs
    discard : np.array, 
        A boolean array specifying whether or not each protein
        passess the `min_val` threshold. 
    """
    rng = np.random.default_rng(random_state)
    if val_frac + test_frac > 1:
        raise ValueError(
        	"'val_frac' and 'test_frac' cannot sum to more than 1.")

    # Assign splits:
    indices = np.vstack(np.nonzero(~np.isnan(matrix)))
    rng.shuffle(indices, axis=1)

    n_val = int(indices.shape[1] * val_frac)
    n_test = int(indices.shape[1] * test_frac)
    n_train = indices.shape[1] - n_val - n_test

    train_idx = tuple(indices[:, :n_train])
    val_idx = tuple(indices[:, n_train:(n_train + n_val)])
    test_idx = tuple(indices[:, -n_test:])

    train_set = np.full(matrix.shape, np.nan)
    val_set = np.full(matrix.shape, np.nan)
    test_set = np.full(matrix.shape, np.nan)

    train_set[train_idx] = matrix[train_idx]
    val_set[val_idx] = matrix[val_idx]
    test_set[test_idx] = matrix[test_idx]

    # Remove proteins with too many missing values:
    num_present = np.sum(~np.isnan(train_set), axis=1)
    discard = num_present < min_present
    num_discard = discard.sum()

    train_set = np.delete(train_set, discard, axis=0)
    val_set = np.delete(val_set, discard, axis=0)
    test_set = np.delete(test_set, discard, axis=0)

    return train_set, val_set, test_set, discard

def mnar_partition_thresholds_matrix(
		mat, 
		q_anchor=0.2, 
		t_std=0.1, 
		brnl_prob=0.5, 
		min_pres=4,
		rand_state=None,
):
    """
    For a given peptide/protein quants matrix, constructs an 
    equally sized thresholds matrix that is filled with Gaussian 
    selected values, anchored at a given percentile of the 
    peptide/protein quants distribution, with a given standard 
    deviation. For each peptide quants matrix element X_ij, if 
    the corresponding thresholds matrix element T_ij is less, 
    pass. Else, conduct a single Bernoulli trial with specified 
    success probability. If success, X_ij is selected for the mask. 
    Else, pass. 

    Also returning the discard boolean array so that we can 
    keep track of which proteins were removed.

    Parameters
    ----------
    mat : np.ndarray, 
        The unpartitioned peptide/protein quants matrix
    q_anchor : float, 
        The percentile of the abundance values on which to 
        anchor the thresholds matrix
    t_std : float, 
        How many standard deviations of the quants matrix to use 
        when constructing the thresholds matrix? 
    brnl_prob : float, 
        The probability of success for the Bernoulli draw
    min_pres : int, 
        The minimum number of present values for each row in the
        training and validation sets. 
    rand_state : int, 
        The integer for seeding numpy's random number generator.
        The default is None, which will just choose a random seed,
        I think

    Returns
    -------
    train_mat, val_mat, test_mat : np.ndarray, 
        The training, validation & test matrices, respectively
    discard : np.array, 
        A boolean array specifying whether or not each protein
        passess the `min_val` threshold. 
    """
    rng = np.random.default_rng(seed=rand_state)

    # get the specified quantile from the original matrix
    q_thresh = np.nanquantile(mat, q_anchor)
    # get the standard deviation from the original matrix
    quants_std = np.nanstd(mat)

    thresh_mat = rng.normal(
                       loc=q_thresh, 
                       scale=(quants_std * t_std), 
                       size=mat.shape,
    )
    # no longer strictly Gaussian
    thresh_mat = np.abs(thresh_mat)

    # define the training mask
    zeros = np.zeros(shape=mat.shape)
    mask = zeros > 1

    # loop through every entry in the matrix
    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[1]):
            quants_mat_curr = mat[i,j]
            thresh_mat_curr = thresh_mat[i,j]

            if thresh_mat_curr > quants_mat_curr: 
                # for n=1 trials, I believe binomial == bernoulli
                success = rng.binomial(1, brnl_prob)
            else: 
                success = 0

            mask[i,j] = success

    # get the indices corresponding to `True` in the mask
    indices = np.vstack(np.nonzero(mask))
    # shuffle the indices
    rng.shuffle(indices, axis=1)
    # get n=half the number of Trues in the mask
    n_idx = np.int32(np.floor(indices.shape[1] / 2))
    # divide the indices into two disjoint sets
    val_idx = tuple(indices[:,:n_idx])
    test_idx = tuple(indices[:,n_idx:])
    # init val and test sets
    val_mat = np.full(mat.shape, np.nan)
    test_mat = np.full(mat.shape, np.nan)
    # assign values to val and test sets
    val_mat[val_idx] = mat[val_idx]
    test_mat[test_idx] = mat[test_idx]    
    # define the training set
    train_mat = mat.copy()
    train_mat[mask] = np.nan

    # remove peptides with fewer than min_present present values
    num_present = np.sum(~np.isnan(train_mat), axis=1)
    discard = num_present < min_pres

    train_mat = np.delete(train_mat, discard, axis=0)
    val_mat = np.delete(val_mat, discard, axis=0)
    test_mat = np.delete(test_mat, discard, axis=0)

    return train_mat, val_mat, test_mat, discard

def mnar_partition_single_split(
        mat, 
        q_anchor=0.2, 
        t_std=0.1, 
        brnl_prob=0.5, 
        min_pres=4,
        rand_state=None,
):
    """
    Same as the previous function, except this one returns 
    only two disjoint sets instead of three. This one is far 
    simplier. 

    Parameters
    ----------
    mat : np.ndarray, 
        The unpartitioned peptide/protein quants matrix
    q_anchor : float, 
        The percentile of the abundance values on which to 
        anchor the thresholds matrix
    t_std : float, 
        How many standard deviations of the quants matrix to use 
        when constructing the thresholds matrix? 
    brnl_prob : float, 
        The probability of success for the Bernoulli draw
    min_pres : int, 
        The minimum number of present values for each row in the
        training and validation sets. 
    rand_state : int, 
        The integer for seeding numpy's random number generator.
        The default is None, which will just choose a random seed,
        I think

    Returns
    -------
    train_mat, val_mat np.ndarray, 
        The training, validation matrices, respectively. 
    """
    rng = np.random.default_rng(seed=rand_state)

    # get the specified quantile from the original matrix
    q_thresh = np.nanquantile(mat, q_anchor)
    # get the standard deviation from the original matrix
    quants_std = np.nanstd(mat)

    thresh_mat = rng.normal(
                       loc=q_thresh, 
                       scale=(quants_std * t_std), 
                       size=mat.shape,
    )
    # no longer strictly Gaussian
    thresh_mat = np.abs(thresh_mat)

    # define the training mask
    zeros = np.zeros(shape=mat.shape)
    mask = zeros > 1

    # loop through every entry in the matrix
    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[1]):
            quants_mat_curr = mat[i,j]
            thresh_mat_curr = thresh_mat[i,j]

            if thresh_mat_curr > quants_mat_curr: 
                # for n=1 trials, I believe binomial == bernoulli
                success = rng.binomial(1, brnl_prob)
            else: 
                success = 0

            mask[i,j] = success

    # define the training and validation sets
    train_mat = mat.copy()
    val_mat = mat.copy()
    train_mat[mask] = np.nan
    val_mat[~mask] = np.nan

    # remove peptides with fewer than min_present present values
    num_present = np.sum(~np.isnan(train_mat), axis=1)
    discard = num_present < min_pres

    train_mat = np.delete(train_mat, discard, axis=0)
    val_mat = np.delete(val_mat, discard, axis=0)

    return train_mat, val_mat

def mse_func(x_mat, y_mat):
    """
    Calculate the MSE for two matricies with missing values. Each
    matrix can contain MVs, in the form of np.nans
    
    Parameters
    ----------
    x_mat : np.ndarray, 
        The first matrix 
    y_mat : np.ndarray, T
        he second matrix
    
    Returns
    -------
    float, the mean squared error between values present 
            across both matrices
    """
    x_rav = x_mat.ravel()
    y_rav = y_mat.ravel()
    missing = np.isnan(x_rav) | np.isnan(y_rav)
    mse = np.sum((x_rav[~missing] - y_rav[~missing])**2)

    return mse / np.sum(~missing)

def plot_partition_distributions(
        train_mat, 
        train_loader, 
        val_loader, 
        outstr=None, 
        save_fig=False,
):
    """
    Want to see what the partitions look like, exactly as the MODEL
    sees them. This function plot the distributions of the training
    set and the train and validation sets after mini-batch selection.

    Parameters
    ----------
    train_mat : np.ndarray, 
        The full training set matrix. Logged
    train_loader : np.ndarray, 
        The batched training matrix. Logged
    val_loader : np.ndarray, 
        The batched validation matrix. Logged
    outstr : str, optional
        What to call the output file (.png)?
    save_fig : bool, optional
        Write the figure to a png? 

    Returns
    ----------
    none
    """
    # get all of the X_ijs in both loaders
    train_targets = np.array([])
    for locs, target in train_loader:
        locs = locs.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        train_targets = np.append(train_targets, target)

    valid_targets = np.array([])
    for locs, target in val_loader:
        locs = locs.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        valid_targets = np.append(valid_targets, target)

    # PLOT THE DISTRIBUTIONS
    # flatten
    train_mat = np.array(train_mat)
    train_rav = train_mat.ravel()

    # get present values from the training matrix
    train_nans = np.isnan(train_rav)
    train_present = train_rav[~train_nans]

    # get the means
    train_mean = np.around(np.mean(train_present), 2)
    train_mb_mean = np.around(np.mean(train_targets), 2)
    val_mb_mean = np.around(np.mean(valid_targets), 2)

    # get the medians
    train_med = np.around(np.median(train_present), 2)
    train_mb_med = np.around(np.median(train_targets), 2)
    val_mb_med = np.around(np.median(valid_targets), 2)

    # plot
    plt.figure()
    plt.hist(
        train_present, 
        density=False, 
        bins=60, 
        linewidth=0.01, 
        color='firebrick', 
        edgecolor='firebrick', 
        alpha=1.0, 
        label="Train (med: " + str(train_med) + ")",
    )
    plt.hist(
        train_targets, 
        density=False, 
        bins=60, 
        linewidth=0.01, 
        color='olivedrab', 
        edgecolor='olivedrab', 
        alpha=0.8, 
        label="Train Batched (med: " + str(train_mb_med) + ")",
    )
    plt.hist(
        valid_targets, 
        density=False, 
        bins=60, 
        linewidth=0.01, 
        color='gold', 
        edgecolor='gold', 
        alpha=0.7, 
        label="Validation (med: " + str(val_mb_med) + ")",
    )
    plt.minorticks_off()

    plt.xlabel("Intensity", labelpad=12)
    plt.ylabel("Counts", labelpad=12)

    leg = plt.legend(bbox_to_anchor=(1.9, 1.05))
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    #plt.ticklabel_format(style="sci", axis="x", scilimits=(0,0))

    if save_fig:
        plt.savefig(outstr + ".png", dpi=250, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
    return

def plot_loss_curves_utils(model, name=None):
    """ 
    Generate model loss vs training epoch plot. For both training
    and validation sets. A basic sanity check method. Note that 
    the scale of the y axis will reflect the scaled values. 

    Parameters
    ----------
    model : {NNImputer, NMFImputer, TransformerFactorizationImputer,
             TransformerFactorizationNNImputer}
        The imputation model
    name : str, 
        The full path to write the output file to
    """
    plt.figure()
    plt.plot(list(model.history.epoch[1:]),  # model.history.epoch[1:]
        list(model.history["Train"][1:]), 
        label="Training loss")
    plt.plot(list(model.history.epoch[1:]), 
        list(model.history["Validation"][1:]), 
        label="Validation loss")

    plt.ylim(ymin=0)

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    if name is not None:
        plt.savefig(
            name + ".png", 
            dpi=250, 
            bbox_inches="tight",
        )
    else:
        plt.show()
    plt.close()

    return

def plot_loss_curves_no_val(model, name=None):
    """ 
    Generate model loss vs training epoch plot. For both training
    and validation sets. A basic sanity check method. Note that 
    the scale of the y axis will reflect the scaled values. For a
    Lupine model that has not been trained with a validation set; 
    will plot the training loss only. 

    Parameters
    ----------
    model : Lupine
        The imputation model
    name : str, 
        The full path to write the output file to
    """
    plt.figure()
    plt.plot(list(model.history.index[1:]),
        list(model.history["Train"][1:]), 
        label="Training loss")

    plt.ylim(ymin=0)

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    if name is not None:
        plt.savefig(
            name + ".png", 
            dpi=250, 
            bbox_inches="tight",
        )
    else:
        plt.show()
    plt.close()

    return

def nrmse_loss(x_vals, y_vals):
    """
    Get the (normalized) Root Mean Squared Error loss between 
    two tensors. One question is how to actually do the 
    normalization: could be standard deviation, could be mean, 
    could be (max-min).

    Parameters
    ----------
    x_vals, y_vals : np.ndarray,
        The observed and expected values, respectively

    Returns
    ----------
    nrmse_loss : float,
        The normalized root mean squared error loss
    """
    # Exclude NaNs
    x_rav = x_vals.ravel()
    y_rav = y_vals.ravel()
    missing = np.isnan(x_rav) | np.isnan(y_rav)

    # Get the MSE
    mse = np.sum((x_rav[~missing] - y_rav[~missing])**2) \
                / np.sum(~missing)
    # Get the RMSE
    rmse = np.sqrt(mse)

    # Normalize by the standard deviation of the expected values
    #   How to do the normalization?
    y_std = np.std(y_rav[~missing])
    y_mean = np.mean(y_rav[~missing])
    y_diff = \
        np.max(y_rav[~missing]) - np.min(y_rav[~missing])
    nrmse = rmse / y_std

    return nrmse

def plot_observed_vs_expected(
    observed_set, expected_set, 
    set_name, 
    file_name, 
):
    """
    Makes an observed vs expected scatter plot for a Lupine model. 
    Optionally saves to a png in the results directory. 

    Parameters
    ----------
    observed_set, expected_set : np.ndarray, 
        The observed and expected matrices, respectively.
        Can be train, validation or test sets. 
    set_name : str, 
        The name of the set. {"Validation", "Test"}
    file_name : str, 
        The name for the output file

    Returns
    ----------
    none
    """
    obs_rav = observed_set.ravel()
    exp_rav = expected_set.ravel()

    missing = np.isnan(obs_rav) | np.isnan(exp_rav)
    obs_rav = obs_rav[~missing]
    exp_rav = exp_rav[~missing]

    exp_min = np.nanmin(exp_rav)
    exp_max = np.nanmax(exp_rav)

    # get Pearson's correlation coefficient
    corr_mat = np.corrcoef(x=obs_rav, y=exp_rav)
    pearson_r = np.around(corr_mat[0][1], 2)

    plt.figure(figsize=(5,5))
    ax = sns.scatterplot(x=exp_rav, y=obs_rav, alpha=0.05)

    ax.set_xlabel("Observed Quant")
    ax.set_ylabel("Imputed Quant")

    ax.set_xlim([exp_min, exp_max])
    ax.set_ylim([exp_min, exp_max])

    # Add diagonal line
    x = np.linspace(exp_min, exp_max, 30)
    y = x
    plt.plot(
        x, y, label="y=x", alpha=0.9, linewidth=3, color="#ff7f0e")
    #plt.title("MNAR", pad=12, fontsize=24)

    # Add the correlation coefficient
    ax.text(
        0.95, 0.05, 
        "R: %s"%(pearson_r),
        verticalalignment="bottom", 
        horizontalalignment="right",
        transform=ax.transAxes,
        color="#2ca02c", 
        fontsize=20,
        alpha=1.0
    )
    plt.minorticks_off()
    plt.show()

    if file_name is not None:
        plt.savefig(
            file_name + ".png", 
            dpi=250, 
            bbox_inches="tight",
        )
    else:
        plt.show()
    plt.close()
    return
