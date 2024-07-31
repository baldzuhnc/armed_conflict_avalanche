import os
os.chdir('/home/clemens/armed_conflict_avalanche')
import warnings

from workspace.utils import load_pickle 
from voronoi_globe.interface import load_voronoi
from arcolanche.pipeline import set_ax

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as stats

import cartopy.feature as cfeature
import cartopy.crs as ccrs

import pyvinecopulib as pv


def get_coarsegrained(conflict_type, scale, binary=True):
    
    # Load data
    dt, dx, gridix = scale
    
    load_pickle(f"avalanches/{conflict_type}/gridix_{gridix}/te/conflict_ev_{str(dt)}_{str(dx)}.p")
    ts_unique = conflict_ev[["t", "x"]]
    
    # Remove duplicate rows/events
    #ts_unique = ts.drop_duplicates()
    
    # Get unique sorted column labels (= cell numbers)
    col_labels = np.sort(np.unique(ts_unique["x"].to_numpy()))
    num_cols = len(col_labels)  # number of columns of the matrix
    
    max_t = ts_unique["t"].max()  # max time (= max length y-axis of the matrix)
    
    # Create a dictionary mapping the unique column labels to indices
    pol_index_mapping_dict = {x: i for i, x in enumerate(col_labels)}
    
    # Create the time series coarse-grained matrix
    CG_matrix = np.zeros((max_t + 1, num_cols), dtype=int)

    # Convert x values to column indices (x axis = columns, t axis = rows)
    col_indices = ts_unique["x"].map(pol_index_mapping_dict).to_numpy()
    row_indices = ts_unique["t"].to_numpy()
    
    if binary:
        # Mark presence
        CG_matrix[row_indices, col_indices] = 1
    else:
        # Count occurrences
        np.add.at(CG_matrix, (row_indices, col_indices), 1)
        
    return pd.DataFrame(CG_matrix, columns=col_labels, dtype=int)


def plot_cells(cell_ids, scale, conflict_type, verbose=False):
    warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect.")
    
    dt, dx, gridix = scale
    
    print(f'dt: {dt}, dx: {dx} gridix: {gridix}')
    
    load_pickle(f"avalanches/{conflict_type}/gridix_{gridix}/te/conflict_ev_{str(dt)}_{str(dx)}.p") 
    polygons = load_voronoi(dx, gridix)
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection=ccrs.PlateCarree())

    #specify polygon and event here
    for i,x in enumerate(cell_ids):
        poly = polygons.loc[[x]]
        poly.plot(ax=ax, color="red")
        
        # Get the centroid of the polygon
        centroid = poly.geometry.centroid.iloc[0]
        # Place the index of the polygon as text on the map
        
        if verbose:
            ax.text(centroid.x, centroid.y, str(x), color='blue', transform=ccrs.Geodetic())

    
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS , linewidth=0.2)

    ax.add_feature(cfeature.LAND)
    #ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.RIVERS)


    ax.set_extent(set_ax("All"))
    
    
    

def fit_and_log_likelihood(dist_name, data):
    """
    Fit a distribution and calculate its log likelihood.
    
    Parameters:
    - dist_name: Name of the distribution as a string.
    - data: Data to fit the distribution to.
    
    Returns:
    - Tuple of (distribution name, log likelihood).
    """
    dist = getattr(stats, dist_name)
    params = dist.fit(data)
    #print(params)
    log_likelihood = np.sum(dist.logpdf(data, *params))
    return (dist_name, log_likelihood)

def fit_distributions(data):
    """
    Fit multiple distributions and sort them by log likelihood in descending order.
    
    Parameters:
    - data: Data to fit the distributions to.
    
    Returns:
    - Sorted list of tuples (distribution name, log likelihood).
    """
    distributions = ['powerlaw', 'cauchy', 'weibull_min', 'pareto', 'genextreme', 'beta']
    results = [fit_and_log_likelihood(dist, data) for dist in distributions]
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    return sorted_results



def empirical_copula(U_xy):
    """
    Calculate the empirical copula of a 2D dataset.
    
    U_xy: 2D numpy array of shape (n, 2)
    """

    n = U_xy.shape[0]
    empirical_copula_values = np.zeros(n)
    
    for i in range(n):
        u_i, v_i = U_xy[i]
        count = np.sum((U_xy[:, 0] <= u_i) & (U_xy[:, 1] <= v_i))
        empirical_copula_values[i] = count / n
    
    return empirical_copula_values


def fit_and_evaluate_copula(data, family):
    """
    Fits a copula model to the given data using the specified family and evaluates its performance.

    Parameters:
    - data: The input data for fitting the copula model.
    - family: The family of copula to be used for fitting the model.

    Returns:
    A dictionary containing the following evaluation metrics:
    - loglik: The log-likelihood of the fitted copula model.
    - aic: The Akaike Information Criterion (AIC) of the fitted copula model.
    - bic: The Bayesian Information Criterion (BIC) of the fitted copula model.
    - copula: The fitted copula model object.
    - name: The name of the copula family used for fitting the model.
    """
    controls = pv.FitControlsBicop(family_set=[family])
    copula = pv.Bicop(data, controls=controls)
    name = str(copula.family).split(".")[1]
    loglik = copula.loglik()
    aic = copula.aic()
    bic = copula.bic()
    return {
        'loglik': loglik,
        'aic': aic,
        'bic': bic,
        'copula': copula,
        'name': name
    }