# ====================================================================================== #
# Utility functions for the avalanche project.
# Author: Clemens Baldzuhn
# ====================================================================================== #

import os
os.chdir('/home/clemens/armed_conflict_avalanche')
import warnings
import pickle

from workspace.utils import load_pickle, save_pickle
from arcolanche.pipeline import set_ax
from voronoi_globe.interface import load_voronoi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as stats

#plotting
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
import cartopy.crs as ccrs
import geopandas as gpd

import pyvinecopulib as pv


# Subsetting and plotting cells
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

def get_ids_from_centroid(df, size, centroid):
        neighbors = df.loc[centroid].neighbors
        if size == 1:
            return [centroid]+neighbors
        else:
            for _ in range(size-1):
                new_neighbors = []
                for neighbor in neighbors:
                    new_neighbors += df.loc[neighbor].neighbors
                neighbors = list(set(new_neighbors))
            return neighbors

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
    
    
# plot avalanches    
def plot_avalanches(avalanche, dt, dx, gridix, conflict_type, degree, save=False):
    
    # Combine conflict events with avalanches
    load_pickle(f"avalanches/{conflict_type}/gridix_{gridix}/te/conflict_ev_{str(dt)}_{str(dx)}.p")
    ava_event = avalanche['ava_event']
    avalanche_number_dict = dict.fromkeys(conflict_ev.index, 0)

    for ava, index in zip(ava_event, range(len(ava_event))):
        for event in ava:
            avalanche_number_dict[event] = index

    conflict_ev["avalanche_number"] = avalanche_number_dict.values()
    # Combine conflict events with avalanches
    
    
    #Plot
    african_countries_iso = [
        "DZA", "AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD",
        "COM", "COG", "COD", "CIV", "DJI", "EGY", "GNQ", "ERI", "SWZ", "ETH",
        "GAB", "GMB", "GHA", "GIN", "GNB", "KEN", "LSO", "LBR", "LBY", "MDG",
        "MWI", "MLI", "MRT", "MUS", "MAR", "MOZ", "NAM", "NER", "NGA", "RWA",
        "STP", "SEN", "SYC", "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA", "TGO",
        "TUN", "UGA", "ZMB", "ZWE"
    ]

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world[['continent', 'geometry']]
    continents = world.dissolve(by='continent')
    africa = gpd.GeoDataFrame(continents["geometry"]["Africa"], geometry=0)

    african_countries_shp = gpd.read_file("data/africa-outline-with-countries_6.geojson")

    dx = dx
    gridix = gridix
    conflict_type = "all"

    #conflict_ev = load_conflict_ev(dx, gridix, conflict_type) #i dont have this function
    avas = conflict_ev.groupby("avalanche_number") #avalanche number in conflict ev?

    ava_ixs = avas.size()[avas.size() > 1].index

    points = []
    colors = []

    for ava_ix in ava_ixs:
        color = (np.random.random(), np.random.random(), np.random.random())
        ava = avas.get_group(ava_ix)
        ava = ava[~ava.duplicated(subset=["latitude", "longitude"], keep='first')]
        points += list(ava.geometry)
        colors += [color for _ in range(len(ava))]

    data = {"geometry": points, "color": colors}
    data = gpd.GeoDataFrame(data, geometry="geometry")

    country = "all"

    if country == "all":
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(projection=ccrs.PlateCarree())
        coastline_width = 0.6
        country_border_width = 0.2
        alpha = 0.2
        markersize = 5

        africa.plot(ax=ax, linewidth=coastline_width, edgecolor='black', facecolor='none')
        for ISO3 in african_countries_iso:
            african_countries_shp[african_countries_shp["iso_a3"] == ISO3].plot(ax=ax, linewidth=country_border_width, edgecolor='black', facecolor='none')
    else:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection=ccrs.PlateCarree())
        coastline_width = 0.8
        country_border_width = 0.2
        alpha = 0.6
        markersize = 5.8

        ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    ax.add_feature(cfeature.OCEAN)

    data.plot(ax=ax, column="color", markersize=markersize, alpha=alpha)

    ax.spines['geo'].set_linestyle('-')
    ax.spines['geo'].set_color('none')
    
    ax.set_title(f"Avalanches from {degree} degree causal graphs in on dt={dt}, dx={dx}")

    ax.set_extent(set_ax(f"{country}"))
    
    if save:
        plt.savefig(f"Results/plots/avalanches/avalanches_d{degree}_dt{dt}_dx{dx}.png")

    return ax


# Copulas
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
    
    
    

# Saving and loading avalanches
def save_avalanche(ava, conflict_type, gridix, dt, dx, degree):
    ava_box = [[tuple(i) for i in ava.time_series.loc[a].drop_duplicates().values[:, ::-1]] for a in ava.avalanches]
    ava_event = ava.avalanches
    
    path = f"Results/avalanches/{conflict_type}/gridix_{gridix}/"
    if not os.path.exists(path):
        os.makedirs(path)
        
    save_pickle(["ava_box", "ava_event"], f"{path}/d{degree}_ava_{str(dt)}_{str(dx)}.p", True)


def open_avalanche(conflict_type, gridix, dt, dx, degree):
    path = f"Results/avalanches/{conflict_type}/gridix_{gridix}/d{degree}/d{degree}_ava_{str(dt)}_{str(dx)}.p"
    
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data