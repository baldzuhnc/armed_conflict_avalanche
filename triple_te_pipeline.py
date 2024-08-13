# ====================================================================================== #
# Module for pipelineing the analysis of causal graphs with neighborhood triplets.
# Author: Clemens Baldzuhn
# ====================================================================================== #

#imports
import os
os.chdir('/home/clemens/armed_conflict_avalanche/')

import sys
import time
import tqdm

from arcolanche.pipeline import *
import statsmodels.api as sm
 
from shapely.geometry import LineString
import json
from keplergl import KeplerGl

 
 
def extract_significant_edges(ava_pairs, ava_triples):
    significant_pairs = []
    for pair, (te, te_shuffle) in ava_pairs.items():
        if (te > te_shuffle).mean() >= (95/100): #threshold!
            significant_pairs.append([(pair[0], pair[1]), te])
    
    significant_triples = []
    for triplet, (te, te_shuffle) in ava_triples.items():
        if (te > te_shuffle).mean() >= (95/100): #threshold!
            significant_triples.append([(triplet[0], triplet[1], triplet[2]), te])
    
    return significant_pairs, significant_triples

def plot_save_ecdf(t,s, significant_pairs, significant_triples):
    tes = [te for _, te in significant_pairs]    
    
    ecdf = sm.distributions.ECDF(tes)
    x = np.sort(tes)
    y = ecdf(x)

    plt.plot(x, y, 'o', markersize=3, label='Pairs')
    
    triple_tes = [te for _, te in significant_triples]
    
    triple_ecdf = sm.distributions.ECDF(triple_tes)
    triple_x = np.sort(triple_tes)
    triple_y = triple_ecdf(triple_x)
    
    plt.plot(triple_x, triple_y, 'o', markersize=3, label='Triples')
    
    plt.title('CDF of TE Pairs and Triples')
    plt.xlabel('TE')
    plt.ylabel('CDF')
    plt.legend()
    
    #SAVE PLOT IN FIGURES
    plt.savefig(f"Results/triples_vs_pairs/TE_CDF_dt{t}_dx{s}.png")
    plt.close()


def save_map(t, s, polygons, significant_pairs, significant_triples):
    
    map_ = KeplerGl()

    edges_pairs = [(pair[0], pair[1]) for pair, _ in significant_pairs]
    edges_triples = [(triple[0], triple[1], triple[2]) for triple, _ in significant_triples]

    edges_pairs_df = pd.DataFrame(edges_pairs, columns=["source", "target"])
    edges_triples_df = pd.DataFrame(edges_triples, columns=["source", "target1", "target2"]) # source: x, target1: y, target2: z

    # First loop for significant pairs
    lines_pairs = []
    for index, row in edges_pairs_df.iterrows():
        source_polygon = polygons[polygons.index == row['source']].geometry.values[0]
        target_polygon = polygons[polygons.index == row['target']].geometry.values[0]

        # Create a line between centroids of source and target polygons
        line = LineString([source_polygon.centroid, target_polygon.centroid])
        lines_pairs.append({
            'type': 'Feature',
            'geometry': line.__geo_interface__,
            'properties': {
                'source_id': int(row['source']),  # Ensure integer conversion
                'target_id': int(row['target'])   # Ensure integer conversion
            }
        })

    #add to map
    lines_geojson = {'type': 'FeatureCollection', 'features': lines_pairs}
    lines_geojson = json.dumps(lines_geojson)
    map_.add_data(data=json.loads(lines_geojson), name=f'1st degree, pairs')

    # Second loop for significant triples
    lines_triples = []
    for index, row in edges_triples_df.iterrows():
        source_polygon = polygons[polygons.index == row['source']].geometry.values[0]
        target1_polygon = polygons[polygons.index == row['target1']].geometry.values[0]
        target2_polygon = polygons[polygons.index == row['target2']].geometry.values[0]

        # Create a joined polygon from target1 and target2
        combined_polygon = target1_polygon.union(target2_polygon)

        # Create a line between the centroid of source polygon and the centroid of the combined polygon
        line = LineString([source_polygon.centroid, combined_polygon.centroid])
        lines_triples.append({
            'type': 'Feature',
            'geometry': line.__geo_interface__,
            'properties': {
                'source_id': int(row['source']),  # Ensure integer conversion
                'target1_id': int(row['target1']),  # Ensure integer conversion
                'target2_id': int(row['target2'])   # Ensure integer conversion
            }
        })

    lines_geojson = {'type': 'FeatureCollection', 'features': lines_triples}
    lines_geojson = json.dumps(lines_geojson)
    map_.add_data(data=json.loads(lines_geojson), name=f'1st degree, triples')

    
    # Add polygons to the map
    polygons_geojson = polygons.to_json()
    map_.add_data(data=json.loads(polygons_geojson), name=f'Polygons, dx={s}')
    
    # Save the map as HTML
    map_.save_to_html(file_name=f'Results/kepler_triples/kepler_d{degree}_dt{t}_dx{s}.html')


#######################################################
###################### Run ############################
#######################################################

conflict_type = "battles"
gridix = 3
degree = 1

dt = [16, 32]
dx = [320]

#run avalanches for different scales, 
for t in tqdm.tqdm(dt):
    for s in dx:
        print(f'Computing dt={t} and dx={s}')
        start_time = time.time()
        ava_pairs = Avalanche(dt = t, dx = s, gridix=gridix, 
                              degree = degree, 
                              triples=False, 
                              setup_causalgraph=True,
                              construct_avalanche=False)
        
        ava_triples = Avalanche(dt = t, dx = s, gridix=gridix, 
                              degree = degree, #can only handle 1 degree anyways
                              triples=True, 
                              setup_causalgraph=True,
                              construct_avalanche=False)

        significant_pairs, significant_triples = extract_significant_edges(ava_pairs.pair_edges, ava_triples.pair_edges)
        
        plot_save_ecdf(t,s,significant_pairs, significant_triples)
        
        save_map(t, s, ava_pairs.polygons, significant_pairs, significant_triples)
        
        end_time = time.time()    
        print(f"dt:{t}, Time taken: {int((end_time - start_time) / 60)} minutes {int((end_time - start_time) % 60)} seconds")
        
        
