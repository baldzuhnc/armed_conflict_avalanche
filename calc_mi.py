import os
os.chdir('/home/clemens/armed_conflict_avalanche/')
import sys
import time

from arcolanche.pipeline import *

#mapshit
from shapely.geometry import LineString
import json
from keplergl import KeplerGl

import tqdm


def calculate_sig_ratio(sig_mi_tuples, te_tuples):
    all_te_tuples = []
    sig_te_tuples = []

    for poly, (te, te_shuffle) in te_tuples.items():
        all_te_tuples.append((poly[0], poly[1]))
        if (te > te_shuffle).mean() >= (95 / 100):
            sig_te_tuples.append((poly[0], poly[1]))

    all_mi_in_te = all([x in all_te_tuples for x in sig_mi_tuples])
    ratio_sig = len([x for x in sig_mi_tuples if x in sig_te_tuples]) / len(sig_mi_tuples)

    with open('Results/sig_ratio.txt', 'w') as file:
        file.write(f"dt={t}, dx={s}\n")
        file.write(f"All MI tuples in TE: {all_mi_in_te}\n")
        file.write(f"Ratio of MI tuples significant in TE: {ratio_sig:.3f}\n")

def calculate_significant_edges(self_edges, pair_edges, summary = False):
        
    significant_edges = [[] for _ in range(degree+1)]
    counts = np.zeros(degree+1)
    
    #self edges
    for poly, (te, te_shuffle) in self_edges.items():
        if (te > te_shuffle).mean() >= (95 / 100):
            significant_edges[0].append([(poly, poly), te])
    counts[0] = len(self_edges)

    #pair edges
    for pair, (te, te_shuffle) in pair_edges.items():
        d = pair[2]
        if 1 <= d <= degree:
            counts[d] += 1
            if (te > te_shuffle).mean() >= (95/100): #threshold!
                significant_edges[d].append([(pair[0], pair[1]), te])            
    
    #ratio dataframe 
    if summary:
        df = pd.DataFrame(
            {
                "Degree": np.arange(degree+1),
                "Total edges": counts,
                "Significant edges": [len(edges) for edges in significant_edges],
            }
        )
        df["Ratio"] = round(df["Significant edges"] / df["Total edges"], 3)
        
        return df, significant_edges
    
    else:
        return significant_edges




def save_map(t, s, polygons, significant_edges): 
    map_ = KeplerGl()
    
    for d in range(1, degree+1):
        sig = significant_edges[d]
        edges = [(pair[0], pair[1]) for pair, te in sig]
        edges_df = pd.DataFrame(edges, columns=["source", "target"])
        lines = []

        for index, row in edges_df.iterrows():
            source_polygon = polygons[polygons.index == row['source']].geometry.values[0]
            target_polygon = polygons[polygons.index == row['target']].geometry.values[0]

            # Create a line between centroids of source and target polygons
            line = LineString([source_polygon.centroid, target_polygon.centroid])
            lines.append({
                'type': 'Feature',
                'geometry': line.__geo_interface__,
                'properties': {
                    'source_id': int(row['source']),  # Ensure integer conversion
                    'target_id': int(row['target'])   # Ensure integer conversion
                }
            })

        lines_geojson = {'type': 'FeatureCollection', 'features': lines}
        lines_geojson = json.dumps(lines_geojson)
        map_.add_data(data=json.loads(lines_geojson), name=f'{d}st degree')
    
    polygons_geojson = polygons.to_json()
    map_.add_data(data=json.loads(polygons_geojson), name=f'Polygons, dx={s}')
    
    # Save the map as HTML or display it
    map_.save_to_html(file_name=f'Results/kepler_mi/kepler_mi{mi_threshold}_d{degree}_dt{t}_dx{s}.html')

 
# ==================== # RUN # ==================== #

conflict_type = "battles"
gridix = 3
degree = 2
mi_threshold = 0.05

dt = [32, 64, 128]
dx = [320]

start_time = time.time()

for t in tqdm.tqdm(dt):
    for s in dx:
        print(f'Computing dt={t} and dx={s}')
        start_time = time.time()

        ava_mi = Avalanche(dt = t, dx = s,
                        gridix=gridix,
                        conflict_type= conflict_type,
                        degree = degree,
                        size = None,
                        triples = False,
                        sig_threshold=95,
                        mi_connections = True,
                        mi_threshold = mi_threshold,   
                        rng=None,
                        iprint=False,
                        setup_causalgraph=True,
                        construct_avalanche=False,
                        shuffle_null=False,
                        year_range=False)

        sig_mi_edges = ava_mi.mi_edges
        
        self_edges = ava_mi.self_edges
        pair_edges = ava_mi.pair_edges
        
        #print ratio of significant edges
        #calculate_sig_ratio(sig_mi_edges, pair_edges)

        significant_edges = calculate_significant_edges(self_edges, pair_edges, summary = False)

        save_map(t, s, ava_mi.polygons, significant_edges)
        
        end_time = time.time()    
        print(f"dt:{t}, Time taken: {int((end_time - start_time) / 60)} minutes {int((end_time - start_time) % 60)} seconds")