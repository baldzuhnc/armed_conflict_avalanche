#imports
import os
os.chdir('/home/clemens/armed_conflict_avalanche/')
import sys
import time

from arcolanche.pipeline import *
 
import statsmodels.api as sm
import networkx as nx
from tqdm import tqdm

#mapshit
from shapely.geometry import LineString
import json
from keplergl import KeplerGl


#arguments
degree = int(sys.argv[1])
#dt = int(sys.argv[2])
#dx = int(sys.argv[3])
 
#Functions
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

def plot_save_ecdf(t, s,  significant_edges):
    #get TEss
    for d in range(1, degree+1):
        sig = significant_edges[d]
        tes = [te for tuples, te in sig]
        
        # Compute ECDFs using statsmodels
        ecdf = sm.distributions.ECDF(tes)
        x = np.sort(tes)
        y = ecdf(x)
        plt.plot(x, y, 'o', label=f"{d}st", color="C"+str(d), markersize=3)
    
    
    # Add labels and title
    plt.ylabel("CDF")
    plt.xlabel("TE")
    plt.title(f"TE CDF d{degree} dt{t} dx{s}")

    # Add legend
    plt.legend()

    #SAVE PLOT IN FIGURES
    plt.savefig(f"Results/TE_CDF_d{degree}_dt{t}_dx{s}.png")
    plt.close()

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
    map_.save_to_html(file_name=f'Results/kepler/kepler_d{degree}_dt{t}_dx{s}.html')


# ==================== # RUN # ==================== #
conflict_type = "battles"
gridix = 3

dt = [32, 64, 128]
dx = [320]


for t in tqdm(dt):
    for s in dx:
        print(f'Computing dt={t} and dx={s}')
        start_time = time.time()
        #Creates avalanche object, only creates causal graph
        ava = Avalanche(dt = t, dx = s, gridix=gridix, degree=degree, setup_causalgraph=True, construct_avalanche=False)

        #save graph
        G = ava.causal_graph
        nx.write_graphml(G, f"Results/d{degree}_dt{t}_dx{s}.graphml")

        #get (tuple), TEs
        self_edges = ava.self_edges
        pair_edges = ava.pair_edges

        summary_df, significant_edges = calculate_significant_edges(self_edges, pair_edges, summary = True)
        #save summary df
        summary_df.to_csv(f"Results/d{degree}_dt{t}_dx{s}.csv")

        #save ecdf plot
        plot_save_ecdf(t, s, significant_edges)

        #save map    
        save_map(t, s, ava.polygons, significant_edges)

        end_time = time.time()    
        print(f"dt:{t}, Time taken: {int((end_time - start_time) / 60)} minutes {int((end_time - start_time) % 60)} seconds")