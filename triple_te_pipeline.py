#imports
import os
os.chdir('/home/clemens/armed_conflict_avalanche/')

import sys
import time
import tqdm

from arcolanche.pipeline import *
import statsmodels.api as sm

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


#######################################################
###################### Run ############################
#######################################################

conflict_type = "battles"
gridix = 3
degree = 1

dt = [32, 64, 128]
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
        
        end_time = time.time()    
        print(f"dt:{t}, Time taken: {int((end_time - start_time) / 60)} minutes {int((end_time - start_time) % 60)} seconds")
        
        
