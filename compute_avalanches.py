# ====================================================================================== #
# Module for pipelineing the construction of avalanches from conflict data.
# Author: Clemens Baldzuhn
# ====================================================================================== #

import os
os.chdir('/home/clemens/armed_conflict_avalanche/')

from arcolanche.pipeline import *
from workspace.utils import save_pickle


def save_avalanche(ava, conflict_type, gridix, dt, dx, degree):
    ava_box = [[tuple(i) for i in ava.time_series.loc[a].drop_duplicates().values[:, ::-1]] for a in ava.avalanches]
    ava_event = ava.avalanches
    
    path = f"Results/avalanches/{conflict_type}/gridix_{gridix}/"
    if not os.path.exists(path):
        os.makedirs(path)
        
    save_pickle(["ava_box", "ava_event"], f"{path}/d{degree}_ava_{str(dt)}_{str(dx)}.p", True)


conflict_type = "battles"
dt = 32
dx = 320
gridix = 3
degree = [1,2,3,4,5]

for d in degree:
    ava = Avalanche(dt = dt, dx = dx, gridix=gridix, degree=d, setup_causalgraph=True, construct_avalanche=True)
    save_avalanche(ava, conflict_type, gridix, dt, dx, d)