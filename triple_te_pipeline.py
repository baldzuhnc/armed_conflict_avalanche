#imports
import os
os.chdir('/home/clemens/armed_conflict_avalanche/')

import sys
import time

from arcolanche.pipeline import *
from utils_CB import save_avalanche, open_avalanche

#arguments

zp = True if sys.argv[1].lower() == "true" else False
print(zp)
#dt = int(sys.argv[2])
#dx = int(sys.argv[3])

#######################################################
###################### Run ############################
#######################################################

