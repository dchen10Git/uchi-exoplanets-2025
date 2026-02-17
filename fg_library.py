# Import celmech (in different directory)
import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('trappist1_keller.ipynb'), '..')))
from celmech.celmech.disturbing_function import get_fg_coefficients

def build_fg_library(p_max, q_max):
    assert p_max >= q_max
    
    fg_library = {}
    
    for p in range(1, p_max+1):
        for q in range(1, p):
            fg = get_fg_coefficients(p, q)
            fg_library[(p, q)] = fg

    return fg_library

fg_lib = build_fg_library(p_max=10, q_max=7)
print("fg library created")
with open("fg_library.pkl", "wb") as f:
    pickle.dump(fg_lib, f)