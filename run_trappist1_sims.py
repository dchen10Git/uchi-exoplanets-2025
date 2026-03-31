# Multiprocessing version of the trappist1_keller.ipynb notebook
# To run, use python3 -W ignore run_trappist1_sims.py

import numpy as np
import pandas as pd
import scipy
import multiprocessing
import matplotlib.pyplot as plt
import pickle as pkl
import trappist1_sim as t1
import mmr_id
from time import time
from astropy import units as u
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Unit conversions
AU = u.AU.to(u.cm)    
Msun = u.Msun.to(u.g) 
yr = u.yr.to(u.s)    
r_earth = u.earthRad.to(u.AU)
m_earth = u.Mearth.to(u.Msun)
r_sun = u.Rsun.to(u.AU) 

def generate_params(planet_names):
    # Nested dict containing params for each planet in sim
    # Randomly generate mass, radius, & semimajor axis values
    planet_params = {f"{planet_name}": t1.generate_params_from_csv(f'TRAPPIST-1_params/TRAPPIST-1_{planet_name}_planet_params.csv', ('a (au)', 'Rp (R⨁)', 'Mp (M⨁)'), random=False) for planet_name in planet_names}
    stellar_params = t1.generate_params_from_csv('TRAPPIST-1_params/TRAPPIST-1_stellar_params.csv', ('R✶ (R⦿)', 'M✶ (M⦿)'), random=False)
    
    # Define planet masses (m)
    m_vals = np.array([planet_params[planet_name]['Mp (M⨁)'] for planet_name in planet_names])

    # m_vals[0] = np.random.uniform(1, 1.5) # (actual observed is 1.37 M_earth) # random b mass
    # m_vals[1] = np.random.uniform(1, 1.5) # (actual observed is 1.31 M_earth) # random c mass
    m_vals[2] = np.random.uniform(0.2, 0.6) # (actual observed is 0.39 M_earth) # random d mass
    
    m_vals *= m_earth # convert to Msun

    # Define planet radii (r)
    r_vals = np.array([planet_params[planet_name]['Rp (R⨁)'] for planet_name in planet_names])
    r_vals *= r_earth # convert to AU

    # Define stellar parameters
    m_star = stellar_params['M✶ (M⦿)']
    r_star = stellar_params['R✶ (R⦿)'] * r_sun

    # Draw initial ratios from log normal
    initial_P_ratios = np.random.lognormal(0.6, 0.2, size=len(planet_names)-1) 
                                        # In Keller, 0.703 & 0.313
                                        
    # NOTE: This is set to simplify the current problem
    initial_P_ratios = np.full(len(planet_names)-1, 1.9)
                                        
    # Draw surface density at 1au from log uniform
    Sigma_1au = scipy.stats.loguniform.rvs(a=50, b=1000, size=1) # in g/cm^2
                                # In Keller, 10 & 10000
    Sigma_1au *= AU**2 / Msun # unit conversion for sim

    # Draw K-factor from log uniform and solve for h
    K_factor = scipy.stats.loguniform.rvs(a=10, b=200, size=1)
                                # In Keller, 10 & 1000
                                
    return m_vals, r_vals, m_star, r_star, initial_P_ratios, Sigma_1au, K_factor

planet_names = ['b', 'c', 'd', 'e', 'f', 'g'] # h-less sytem

# Remember to change these before running each time
dataset_id = 10
n_sims = 2000

def run_sim(sim_id):
    # Set where to save the data
    base_dir = Path.cwd()
    file_path = base_dir.parent / "sim_results" / f"dataset{dataset_id}" / f"sim{sim_id}.h5"
    
    # Get random param values
    m_vals, r_vals, m_star, r_star, initial_P_ratios, Sigma_1au, K_factor = generate_params(planet_names)
    
    # Sim integration!
    outcome = t1.simulate_trappist1(m_vals, r_vals, m_star, r_star, initial_P_ratios, Sigma_1au, K_factor, planet_names, sim_id, file_path)
    print(f"Sim ID: {sim_id:<2d} | Outcome: {outcome}")
    return (sim_id, m_vals, r_vals, m_star, r_star, initial_P_ratios, Sigma_1au, K_factor, outcome)
    
if __name__ == "__main__":
    print(f"Dataset: {dataset_id}")
    tstart = time()
    
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Map the simulation function to the sim_id's
        outcomes = pool.map(run_sim, range(n_sims))
    
    # Save the outcomes
    outcome_file = f"../sim_results/dataset{dataset_id}/outcomes.pkl"
    with open(f"../sim_results/dataset{dataset_id}/outcomes.pkl", "wb") as f:
        pkl.dump(outcomes, f)
        print(f"Saved to {outcome_file}")
    
    # Load to verify
    with open(outcome_file, "rb") as f:
        sim_outcomes = pkl.load(f)
    
    # print(sim_outcomes)
    
    print(f'Time elapsed: {np.round(time()-tstart)} sec')
    
    
'''
Dataset documentation

test: for testing
0-1: params: K, Sigma1au, P_ratios; outcome: single value
2-6: params: K, Sigma1au, P_ratios; outcome: vectorized
7: params: K, Sigma1au; P_ratios kept constant at 1.8 for all; outcome: vectorized
8: params: K, Sigma1au; P_ratios kept constant at 1.9 for all; outcome: vectorized
9-10: params: K, Sigma1au, mass of d; outcome: vectorized
11: params: K, Sigma1au, masses of b, c, & d; P_ratios kept constant at 1.8 for all; outcome: vectorized
12: params: K, Sigma1au, P_ratios in U(1.8, 1.9); outcome: vectorized
'''
