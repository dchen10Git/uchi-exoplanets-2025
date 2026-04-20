# Dask version of the trappist1_keller.ipynb notebook
# To run, use python3 -W ignore run_trappist1_sims_v2.py

import numpy as np
import scipy
import dask
from dask.distributed import Client, LocalCluster
import pickle as pkl
import os
import sys
import trappist1_sim as t1
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

def generate_params(planet_names, rng):
    # Nested dict containing params for each planet in sim
    # Randomly generate mass, radius, & semimajor axis values
    planet_params = {f"{planet_name}": t1.generate_params_from_csv(f'TRAPPIST-1_params/TRAPPIST-1_{planet_name}_planet_params.csv', ('a (au)', 'Rp (R⨁)', 'Mp (M⨁)'), random=False) for planet_name in planet_names}
    stellar_params = t1.generate_params_from_csv('TRAPPIST-1_params/TRAPPIST-1_stellar_params.csv', ('R✶ (R⦿)', 'M✶ (M⦿)'), random=False)
    
    # Define planet masses (m)
    m_vals = np.array([planet_params[planet_name]['Mp (M⨁)'] for planet_name in planet_names])
    m_vals *= m_earth # convert to Msun

    # Define planet radii (r)
    r_vals = np.array([planet_params[planet_name]['Rp (R⨁)'] for planet_name in planet_names])
    r_vals *= r_earth # convert to AU

    # Define stellar parameters
    m_star = stellar_params['M✶ (M⦿)']
    r_star = stellar_params['R✶ (R⦿)'] * r_sun

    # Uniform random initial period ratios just above 2:1
    initial_P_ratios = rng.uniform(2.05, 2.1, size=len(planet_names)-1) 
    
    initial_P_ratios = np.full(len(planet_names)-1, 1.55)
                                        
    # Draw surface density at 1au from log uniform
    Sigma_1au = scipy.stats.loguniform.rvs(a=50, b=1000, size=1, random_state=rng) # in g/cm^2
                                # In Keller, 10 & 10000
                                
    Sigma_1au = 50 # no random
    Sigma_1au *= AU**2 / Msun # unit conversion for sim

    # Draw K-factor from log uniform and solve for h
    K_factor = scipy.stats.loguniform.rvs(a=10, b=200, size=1, random_state=rng)
                                # In Keller, 10 & 1000
    K_factor = 50 # no random
                                
    return m_vals, r_vals, m_star, r_star, initial_P_ratios, Sigma_1au, K_factor

planet_names = ['b', 'c', 'd', 'e', 'f', 'g', 'h']
planet_names = ['b', 'c', 'd', 'e']

# Remember to change these before running each time
dataset_id = 14
n_sims = 10

def run_sim(sim_id):
    # Different rng for each sim
    rng = np.random.default_rng(seed=sim_id + os.getpid())
    
    # Set where to save the data
    base_dir = Path.cwd()
    file_path = base_dir.parent / "sim_results" / f"dataset{dataset_id}" / f"sim{sim_id}.h5"
    
    # Get random param values
    m_vals, r_vals, m_star, r_star, initial_P_ratios, Sigma_1au, K_factor = generate_params(planet_names, rng)
    
    # Sim integration!
    outcome = t1.simulate_trappist1(m_vals, r_vals, m_star, r_star, initial_P_ratios, Sigma_1au, K_factor, planet_names, sim_id, file_path, 
                                    integrator="whfast", test=False)
    print(f"Sim ID: {sim_id:<2d} | Outcome: {outcome}")
    return (sim_id, m_vals, r_vals, m_star, r_star, initial_P_ratios, Sigma_1au, K_factor, outcome)
    
if __name__ == "__main__":
    dataset_dir = Path.cwd().parent / "sim_results" / f"dataset{dataset_id}"
    
    # Create the folder
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {dataset_dir}")

    print(f"Dataset: {dataset_id}")
    tstart = time()
    
    # Start a local Dask cluster matching available CPU count
    cluster = LocalCluster()
    client = Client(cluster)
    print(f"Running sims on {len(client.scheduler_info()['workers'])} workers")
    print(f"Dask dashboard: {client.dashboard_link}")

    try:
        # Submit all simulations as Dask futures
        futures = [client.submit(run_sim, sim_id) for sim_id in range(n_sims)]
        
        # Gather results (blocks until all futures are complete)
        outcomes = client.gather(futures)
    finally:
        client.close()
        cluster.close()
    
    # Save the outcomes
    outcome_file = f"../sim_results/dataset{dataset_id}/outcomes.pkl"
    with open(outcome_file, "wb") as f:
        pkl.dump(outcomes, f)
        print(f"Saved to {outcome_file}")
    
    # Load to verify
    with open(outcome_file, "rb") as f:
        sim_outcomes = pkl.load(f)
    
    print(f'Time elapsed: {np.round(time()-tstart)} sec')