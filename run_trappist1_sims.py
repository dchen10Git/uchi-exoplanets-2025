# Identical to trappist1_keller.ipynb, but in .py format

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import pickle
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
                                        
    # Draw surface density at 1au from log uniform
    Sigma_1au = scipy.stats.loguniform.rvs(a=50, b=1000, size=1) # in g/cm^2
                                # In Keller, 10 & 10000
    Sigma_1au *= AU**2 / Msun # unit conversion for sim

    # Draw K-factor from log uniform and solve for h
    K_factor = scipy.stats.loguniform.rvs(a=10, b=200, size=1)
                                # In Keller, 10 & 1000
                                
    return m_vals, r_vals, m_star, r_star, initial_P_ratios, Sigma_1au, K_factor

planet_names = ['b', 'c', 'd', 'e', 'f', 'g', 'h']
planet_names = ['b', 'c', 'd', 'e', 'f', 'g']

# Set where to save the data
base_dir = Path.cwd()
file_path = base_dir.parent / "sim_results" / "simulation_data2.h5"

n_sims = 100
outcomes = []
for i in range(n_sims):
    sim_id = i
    print("==========================")
    print(f"sim_id: {i}")
    
    # Get random param values
    m_vals, r_vals, m_star, r_star, initial_P_ratios, Sigma_1au, K_factor = generate_params(planet_names)
    
    # Set custom initial conditions (for debugging)
    # initial_P_ratios = [1.3481, 1.3329, 1.3, 1.6013, 1.3]
    # Sigma_1au = 7.24e-05
    # K_factor = 131
    
    print(f"Initial P ratios: {np.round(initial_P_ratios, decimals=4)}")
    print(f"Sigma_1au: {np.round(Sigma_1au, decimals=7)}")
    print(f"K-factor: {np.round(K_factor)}")
    
    # Sim integration!
    outcome = t1.simulate_trappist1(m_vals, r_vals, m_star, r_star, initial_P_ratios, Sigma_1au, K_factor, planet_names, sim_id, file_path)
    outcomes.append(outcome)
    print(f"Outcome: {outcome}")
    
with open("trappist1_sim_outcomes.pkl", "wb") as f:
    pickle.dump(outcomes, f)
    
with open("trappist1_sim_outcomes.pkl", "rb") as f:
    sim_outcomes = pickle.load(f)

print(sim_outcomes)