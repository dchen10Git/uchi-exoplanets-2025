# Slow, un-multithreaded version of run_trappist1_sims. Can be helpful for testing.

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

planet_names = ['b', 'c']

# Set where to save the data
base_dir = Path.cwd()
file_path = "test_sim.h5"

tstart = time()

n_sims = 1
outcomes = []
for i in range(n_sims):
    sim_id = i
    
    # Get random param values
    m_vals, r_vals, m_star, r_star, initial_P_ratios, Sigma_1au, K_factor = generate_params(planet_names)
    
    # Set custom initial conditions (for debugging)
    initial_P_ratios = [1.5]
    Sigma_1au = 1.5e-05
    K_factor = 10
    
    # Sim integration!
    outcome = t1.simulate_trappist1(m_vals, r_vals, m_star, r_star, initial_P_ratios, Sigma_1au, K_factor, planet_names, sim_id, file_path)
    outcomes.append(outcome)
    print(f"Sim ID: {sim_id} | Outcome: {outcome}")
    
with open("test_sim.pkl", "wb") as f:
    pickle.dump(outcomes, f)
    
with open("test_sim.pkl", "rb") as f:
    sim_outcomes = pickle.load(f)

# print(sim_outcomes)

print(f'Time elapsed: {time()-tstart:.4} sec')