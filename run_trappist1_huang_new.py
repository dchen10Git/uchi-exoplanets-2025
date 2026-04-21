import numpy as np
from astropy import units as u
from time import time
import pickle as pkl
from dask.distributed import Client, LocalCluster
import rebound
import reboundx
import os
from pathlib import Path
import trappist1_sim as t1

# Unit conversions
AU = u.AU.to(u.cm)    
Msun = u.Msun.to(u.g) 
yr = u.yr.to(u.s)    
r_earth = u.earthRad.to(u.AU)
m_earth = u.Mearth.to(u.Msun)
r_sun = u.Rsun.to(u.AU) 

# Free parameters
tau_a_earth = 5e3
C_e = 0.1 
r_c = 0.023
A_a = 100
A_e = 40
    
# Fixed parameters
h = 0.03
M_g_dot = 10e-10
M_star = 0.0898
gamma_I = 2
tau_d = 1e5
Delta = 2*h*r_c
Q_sim = 100
q_earth =  3.003e-6 / M_star
    
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
    initial_P_ratios = rng.uniform(1.53, 1.55, size=len(planet_names)-1) 
    initial_P_ratios = [1.53, 1.53, 1.8] # for b/c/d/e in early cavity infall model
                                    
    return m_vals, r_vals, m_star, r_star, initial_P_ratios

def f_functions(r, r_c, Delta, A_a, A_e):
    # Piecewise functions f_a and f_e
    conditions = [
        r < r_c - Delta,
        (r_c - Delta <= r) & (r < r_c),
        (r_c <= r) & (r < r_c + Delta + Delta/A_a),
        r >= r_c + Delta + Delta / A_a
    ]

    f_a = [
        0,          
        A_a * (r_c - Delta - r) / Delta,
        A_a * (r - r_c - Delta) / Delta,
        1
    ]

    f_e = [
        0,          
        A_e * (r - r_c + Delta) / Delta,
        (A_e - 1) * (r_c + Delta + Delta/A_a - r) / (Delta + Delta/A_a) + 1, 
        1
    ]

    f_a_vals = np.select(conditions, f_a, default=np.nan)
    f_e_vals = np.select(conditions, f_e, default=np.nan)
    return f_a_vals, f_e_vals

def get_taus(a_vals, parameters):
    m_vals, m_star, r_vals, r_c, Delta, A_a, A_e, C_e, Q_sim = parameters["m_vals"], parameters["m_star"], parameters["r_vals"], parameters["r_c"], parameters["Delta"], parameters["A_a"], parameters["A_e"], parameters["C_e"], parameters["Q_sim"]
    q_vals = m_vals / m_star
    f_a_vals, f_e_vals = f_functions(a_vals, r_c, Delta, A_a, A_e)
    tau_a = - tau_a_earth * (q_earth / q_vals) / f_a_vals # negative so damping?
    tau_e_disk = C_e * tau_a * f_a_vals * (h**2) / f_e_vals
    tau_e_star = 7.63e5 * Q_sim * (m_vals/m_earth) * (1/m_star)**1.5 * (r_earth/r_vals)** 5 * (a_vals/0.05)**6.5
    tau_e = (tau_e_disk * tau_e_star) / (tau_e_disk + tau_e_star) # combining the two based on Eqs. 4 and 13
    return tau_a, tau_e

def integrate_sim(sim, planets, planet_names, parameters, years, start_time=0):
    '''
    Integrates a REBOUND simulation over a given number of years,
    saves the new state of the sim and returns the data as a Pandas DataFrame. 
    Also returns whether the integration completed as a bool.
    '''
    m_vals, m_star = parameters["m_vals"], parameters["m_star"]
    num_planets = len(planet_names)
    
    # Set up times for integration & data collection
    n_out = int(years) # number of data points to collect
    stage_times = np.linspace(start_time, years+start_time, n_out, endpoint=False)  # all times to integrate over
    stage_data = {name : t1.data_df(n_out, stage_times) for name in planet_names[:num_planets]}
    sim.random_seed = 13741154 # for reproducibility

    completed_sim = True
    
    for i, t in enumerate(stage_times): 
        sim.dt = planets[0].P / 20 # 1/20 of planet b
        sim.integrate(t)
        
        a_vals = np.array([p.a for p in sim.particles[1:]])
        tau_a, tau_e = get_taus(a_vals, parameters)
        
        for p in range(num_planets):
            # Update damping timescales
            planets[p].params["tau_a"] = tau_a[p]
            planets[p].params["tau_e"] = tau_e[p]
            
            # save data
            name = planet_names[p]
            stage_data[name].loc[i, "a"] = planets[p].a
            stage_data[name].loc[i, "e"] = planets[p].e
            stage_data[name].loc[i, "l"] = planets[p].l
            stage_data[name].loc[i, "pomega"] = planets[p].pomega   
            
            if p != num_planets-1: # don't record period ratio for last planet
                stage_data[name].loc[i, "P_ratio"] = planets[p+1].P / planets[p].P
                                
                # Stop sim if separation within 5*r_hill
                r_hill = t1.get_hill_radius(m_vals[p], a_vals[p], m_vals[p+1], a_vals[p+1], m_star)
                if np.abs(a_vals[p] - a_vals[p+1]) < 5*r_hill:
                    completed_sim = False
                    break
                    
                # Also stop sim if planets crossed each other(P_ratio < 1)
                if planets[p+1].P / planets[p].P < 1:
                    completed_sim = False
                    break
                
            # Stop sim if planet goes into star
            if planets[p].a < 0.001:
                completed_sim = False
                break
        
        # Prevent stop in data collection       
        if np.isnan(stage_data['b']["a"][i]):
            completed_sim = False
            break

        # Stop simulation early if failed
        if not completed_sim:
            break
    
    return stage_data, completed_sim

def simulate_trappist1(sim_id, file_path, planet_names, parameters, integrator="whfast"):
    m_vals, m_star, r_vals, r_star, r_c, Delta, A_a, initial_P_ratios = parameters["m_vals"], parameters["m_star"], parameters["r_vals"], parameters["r_star"], parameters["r_c"], parameters["Delta"], parameters["A_a"], parameters["initial_P_ratios"]
    
    # Create the simulation
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.integrator = integrator
    
    # Add the star
    sim.add(m=m_star, r=r_star)
    num_planets = len(planet_names)
    
    # Define initial periods (P) and semimajor axes (a) 
    P_vals = [((r_c+Delta+Delta/A_a)**3 / m_star)**(1/2) / 1.53 / 1.53] # for d to be at the disk edge (0.023)
    for i in range(num_planets-1):
        P_vals = np.append(P_vals, P_vals[i] * initial_P_ratios[i])
        
    a_vals = (P_vals**2 * parameters["m_star"])**(1/3)

    # Add planets 
    for i in range(num_planets):
        sim.add(m=m_vals[i], r=r_vals[i], a=a_vals[i])

    # Move to center of momentum
    sim.move_to_com()
    ps = sim.particles
    planets = ps[1:] # for easier indexing; ps[0] = planet b
    
    rebx = reboundx.Extras(sim)
    mof = rebx.load_force("modify_orbits_forces")
    rebx.add_force(mof)

    years = -get_taus(a_vals, parameters)[0][-1] # tau_a of the last planet
    if years < 1000:
        years = 1000
        
    print(f"Sim ID: {sim_id:<2d} | years: {years:.3g}")
    data, complete_sim = integrate_sim(sim, planets, planet_names, parameters, years, start_time=0)
    
    # Save data
    t1.save_simulation_run(data, sim_id, file_path, sim_metadata={
                        "num_planets": num_planets, 
                        "planet_names": planet_names} | parameters)
        
planet_names = ['b', 'c', 'd', 'e']
# Remember to change these before running each time
dataset_id = 15
n_sims = 1

def run_sim(sim_id):
    # Different rng for each sim
    rng = np.random.default_rng(seed=sim_id + os.getpid())
    
    # Set where to save the data
    base_dir = Path.cwd()
    file_path = base_dir.parent / "sim_results" / f"dataset{dataset_id}" / f"sim{sim_id}.h5"
    
    # Get random param values
    m_vals, r_vals, m_star, r_star, initial_P_ratios = generate_params(planet_names, rng)
    tau_a_earth = (sim_id//10)*1e3 + 1e4
    C_e = 0.1*sim_id + 0.1
    tau_1s = 1/(0.0054/tau_a_earth) # damping on c when in disk
    tau_pl = 2e4 # planet formation interval time-scale
    
    parameters = {"m_vals": m_vals,
                  "m_star": m_star,
                  "r_vals": r_vals,
                  "r_star": r_star,
                  "r_c": r_c,
                  "Delta": Delta,
                  "A_a": A_a,
                  "A_e": A_e,
                  "C_e": C_e,
                  "tau_a_earth": tau_a_earth,
                  "Q_sim": Q_sim,
                  "tau_1s": tau_1s,
                  "tau_pl": tau_pl,
                  "initial_P_ratios": initial_P_ratios,
                }
    
    # Sim integration!
    outcome = simulate_trappist1(sim_id, file_path, planet_names, parameters, integrator="trace")
    return (sim_id, m_vals, r_vals, m_star, r_star, initial_P_ratios)
    
if __name__ == "__main__":
    dataset_dir = Path.cwd().parent / "sim_results" / f"dataset{dataset_id}"
    
    # Create the folder
    dataset_dir.mkdir(parents=True, exist_ok=True) # change to False to be safe
    print(f"Created directory: {dataset_dir}")

    print(f"Dataset: {dataset_id}")
    tstart = time()

    # Start a local Dask cluster
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print(f"CPUs: {n_cpus}")
    cluster = LocalCluster(
        n_workers=n_cpus,
        threads_per_worker=1,
        processes=True
    )    
    client = Client(cluster)
    
    # print(f"Running sims on {len(client.scheduler_info()['workers'])} workers")
    # print(f"Dask dashboard: {client.dashboard_link}")

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