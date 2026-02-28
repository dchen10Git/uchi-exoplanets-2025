import re
import rebound
import reboundx
import numpy as np
import pandas as pd
import pickle
import scipy
import matplotlib.pyplot as plt
from time import time
from astropy import constants as const
from astropy import units as u
import warnings
warnings.filterwarnings('ignore')

# Unit conversions
AU = u.AU.to(u.cm)    
Msun = u.Msun.to(u.g) 
yr = u.yr.to(u.s)    
r_earth = u.earthRad.to(u.AU)
m_earth = u.Mearth.to(u.Msun)
r_sun = u.Rsun.to(u.AU) 

def parse_entry(entry):
    """
    Parses strings of form:
        '1.0±0.01'
        '1.04 +0.01 -0.02'
    
    Returns:
        mu (float), sigma (float)
    """
    
    if pd.isna(entry):
        raise ValueError("Entry is NaN")
    
    # Remove extra whitespace
    entry = entry.strip()
    
    # Remove spaces for easier parsing
    entry_nospace = entry.replace(" ", "")
    
    # Case 1: symmetric uncertainty (±)
    match_pm = re.match(r"^([0-9.+\-eE]+)±([0-9.+\-eE]+)$", entry_nospace)
    if match_pm:
        mu = float(match_pm.group(1))
        sigma = float(match_pm.group(2))
        return mu, sigma
    
    # Case 2: asymmetric uncertainty (+x -y)
    match_asym = re.match(
        r"^([0-9.+\-eE]+)\+([0-9.+\-eE]+)\-([0-9.+\-eE]+)$",
        entry_nospace
    )
    if match_asym:
        mu = float(match_asym.group(1))
        sigma_plus = float(match_asym.group(2))
        sigma_minus = float(match_asym.group(3))
        
        # Convert asymmetric → effective symmetric σ
        sigma = 0.5 * (sigma_plus + sigma_minus)
        
        return mu, sigma
    
    raise ValueError(f"Could not parse entry: {entry}")

def generate_params(csv_file, params, n_samples=1):
    """
    Reads planet parameter CSV and returns Monte Carlo samples
    for mass, radius, and semimajor axis.
    """
    
    df = pd.read_csv(csv_file)
    
    # Set index to Source column for easy lookup
    df = df.set_index("Source")
    
    # Extract Agol et al. 2021 column
    col = "Agol et al. 2021"
    
    params_dict = {}
    
    for param in params:
        # Parse parameter
        mu, sigma = parse_entry(df.loc[param, col])
    
        # Draw Gaussian samples
        samples  = np.random.normal(mu, sigma, n_samples)

        # Add to dict
        params_dict[param] = samples
    
    return params_dict

def get_taus(a_vals, m_vals, M_star, h, Sigma, alpha=1.5):
    '''
    Computes damping timescales based on current semimajor axis values.
    
    Parameters:
        a_vals: 1D NumPy array of current semimajor axis values.
    
    Returns:
        tau_a: semimajor axis damping timescale.
        tau_e: eccentricity damping timescale.
    '''
    tau_a = (1/(2.7+1.1*alpha)) * (M_star/m_vals) * (M_star/(Sigma*a_vals**2)) * (h**2 / np.sqrt(const.G.value*M_star/a_vals**3))
    tau_e = (1/0.780) * (M_star/m_vals) * (M_star/(Sigma*a_vals**2)) * (h**4 / np.sqrt(const.G.value*M_star/a_vals**3))
    return tau_a, tau_e

def get_Sigma(a_vals, Sigma_1au, alpha=1.5):
    return Sigma_1au * a_vals**(-alpha)

def get_h(K, alpha=1.5):
    return np.sqrt(0.780/(2.7+1.1*alpha)/K)

def get_Sigma(Sigma_1au, a_vals, alpha=1.5):
    return Sigma_1au * (a_vals**-alpha)

def get_hill_radius(m1, a1, m2, a2, M_star):
    return ((m1+m2)/(3*M_star))**1/3 * (a1+a2)/2

def data_df(n_out, times):        
    return pd.DataFrame({
        "time": times,
        "a": np.zeros(n_out),
        "e": np.zeros(n_out),
        "P": np.zeros(n_out),
        "P_ratio": np.zeros(n_out),
        "l": np.zeros(n_out),
        "pomega": np.zeros(n_out)
    })

def integrate_sim(sim, num_planets, planets, planet_names, m_vals, m_star, years, start_time=0):
    '''
    Integrates a REBOUND simulation over a given number of years,
    saves the new state of the sim and returns the data as a Pandas DataFrame.
    '''
    # Set up times for integration & data collection
    n_out = 2000 # number of data points to collect
    stage_times = np.linspace(start_time, years+start_time, n_out, endpoint=False)  # all times to integrate over
    stage_data = {name : data_df(n_out, stage_times) for name in planet_names[:num_planets]}
    
    tstart = time()
    stop_sim = False
    
    # For showing progress
    percents = np.linspace(0, n_out, int(n_out/100), endpoint=False) # n_out should be multiple of 100

    for i, t in enumerate(stage_times): 
        sim.dt = planets[0].P / 20 # 1/20 of planet b
        sim.integrate(t)
        
        current_a_vals = np.array([p.a for p in sim.particles[1:]])
        
        for p in range(num_planets):
            name = planet_names[p]
            stage_data[name]["a"][i] = planets[p].a
            stage_data[name]["e"][i] = planets[p].e
            stage_data[name]["l"][i] = planets[p].l
            stage_data[name]["pomega"][i] = planets[p].pomega
            
            if p != num_planets-1: # don't record period ratio for last planet
                stage_data[name]["P_ratio"][i] = planets[p+1].P / planets[p].P
                                
                # Stop sim if separation within 5*r_hill
                # (since symplectic integrators are not designed to handle close encounters)
                r_hill = get_hill_radius(m_vals[p], current_a_vals[p], m_vals[p+1], current_a_vals[p+1], m_star)
                
                if np.abs(current_a_vals[p] - current_a_vals[p+1]) < 5*r_hill:
                    stop_sim = True
                    print("\nClose encounter")
                
            # Stop sim if planet goes into star
            if planets[p].a < 0.001:
                stop_sim = True
                print("\nPlanet collided with star")
        
        # Prevent stop in data collection        
        if type(stage_data['b']["a"][i]) != np.float64:
            stop_sim = True
            print(f"\nStopped collecting data at t={t}")
        
        if stop_sim:
            print(f"\nStopped integration at t={t}")
            break
        
        # Show progress
        if i in percents:
            print(f"Integration: {int(100*i/n_out)}% completed", end='\r', flush=True)
        
        if i == n_out-1:
            print("Integration: 100% completed", end='\r', flush=True)
            
    if not stop_sim:
        print(f'\nIntegrated to {(years+start_time)/1000} kyrs in {time()-tstart:.4} sec')
    else:
        print(f'\nTime elapsed: {time()-tstart:.4} sec')
    
    return stage_data
   
def concatenate_data(stages):
    if type(stages) == dict:
        return stages
    
    # Concatenate first two
    all_stage_data = {
            name: pd.concat(
                [stages[0][name], stages[1][name]],
                ignore_index=True
            )
            for name in stages[0]
        }
    
    # Concatenate the rest
    for i in range(2, len(stages)):
        next_stages = stages[i]
        all_stage_data = {
            name: pd.concat(
                [all_stage_data[name], next_stages[name]],
                ignore_index=True
            )
            for name in all_stage_data
        }
    return all_stage_data

def save_simulation_run(stage_data,
                        sim_id,
                        file_path,
                        sim_metadata=None):
    """
    Save all planets from one simulation run into HDF5.
    
    Parameters
    ----------
    stage_data : dict
        {planet_name: DataFrame}
    sim_id : int
        Simulation ID
    sim_metadata : dict (optional)
        e.g. {"m_star": 1.0, "integrator": "ias15"}
    """
    with pd.HDFStore(file_path, mode="a") as store:
        
        sim_group = f"/sim_{sim_id}"
        
        # Save planet list
        planet_list = list(stage_data.keys())
        store.put(f"{sim_group}/planet_list",
                  pd.Series(planet_list))
        
        # Save simulation metadata
        if sim_metadata is not None:
            store.put(f"{sim_group}/metadata",
                      pd.Series(sim_metadata))
        
        # Save each planet
        for planet_name, df in stage_data.items():
            
            key = f"{sim_group}/{planet_name}"
            store.put(key, df, format="table")
            
            # Attach attributes
            storer = store.get_storer(key)
            storer.attrs.planet_name = planet_name
            storer.attrs.sim_id = sim_id
           
def load_simulation_run(sim_id, file_path):
    '''
    Given sim_id and filename of hdf5, returns the result (dict of dataframes)
    and the metadata (containing planet_name, sim_id, ide params, etc.)
    '''
    sim_group = f"/sim_{sim_id}"
    result = {}
    
    with pd.HDFStore(file_path, mode="r") as store:
        
        planet_list = store[f"{sim_group}/planet_list"].tolist()
        
        for planet_name in planet_list:
            key = f"{sim_group}/{planet_name}"
            df = store[key]
            
            # Pull HDF5 attributes
            storer = store.get_storer(key)
            df.attrs["planet_name"] = storer.attrs.planet_name
            df.attrs["sim_id"] = storer.attrs.sim_id
            
            result[planet_name] = df
        
        metadata = store[f"{sim_group}/metadata"].to_dict()
    
    return result, metadata

def plot_trappist1(sim_data, t_units='kyr'):
    '''
    Takes dataframes for planets and plots a, e, and P ratios over time.
    '''
    times = sim_data[0]['b']['time']
    planet_names = list(sim_data[0].keys())
    num_planets = len(planet_names)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    fig.set_figwidth(8)
    fig.set_figheight(8)

    for p in range(num_planets):
        # could also try plotting log
        name = planet_names[p]
        
        if t_units == 'kyr':
            ax1.plot(times/1000, sim_data[0][name]["a"], label=name)
            ax2.plot(times/1000, sim_data[0][name]["e"], label=name)
            plt.xlabel("Time (kyr)")
    
        if p != num_planets-1:
            ax3.plot(times/1000, sim_data[0][name]["P_ratio"], label=f"{name}+{planet_names[p+1]}")
            
    ax1.set_ylabel("Semi-major Axis (AU)")
    ax2.set_ylabel("Eccentricity")
    ax3.set_ylabel("Period ratio")
    
    # ax1.set_ylim(0,0.45)
    # ax2.set_ylim(-0.1,0.45)
    ax3.set_ylim(1,2.2)
    
    # Plot ide location & width
    ax1.axhline(sim_data[1]["ide_position"], color='gray', ls='--', alpha=0.7)
    ax1.axhline(sim_data[1]["ide_position"] - sim_data[1]["ide_width"], color='gray', ls='--', alpha=0.1)
    ax1.axhline(sim_data[1]["ide_position"] + sim_data[1]["ide_width"], color='gray', ls='--', alpha=0.1)

    ax1.legend(); ax2.legend(); ax3.legend()
    ax1.grid(True); ax2.grid(True); ax3.grid(True)
    
    # Add horizontal lines for resonances
    resonances = [2, 3/2, 4/3, 5/3, 5/4]
    for r in resonances:
        ax3.axhline(r, color='gray', ls='--', alpha=0.7)

    fig.subplots_adjust(hspace=0)

    plt.suptitle("TRAPPIST-1 evolution")
    plt.tight_layout(); plt.show()

def simulate_trappist1(m_vals, r_vals, m_star, r_star, initial_P_ratios, Sigma_1au, K_factor, planet_names, years, file_path):
    '''
    Given initial parameters, returns the outcome of the simulation.
    '''
    # Create the simulation
    sim = rebound.Simulation()
    sim.units = ('AU', 'yr', 'Msun')
    sim.integrator = "whfast"

    # Add the star
    sim.add(m=m_star, r=r_star)

    num_planets = len(m_vals)
    
    # Define initial eccentricities (e)
    e_vals = np.zeros_like(m_vals)

    # Draw initial mean anomalies (M)
    M_vals = np.zeros_like(m_vals)

    # Initial semimajor axis of b
    a_b = 0.05

    # Define initial periods (P) and semimajor axes (a)
    P_vals = (a_b**3 / m_star)**(1/2)
    for i in range(num_planets-1):
        P_vals = np.append(P_vals, P_vals[i] * initial_P_ratios[i])
        
    a_vals = (P_vals**2 * m_star)**(1/3)

    print("Initial period ratios:")
    print(np.round(initial_P_ratios, decimals=4), end='\n\n')
    print("Initial period values (yr):")
    print(np.round(P_vals, decimals=4), end='\n\n')
    # print("\nInitial semimajor axis values (AU):")
    # print(np.round(a_vals, decimals=4))

    # Add planets 
    for i in range(num_planets):
        sim.add(m=m_vals[i], r=r_vals[i], a=a_vals[i], e=e_vals[i], M=M_vals[i])

    # Move to center of momentum
    sim.move_to_com()
    ps = sim.particles
    planets = ps[1:] # for easier indexing
    
    h = get_h(K_factor) # here, h = h_1au since there is no flaring
    print(f"h_1au: {float(h):.4g} \n")
    print(f"tau_a of b: {float(get_taus(a_b, m_vals, m_star, h, get_Sigma(Sigma_1au, a_b))[0][0]):.3e} \n")
    
    rebx = reboundx.Extras(sim)
    mig = rebx.load_force("type_I_migration")
    rebx.add_force(mig)

    # Keller et al. (2026) used 0.05 as the position and 0.01 as the width.
    # Huang & Ormel (2022) used positions r_c in [0.013 - 0.030 au] with width 
    # Delta = 2hr_c = 0.06 r_c. In particular, r_c = 0.023 worked best.
    mig.params["tIm_scale_height_1"] = h # = h_1au
    mig.params["tIm_surface_density_1"] = Sigma_1au
    mig.params["tIm_surface_density_exponent"] = 1.5 # alpha
    mig.params["tIm_flaring_index"] = 0 # beta

    mig.params["ide_position"] = 0.023 # inner disk edge
    mig.params["ide_width"] = 2*h*mig.params["ide_position"]
    
    data = integrate_sim(sim, num_planets, planets, planet_names, m_vals, m_star, years, start_time=0)

    all_data = concatenate_data((data))
    
    save_simulation_run(all_data, sim_id=0, file_path=file_path, sim_metadata={
                        "m_star": m_star, 
                        "num_planets": num_planets, 
                        "ide_position": mig.params["ide_position"],
                        "ide_width": mig.params["ide_width"]
                        })

    saved_sim = load_simulation_run(sim_id=0, file_path=file_path)
    
    return saved_sim