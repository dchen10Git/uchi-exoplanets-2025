import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

'''
Code to identify mean motion resonances in planetary systems.
Formulas used from section 2.3 in Keller et al. (2026).
Alternatively, we can use zeta as given in Eq. 11 in Fabrycky et al. (2014) to determine the period ratio.
'''

def find_best_twoBR_pq(m_star, b, c, p_max=7, crit="Delta"):
    '''Finds the best values for p and q at the end of simulation for 
       two planets given some MMR criterion'''
    if crit == 'Delta':
        best_Delta = 100
    elif crit == 'zeta': # Only 1st order formula implemented
        best_zeta = 100
        
    best_p, best_q = 100, 100
    
    for p in range(1, p_max+1):
        for q in range(1, p):
            # Use Kepler's law to find period values
            P_b = (b['a'].iloc[-1]**3 / m_star)**(1/2)
            P_c = (c['a'].iloc[-1]**3 / m_star)**(1/2)
            
            if crit == 'Delta':
                Delta = (P_c/P_b)/(p/q) - 1
                if np.abs(Delta) < np.abs(best_Delta):
                    best_Delta = Delta
                    best_p, best_q = p, q
            elif crit == 'zeta':
                zeta = 3*(1/(P_b/P_c-1) - round(1/(P_b/P_c-1)))
                if np.abs(zeta) < np.abs(best_zeta):
                    best_zeta = zeta
                    best_p, best_q = p, q
                    
    return best_p, best_q

def find_best_threeBR_pq(m_star, b, c, d, p_max=7, crit="Delta"):
    best_p_bc, best_q_bc = find_best_twoBR_pq(m_star, b, c, p_max, crit)
    best_p_cd, best_q_cd = find_best_twoBR_pq(m_star, c, d, p_max, crit)          
    return best_p_bc, best_q_bc, best_p_cd, best_q_cd

with open("fg_library.pkl", "rb") as f:
    fg_lib = pickle.load(f)
    
def twoBR_angle(b, c, p, q):
    f, g = fg_lib[(p, q)]
    pomega_hat = np.arctan2((f*b['e']*np.sin(b['pomega']) + g*c['e']*np.sin(c['pomega'])), (f*b['e']*np.cos(b['pomega']) + g*c['e']*np.cos(c['pomega'])))
    return q*b['l'] - p*c['l'] + (p-q)*pomega_hat

def threeBR_angle(b, c, d, p_bc, q_bc, p_cd, q_cd):
    return (p_cd-q_cd)*(q_bc*b['l'] - p_bc*c['l']) + (p_bc-q_bc)*(-q_cd*c['l'] + p_cd*d['l'])

def libration_amp(angles, N):
    '''Calculates libration amplitude of angles using last N samples'''
    mean_angle = np.average(angles[-N:])
    return np.sqrt(2/N * np.sum((angles[-N:]-mean_angle)**2))

def last_P(m_star, b):
    '''Returns the period value given m_star and planet dateframe'''
    return float(b['a'].iloc[-1]**3 / m_star)**(1/2)

def plot_libration(m_star, b, c, d=None, t_units='kyr', N=100):
    b_name = b.attrs["planet_name"]
    c_name = c.attrs["planet_name"]
    times = b['time']
    
    # 2BR angle
    if d is None:
        print(f"True period ratio: {last_P(m_star, c) / last_P(m_star, b):.4f}")

        # Calculate two-body resonant angle between b and c
        p, q = find_best_twoBR_pq(m_star, b, c)

        assert p != 100 # if not, then the planets are definitely not in resonance
        print(f"p, q: {p}, {q}")
        
        twoBR = np.rad2deg(twoBR_angle(b, c, p, q)) % 360 # mod 360 so it wraps
        amp = libration_amp(twoBR, N)
        print(f"Libration amplitude: {amp:.1f} deg")

        plt.figure(figsize=(8,4))
        if t_units == 'kyr':
            plt.scatter(times/1000, twoBR, s=3) 
        plt.axhline(180, color='gray', ls='--', alpha=0.7)
        plt.xlabel("Time (yrs)")
        plt.ylabel(f"Two-body resonant angle of {b_name} & {c_name} $\phi$ (degrees)")
        plt.ylim(0,360)
        plt.grid(True)
        plt.show()
    
    # 3BR angle
    else:
        d_name = d.attrs["planet_name"]
        print(f"True period ratios: b+c: {last_P(m_star, c) / last_P(m_star, b):.4f}, c+d: {last_P(m_star, d) / last_P(m_star, c):.4f}")
        
        # Calculate three-body resonant angle between b, c, and d
        p_bc, q_bc, p_cd, q_cd = find_best_threeBR_pq(m_star, b, c, d)
        
        assert p_bc != 100; assert p_cd != 100 # if not, then the planets are definitely not in resonance
        print(f"p_{b_name}{c_name}, q_{b_name}{c_name}, p_{c_name}{d_name}, q_{c_name}{d_name}: {p_bc}, {q_bc}, {p_cd}, {q_cd}")
        
        threeBR = np.rad2deg(threeBR_angle(b, c, d, p_bc, q_bc, p_cd, q_cd)) % 360 # mod 360 so it wraps
        amp = libration_amp(threeBR, N)
        print(f"Libration amplitude: {amp:.1f} deg")

        plt.figure(figsize=(8,4))
        if t_units == 'kyr':
            plt.scatter(times/1000, threeBR%360, s=3) 
        plt.axhline(180, color='gray', ls='--', alpha=0.7)
        plt.xlabel("Time (yrs)")
        plt.ylabel(f"Three-body resonant angle of {b_name}, {c_name}, & {d_name} $\phi$ (degrees)")
        plt.ylim(0,360)
        plt.grid(True)
        plt.show()

def check_resonance(m_star, b, c, d=None, N=100):
    '''
    Given m_star and planet dataframes, returns the resonance ratios if resonance is
    detected, otherwise returns (0,0) for 2BR and ((0,0), (0,0)) for 3BR.
    '''
    # 2BR
    if d is None:
        # Calculate two-body resonant angle between b and c
        p, q = find_best_twoBR_pq(m_star, b, c)

        if p == 100: # if not, then the planets are definitely not in resonance
            return (0,0)
        
        else: # resonance detected
            twoBR = np.rad2deg(twoBR_angle(b, c, p, q)) % 360 # mod 360 so it wraps
            amp = libration_amp(twoBR, N)
            if amp < 90: # criterion for libration
                return (p,q)
            else:
                return (0,0)
    
    # 3BR
    else:
        # Calculate three-body resonant angle between b, c, and d
        p_bc, q_bc, p_cd, q_cd = find_best_threeBR_pq(m_star, b, c, d)
        
        if p_bc == 100 or p_cd == 100: # if not, then the planets are definitely not in resonance
            return ((0,0), (0,0))
        
        else: # resonance detected
            threeBR = np.rad2deg(threeBR_angle(b, c, d, p_bc, q_bc, p_cd, q_cd)) % 360 # mod 360 so it wraps
            amp = libration_amp(threeBR, N)
            if amp < 90:
                return ((p_bc, q_bc), (p_cd, q_cd))
            else:
                return ((0,0), (0,0))

def res_chain_order(saved_sim, N=100):
    '''
    Given m_star and list of planet dataframes and planet_names in order, returns the highest order
    of the RC if the chain is complete, 0 otherwise.
    '''
    sim_data = saved_sim[0]
    m_star = saved_sim[1]['m_star']
    planet_names = saved_sim[1]['planet_names']
    planets = {p: sim_data[p] for p in planet_names}
    
    highest_order = 0
    for i in range(len(planets)-1):
        b_name = planet_names[i]
        c_name = planet_names[i+1]
        res = check_resonance(m_star, planets[b_name], planets[c_name], N=N)
        order = res[0] - res[1]
        
        if order > highest_order:
            highest_order = order
        
        if order == 0: # RC is not complete
            return 0
            
    return highest_order

def res_chain_score(saved_sim, N=100):
    '''
    Given saved_sim, returns score based on outcome:
    
        -1: incomplete simulation
        
        0: partial chain
        
        1: 2BRC with all 1st order pairs
        
        2: 2BRC with at least one 2nd order but no 3rd order
        
        3: 2BRC with at least one 3rd order
        
        1x: Complete 3BRC
    '''
    sim_data = saved_sim[0]
    m_star = saved_sim[1]['m_star']
    planet_names = saved_sim[1]['planet_names']
    planets = {p: sim_data[p] for p in planet_names}        
    
    score = 0
    
    # Check existence of a full three-body chain
    threeBRC = True
    for i in range(len(planet_names)-2):
        b_name = planet_names[i]
        c_name = planet_names[i+1]
        d_name = planet_names[i+2]
        res = check_resonance(m_star, planets[b_name], planets[c_name], planets[d_name], N=N)
        if res == ((0,0), (0,0)):
            threeBRC = False
            break
            
    if threeBRC:
        score += 10
    
    # Add order of 2BRC to score
    score += res_chain_order(saved_sim, N)

    return score
    