#!/usr/bin/env python
# coding: utf-8

# ==============================================================================
# ATLAS: Advanced Thermodynamic Liquid & Aqueous Solver
# Copyright (C) 2026 Petteri Vainikka, PhD 
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================

__version__ = "1.0.1"

import os
import sys
import time
import random

# Start the execution timer
start_time = time.time()

# ---------------------------------------------------------
# THREAD LIMITER
# ---------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"

import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import seaborn as sns
import matplotlib

# ---------------------------------------------------------
# ATLAS Physical & Numerical Constants
# ---------------------------------------------------------
RELAX_T = 0.15        # Temperature relaxation factor
RELAX_X = 0.3         # Mole fraction relaxation factor
T_MIN = 180.0         # Minimum allowed temperature (miscibility gap mask)
T_MAX = 600.0         # Maximum allowed temperature ceiling
X_INF = 1e-7          # Infinite dilution mole fraction
MAX_UNITY_ITER = 50   # Max iterations for bit-nudging unity loop
R = 8.3145            # Thermodynamic Gas Constant [J / (mol K)]

# ---------------------------------------------------------
# Plotting Settings
# ---------------------------------------------------------
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['text.usetex'] = True
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
cmap = sns.color_palette("husl", 7)

# ---------------------------------------------------------
# COSMO-RS Setup & Hotfixes
# ---------------------------------------------------------
from opencosmorspy import COSMORS
import opencosmorspy.molecules as mols

# HOTFIX: opencosmorspy's internal dictionary forgets that Halogens exist. 
def _patched_convert_element_symbols(atm_elmnt):
    periodic = {'H':1, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'SI':14, 'P':15, 'S':16, 'CL':17, 'BR':35, 'I':53}
    converted = []
    for el in atm_elmnt:
        val = el.decode('utf-8') if isinstance(el, bytes) else str(el)
        val = val.strip().upper()
        if val in periodic: 
            converted.append(periodic[val])
        else: 
            raise ValueError(f"ATLAS Intercept: Truly unknown element '{val}'")
    return np.array(converted, dtype=int)

mols._convert_element_symbols = _patched_convert_element_symbols

crs = COSMORS(par='default_orca')
crs.par.calculate_contact_statistics_molecule_properties = True

# ---------------------------------------------------------
# The ATLAS Quote Vault
# ---------------------------------------------------------
ATLAS_QUOTES = [
    ("Contradictions do not exist. Whenever you think that you are facing a contradiction, check your premises. You will find that one of them is wrong.", "Francisco d'Anconia"),
    ("I swear by my life and my love of it that I will never live for the sake of another man, nor ask another man to live for mine.", "John Galt"),
    ("There are no evil thoughts except one: the refusal to think.", "Dagny Taggart"),
    ("Money is the barometer of a society's virtue.", "Francisco d'Anconia"),
    ("A rational process is a moral process.", "John Galt"),
    ("I started my life with a single absolute: that the world was mine to shape in the image of my highest values and never to be given up to a lesser standard.", "John Galt"),
    ("Do not let your fire go out, spark by irreplaceable spark in the hopeless swamps of the not-quite, the not-yet, and the not-at-all.", "John Galt"),
    ("It is not advisable, James, to venture unsolicited opinions. You should spare yourself the embarrassing discovery of their exact value to your listener.", "Francisco d'Anconia"),
    ("Man? What is man? He's just a collection of chemicals with delusions of grandeur.", "Dr. Simon Pritchett"),
    ("Don't ever get angry at a man for stating the truth.", "Dagny Taggart"),
    ("You see, Dr. Stadler, people don't want to think. And the deeper they get into trouble, the less they want to think.", "Dr. Floyd Ferris"),
    ("Miss Taggart, do you know the hallmark of the second-rater? It's resentment of another man's achievement. Those touchy mediocrities who sit trembling lest someone's work prove greater than their own—they have no inkling of the loneliness that comes when you reach the top.", "Dr. Robert Stadler"),
    ("The worst guilt is to accept an undeserved guilt.", " Francisco d'Anconia"),
    ("If you saw Atlas, the giant who holds the world on his shoulders, if you saw that he stood, blood running down his chest, his knees buckling, his arms trembling but still trying to hold the world aloft with the last of his strength, and the greater his effort the heavier the world bore down upon his shoulders-what would you tell him to do?\nI ... don't know. What ... could he do? What would you tell him?\nTo shrug.", "Francisco d'Anconia to Hank Rearden"),
    ("The nation which had once held the creed that greatness is achieved by production, is now told that it is achieved by squalor.", "Francisco d'Anconia"),
    ("Guilt is a rope that wears thin.", "James Taggart to Lillian Rearden"),
    ("To arrive at a contradiction is to confess an error in one's thinking; to maintain a contradiction is to abdicate one's mind and to evict oneself from the realm of reality.", "John Galt"),
    ("Devotion to truth is the hallmark of morality; there is no greater, nobler, more heroic form of devotion than the act of a man who assumes the responsibility of thinking.", "John Galt"),
    ("The man who lets a leader prescribe his course is a wreck being towed to the scrap heap.", "John Galt"),
    ("When I disagree with a rational man, I let reality be our final arbiter; if I am right, he will learn; if I am wrong, I will; one of us will win, but both will profit.", "John Galt"),
    ("Force and mind are opposites; morality ends where a gun begins.", "John Galt"),
    ("Achieving life is not the equivalent of avoiding death.", "John Galt"),
    ("Power-lust is a weed that grows only in the vacant lots of an abandoned mind.", "John Galt"),
    ("There are two sides to every issue: one side is right and the other is wrong, but the middle is always evil.", "John Galt"),
    ("Live and act within the limit of your knowledge and keep expanding it to the limit of your life.", "John Galt"),
    ("Never think of pain or danger or enemies a moment longer than is necessary to fight them.","John Galt"),
    ("Get the hell out of my way!", "John Galt")
]

# ---------------------------------------------------------
# Command Line Interface (CLI) Setup
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="ATLAS: Calculate SLE phase diagrams, Solubility, logP, and Extraction using COSMO-RS.")
parser.add_argument('--hba', type=str, 
                    help="Name of the HBA / Solute molecule (e.g., choline_chloride)")
parser.add_argument('--hbd', nargs='+', type=str, 
                    help="List of HBD / Solvent molecules separated by spaces")
parser.add_argument('--status', action='store_true', 
                    help="Check database and verify if .orcacosmo files exist")
parser.add_argument('--tern', action='store_true', 
                    help="Evaluate a Ternary system (Requires exactly 1 Solute and 2 Solvents)")
parser.add_argument('--sol', action='store_true', 
                    help="Calculate Solute solubility in pure or mixed solvent")
parser.add_argument('--logp', action='store_true', 
                    help="Calculate logP and transfer dG between 2 pure solvents")
parser.add_argument('--extract', action='store_true', 
                    help="Calculate extraction logP from Aqueous Phase -> Mixed DES (Requires Water + 2 DES components)")
parser.add_argument('--ratio', nargs=2, type=float, default=[1.0, 2.0], 
                    help="Molar ratio of the two DES components for --extract (default: 1.0 2.0)")
parser.add_argument('--temp', type=float, default=298.15, 
                    help="System temperature in Kelvin (default: 298.15)")
parser.add_argument('--tol', type=float, default=0.1, 
                    help="Convergence tolerance in Kelvin for SLE curves (default: 0.1)")
parser.add_argument('--silent', action='store_true', 
                    help="Suppress the final execution time and quote printout")
parser.add_argument('--csv', type=str, metavar='FILE',
                    help="Save numerical output to the specified CSV file (appends if file exists)")
parser.add_argument('--orcacosmo_dir', type=str, 
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'molecules'), 
                    help="Path to directory containing .orcacosmo files (Default: ./molecules)")
parser.add_argument('--db_file', type=str, 
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database', 'thermo_db.dat'), 
                    help="Path to thermodynamic database CSV (Default: ./database/thermo_db.dat)")

args = parser.parse_args()

molpath = args.orcacosmo_dir  # <--- Updated mapping
db_path = args.db_file
fileext = '_c000.orcacosmo'

# ---------------------------------------------------------
# CSV Helper Function
# ---------------------------------------------------------
def write_csv(filepath, header, row_data_list):
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        for row in row_data_list:
            writer.writerow(row)

# ---------------------------------------------------------
# Database Manager
# ---------------------------------------------------------
def load_thermo_database(filepath):
    db = {}
    try:
        with open(filepath, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                db[row['name'].strip()] = {
                    'T_m': float(row['T_m']),
                    'dH_fus': float(row['dH_fus'])
                }
    except FileNotFoundError:
        print(f"\nCRITICAL ERROR: Database file not found at {filepath}")
        print("Please check your --db_file path.")
        sys.exit(1)
    return db

def get_molecule_info(filepath, db):
    filename = os.path.basename(filepath)
    clean_name = filename.replace('_c000.orcacosmo', '').replace('.orcacosmo', '')
    display_name = clean_name.replace('_', ' ').title()
    
    if clean_name in db:
        return clean_name, display_name, db[clean_name]['T_m'], db[clean_name]['dH_fus']
    else:
        return clean_name, display_name, None, None

thermo_db = load_thermo_database(db_path)

# ---------------------------------------------------------
# ROUTE 0: SYSTEM STATUS CHECKER
# ---------------------------------------------------------
if args.status:
    print(f"\n{'='*85}")
    print(f"{'Molecule Name':<30} | {'T_m (K)':<10} | {'dH_fus (J/mol)':<15} | {'Orcacosmo File'}")
    print(f"{'='*85}")
    for name, props in sorted(thermo_db.items()):
        expected_file = os.path.join(molpath, f"{name}{fileext}")
        display_name = name.title().replace('_', ' ')
        file_status = "\033[92m[FOUND]\033[0m" if os.path.exists(expected_file) else "\033[91m[MISSING]\033[0m"
        print(f"{display_name:<30} | {props['T_m']:<10.2f} | {props['dH_fus']:<15.2f} | {file_status}")
    print(f"{'='*85}\n")
    sys.exit(0)

if not args.hba or not args.hbd:
    print("\nERROR: You must provide both --hba and --hbd to run a calculation.")
    sys.exit(1)

hba_file = os.path.join(molpath, args.hba + fileext)
hbd_files = [os.path.join(molpath, hbd + fileext) for hbd in args.hbd]

# ---------------------------------------------------------
# HELPER FUNCTION: ENSURE UNITY FOR NON-PYTHON BACKEND
# ---------------------------------------------------------
def enforce_unity(x_raw):
    """
    Forces a numpy array to sum to exactly 1.0, bypassing strict 1e-16 tolerance checks.
    Uses 'Bit-Nudging' to shift the final float by the smallest possible binary step.
    """
    clean_x = np.array(x_raw, dtype=np.float64)
    clean_x /= np.sum(clean_x)
    clean_x[-1] = 1.0 - np.sum(clean_x[:-1])
    
    iterations = 0
    while np.abs(1.0 - clean_x.sum()) > 1e-16 and iterations < MAX_UNITY_ITER:
        if clean_x.sum() < 1.0:
            clean_x[-1] = np.nextafter(clean_x[-1], 2.0)
        else:
            clean_x[-1] = np.nextafter(clean_x[-1], -2.0)
        iterations += 1
            
    return clean_x

# ---------------------------------------------------------
# ENGINE 1: Binary SLE
# ---------------------------------------------------------
def get_ideal_curve(tm_a, hf_a, tm_b, hf_b):
    x_a = np.linspace(0.01, 0.99, 99)
    x_b = 1.0 - x_a
    T_ideal_a = 1 / (1/tm_a - (R/hf_a)*np.log(x_a))
    T_ideal_b = 1 / (1/tm_b - (R/hf_b)*np.log(x_b))
    T_ideal = np.maximum(T_ideal_a, T_ideal_b)
    eutectic_idx = np.argmin(T_ideal)
    return x_a, T_ideal, x_a[eutectic_idx]

def solve_real_curve(hba_path, hbd_path, tm_hba, hf_hba, tm_hbd, hf_hbd, hba_disp, hbd_disp, max_iter=40, tol=0.1):
    x_hba = np.linspace(0.05, 0.95, 20) 
    x_hbd = 1.0 - x_hba
    print(f"\n--- Starting Iterations for {hba_disp} + {hbd_disp} ---")
    _, T_guess, _ = get_ideal_curve(tm_hba, hf_hba, tm_hbd, hf_hbd)
    T_guess = np.interp(x_hba, np.linspace(0.01, 0.99, 99), T_guess)
    
    for iteration in range(max_iter):
        crs.clear_jobs()
        crs.clear_molecules()
        crs.add_molecule([hba_path])
        crs.add_molecule([hbd_path])
        
        for i in range(len(x_hba)):
            clean_x = enforce_unity([x_hba[i], x_hbd[i]])
            crs.add_job(x=clean_x, T=T_guess[i], refst='pure_component')
            
        results = crs.calculate()
        lng_hba = results['tot']['lng'][:, 0]
        lng_hbd = results['tot']['lng'][:, 1]
        
        T_new_hba = 1 / (1/tm_hba - (R/hf_hba)*(np.log(x_hba) + lng_hba))
        T_new_hbd = 1 / (1/tm_hbd - (R/hf_hbd)*(np.log(x_hbd) + lng_hbd))
        T_raw = np.maximum(T_new_hba, T_new_hbd)
        
        T_safe = np.clip((RELAX_T * T_raw) + ((1.0 - RELAX_T) * T_guess), T_MIN, T_MAX)
        max_diff = np.max(np.abs(T_safe - T_guess))
        
        eutectic_print = f"{np.nanmin(T_safe):>6.2f}" if not np.all(np.isnan(T_safe)) else "   NaN"
        print(f"  Cycle {iteration + 1:>2}/{max_iter} | Max ΔT: {max_diff:>7.4f} K | Est. Eutectic: {eutectic_print} K")
        
        if max_diff < tol:
            print(f"  -> SUCCESS: Converged!\n")
            T_safe[T_safe <= (T_MIN + 0.1)] = np.nan 
            return x_hba, T_safe
        T_guess = T_safe 
        
    print(f"  -> WARNING: Hit max iterations. Final Max ΔT: {max_diff:.4f} K\n")
    T_guess[T_guess <= (T_MIN + 0.1)] = np.nan 
    return x_hba, T_guess

# ---------------------------------------------------------
# ENGINE 2: Ternary SLE
# ---------------------------------------------------------
def get_ternary_grid(steps=15, eps=0.01):
    grid = []
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            k = steps - i - j
            x_a, x_b, x_c = max(i/steps, eps), max(j/steps, eps), max(k/steps, eps)
            tot = x_a + x_b + x_c
            grid.append([x_a/tot, x_b/tot, (1.0 - x_a/tot - x_b/tot)])
    return np.array(grid)

def solve_ternary_curve(mol_paths, thermo_props, disp_names, max_iter=40, tol=0.1):
    grid = get_ternary_grid(steps=15) 
    x_a, x_b, x_c = grid[:, 0], grid[:, 1], grid[:, 2]
    print(f"\n--- Starting Ternary Iterations for {disp_names[0]} + {disp_names[1]} + {disp_names[2]} ---")
    
    T_id_a = 1 / (1/thermo_props[0][0] - (R/thermo_props[0][1])*np.log(x_a))
    T_id_b = 1 / (1/thermo_props[1][0] - (R/thermo_props[1][1])*np.log(x_b))
    T_id_c = 1 / (1/thermo_props[2][0] - (R/thermo_props[2][1])*np.log(x_c))
    T_guess = np.maximum.reduce([T_id_a, T_id_b, T_id_c])
    
    for iteration in range(max_iter):
        crs.clear_jobs()
        crs.clear_molecules()
        for path in mol_paths: crs.add_molecule([path])

        for i in range(len(grid)):
            clean_x = enforce_unity([x_a[i], x_b[i], x_c[i]])
            crs.add_job(x=clean_x, T=T_guess[i], refst='pure_component')

        results = crs.calculate()
        lng_a, lng_b, lng_c = results['tot']['lng'][:, 0], results['tot']['lng'][:, 1], results['tot']['lng'][:, 2]
        
        T_new_a = 1 / (1/thermo_props[0][0] - (R/thermo_props[0][1])*(np.log(x_a) + lng_a))
        T_new_b = 1 / (1/thermo_props[1][0] - (R/thermo_props[1][1])*(np.log(x_b) + lng_b))
        T_new_c = 1 / (1/thermo_props[2][0] - (R/thermo_props[2][1])*(np.log(x_c) + lng_c))
        
        T_raw = np.maximum.reduce([T_new_a, T_new_b, T_new_c])
        T_safe = np.clip((RELAX_T * T_raw) + ((1.0 - RELAX_T) * T_guess), T_MIN, T_MAX) 
        
        max_diff = np.max(np.abs(T_safe - T_guess))
        eutectic_print = f"{np.nanmin(T_safe):>6.2f}" if not np.all(np.isnan(T_safe)) else "   NaN"
        print(f"  Cycle {iteration + 1:>2}/{max_iter} | Max ΔT: {max_diff:>7.4f} K | Est. Eutectic: {eutectic_print} K")
        
        if max_diff < tol:
            print(f"  -> SUCCESS: Ternary system converged!\n")
            T_safe[T_safe <= (T_MIN + 0.1)] = np.nan 
            return grid, T_safe
        T_guess = T_safe
        
    print(f"  -> WARNING: Hit max iterations. Final Max ΔT: {max_diff:.4f} K\n")
    T_guess[T_guess <= (T_MIN + 0.1)] = np.nan 
    return grid, T_guess

# ---------------------------------------------------------
# ENGINE 3: Pure & Mixed-Solvent Solubility
# ---------------------------------------------------------
def solve_pure_solubility(solute_path, solv_path, tm_s, hf_s, disp_names, T_sys, max_iter=40, tol=1e-5):
    print(f"\n--- Calculating Pure Solubility of {disp_names[0]} in {disp_names[1]} at {T_sys} K ---")
    k_ideal = (hf_s / R) * (1.0 / tm_s - 1.0 / T_sys)
    ideal_solubility = np.exp(k_ideal)
    
    if ideal_solubility >= 1.0:
        print(f"  -> WARNING: System Temp ({T_sys} K) exceeds Solute Melting Point ({tm_s} K).")
        print(f"  -> Solute is physically a liquid. Capping theoretical solubility at x = 0.9999.")
        ideal_solubility = 0.9999
    else:
        print(f"  -> Ideal Solubility limit: {ideal_solubility:.4e} mol fraction")
    
    x_s = ideal_solubility 
    for iteration in range(max_iter):
        x_solv = 1.0 - x_s
        clean_x = enforce_unity([x_s, x_solv])
        
        crs.clear_jobs()
        crs.clear_molecules()
        for path in [solute_path, solv_path]: crs.add_molecule([path])
            
        crs.add_job(x=clean_x, T=T_sys, refst='pure_component')
        results = crs.calculate()
        
        lng_s = results['tot']['lng'][0, 0] 
        x_s_new = np.exp(k_ideal - lng_s)
        x_s_new = min((RELAX_X * x_s_new) + ((1.0 - RELAX_X) * x_s), 0.9999) 
        
        if np.abs(x_s_new - x_s) < tol: 
            print(f"  -> SUCCESS: Converged in {iteration + 1} cycles!")
            print(f"  -> Final Solubility (x_s): {x_s_new:.4e}\n")
            return x_s_new
        x_s = x_s_new
        
    print(f"  -> WARNING: Hit max iterations. Final Solubility: {x_s:.4e}\n")
    return x_s

def solve_solubility_curve(solute_path, solv_a_path, solv_b_path, tm_s, hf_s, disp_names, T_sys, max_iter=40, tol=1e-5):
    solv_a_fracs = np.linspace(0.0, 1.0, 21) 
    solubilities = []
    
    print(f"\n--- Calculating Solubility of {disp_names[0]} in {disp_names[1]} + {disp_names[2]} at {T_sys} K ---")
    k_ideal = (hf_s / R) * (1.0 / tm_s - 1.0 / T_sys)
    ideal_solubility = np.exp(k_ideal)
    
    if ideal_solubility >= 1.0:
        print(f"  -> WARNING: System Temp ({T_sys} K) exceeds Solute Melting Point ({tm_s} K).")
        print(f"  -> Solute is physically a liquid. Capping theoretical solubility at x = 0.9999.")
        ideal_solubility = 0.9999
    else:
        print(f"  -> Ideal Solubility limit: {ideal_solubility:.4e} mol fraction")
    
    for frac_A in solv_a_fracs:
        frac_B = 1.0 - frac_A
        x_s = ideal_solubility 
        for iteration in range(max_iter):
            x_a = frac_A * (1.0 - x_s)
            x_b = frac_B * (1.0 - x_s)
            clean_x = enforce_unity([x_s, x_a, x_b])
            
            crs.clear_jobs()
            crs.clear_molecules()
            for path in [solute_path, solv_a_path, solv_b_path]: crs.add_molecule([path])
                
            crs.add_job(x=clean_x, T=T_sys, refst='pure_component')
            results = crs.calculate()
            
            lng_s = results['tot']['lng'][0, 0]
            x_s_new = np.exp(k_ideal - lng_s)
            x_s_new = min((RELAX_X * x_s_new) + ((1.0 - RELAX_X) * x_s), 0.9999) 
            
            if np.abs(x_s_new - x_s) < tol: break
            x_s = x_s_new
            
        solubilities.append(x_s)
        print(f"  Solvent Ratio [{frac_A:.2f} A / {frac_B:.2f} B] -> Solubility (x_s): {x_s:.4e}")
        
    return solv_a_fracs, np.array(solubilities), ideal_solubility

# ---------------------------------------------------------
# ENGINE 4: Partition Coefficient (logP) & Transfer dG
# ---------------------------------------------------------
def solve_logp_and_dg(solute_path, solv_a_path, solv_b_path, disp_names, T_sys):
    print(f"\n--- Calculating logP and dG ({disp_names[1]} -> {disp_names[2]}) at {T_sys} K ---")
    
    crs.clear_jobs()
    crs.clear_molecules()
    crs.add_molecule([solute_path])
    crs.add_molecule([solv_a_path])
    x_inf_a = enforce_unity([X_INF, 1.0])
    crs.add_job(x=x_inf_a, T=T_sys, refst='pure_component')
    res_a = crs.calculate()
    lng_a = res_a['tot']['lng'][0, 0] 
    
    crs.clear_jobs()
    crs.clear_molecules()
    crs.add_molecule([solute_path])
    crs.add_molecule([solv_b_path])
    x_inf_b = enforce_unity([X_INF, 1.0])
    crs.add_job(x=x_inf_b, T=T_sys, refst='pure_component')
    res_b = crs.calculate()
    lng_b = res_b['tot']['lng'][0, 0] 

    dG_transfer = (R * T_sys * (lng_b - lng_a)) / 1000.0 
    logP_x = (lng_a - lng_b) / np.log(10)
    
    print(f"  -> Solute ln(gamma) in {disp_names[1]}: {lng_a:.4f}")
    print(f"  -> Solute ln(gamma) in {disp_names[2]}: {lng_b:.4f}")
    print(f"  -> ΔG ({disp_names[1]} -> {disp_names[2]}): {dG_transfer:.2f} kJ/mol")
    print(f"  -> logP_x (Mole Fraction Base): {logP_x:.2f}\n")
    return dG_transfer, logP_x

# ---------------------------------------------------------
# ENGINE 5: Extraction to Mixed DES (LLE)
# ---------------------------------------------------------
def solve_extraction(solute_path, water_path, des_a_path, des_b_path, disp_names, T_sys, des_ratio):
    print(f"\n--- Extraction: {disp_names[0]} from {disp_names[1]} -> DES ({disp_names[2]} + {disp_names[3]}) at {T_sys} K ---")
    
    crs.clear_jobs()
    crs.clear_molecules()
    crs.add_molecule([solute_path])
    crs.add_molecule([water_path])
    x_water = enforce_unity([X_INF, 1.0])
    crs.add_job(x=x_water, T=T_sys, refst='pure_component')
    res_w = crs.calculate()
    lng_w = res_w['tot']['lng'][0, 0]

    crs.clear_jobs()
    crs.clear_molecules()
    crs.add_molecule([solute_path])
    crs.add_molecule([des_a_path])
    crs.add_molecule([des_b_path])
    
    r1, r2 = des_ratio
    tot_ratio = r1 + r2
    x_des_a = (r1 / tot_ratio)
    x_des_b = (r2 / tot_ratio)
    
    x_des = enforce_unity([X_INF, x_des_a, x_des_b])
    crs.add_job(x=x_des, T=T_sys, refst='pure_component')
    res_des = crs.calculate()
    lng_des = res_des['tot']['lng'][0, 0]

    dG_transfer = (R * T_sys * (lng_des - lng_w)) / 1000.0 
    logP_x = (lng_w - lng_des) / np.log(10)
    
    print(f"  -> Solute ln(gamma) in {disp_names[1]}: {lng_w:.4f}")
    print(f"  -> Solute ln(gamma) in DES [{r1}:{r2}]: {lng_des:.4f}")
    print(f"  -> ΔG (Transfer to DES): {dG_transfer:.2f} kJ/mol")
    print(f"  -> Extraction logP_x:    {logP_x:.2f}\n")
    return dG_transfer, logP_x

# ---------------------------------------------------------
# EXECUTION ROUTER
# ---------------------------------------------------------
print("Loading Thermodynamic Properties...")
hba_key, hba_disp, tm_hba, hf_hba = get_molecule_info(hba_file, thermo_db)

if not args.logp and not args.extract:
    if tm_hba is None or not os.path.exists(hba_file):
        print(f"\nCRITICAL: Solute '{hba_key}' missing from database or ORCA folder. Cannot proceed with SLE.")
        sys.exit(1)
else:
    if not os.path.exists(hba_file):
        print(f"\nCRITICAL: Solute '{hba_key}' ORCA file missing.")
        sys.exit(1)

# ROUTE 1: TERNARY EXECUTION
if args.tern:
    if len(args.hbd) != 2:
        print("\nERROR: Ternary evaluation requires exactly 2 HBDs.")
        sys.exit(1)
        
    mol_paths = [hba_file, hbd_files[0], hbd_files[1]]
    thermo_props = [(tm_hba, hf_hba)]
    disp_names = [hba_disp]
    
    for path in [hbd_files[0], hbd_files[1]]:
        key, disp, tm, hf = get_molecule_info(path, thermo_db)
        if tm is None or not os.path.exists(path):
            print(f"\nCRITICAL: Missing database entry or ORCA file for {disp}. Aborting ternary.")
            sys.exit(1)
        thermo_props.append((tm, hf))
        disp_names.append(disp)
        
    grid, T_real = solve_ternary_curve(mol_paths, thermo_props, disp_names, tol=args.tol)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.style.use('seaborn-v0_8-white')
    X = grid[:, 1] + 0.5 * grid[:, 2]
    Y = grid[:, 2] * (np.sqrt(3.0) / 2.0)
    
    triang = tri.Triangulation(X, Y)
    mask = np.any(np.isnan(T_real[triang.triangles]), axis=1)
    triang.set_mask(mask)
    
    T_min = np.nanmin(T_real)
    T_max_plot = np.nanpercentile(T_real, 90)
    levels = np.linspace(T_min, T_max_plot, 50)
    
    contour = ax.tricontourf(triang, T_real, levels=levels, cmap='magma', extend='max')
    cbar = fig.colorbar(contour, ax=ax, label='Liquidus Temperature (K)', pad=0.12)
    
    tick_vals = [0.2, 0.4, 0.6, 0.8]
    for frac in np.linspace(0.1, 0.9, 9):
        ax.plot([1-frac, 0.5*(1-frac)], [0, (1-frac)*np.sqrt(3)/2], color='white', alpha=0.25, lw=0.75, ls='--')
        ax.plot([frac, frac + 0.5*(1-frac)], [0, (1-frac)*np.sqrt(3)/2], color='white', alpha=0.25, lw=0.75, ls='--')
        ax.plot([0.5*frac, (1-frac) + 0.5*frac], [frac*np.sqrt(3)/2, frac*np.sqrt(3)/2], color='white', alpha=0.25, lw=0.75, ls='--')

    for val in tick_vals:
        ax.text(val, -0.015, f"{val}", ha='center', va='top', fontsize=7, color='gray')
        ax.text(1 - 0.5*val + 0.015, val*np.sqrt(3)/2 + 0.01, f"{val}", ha='left', va='bottom', fontsize=7, color='gray')
        ax.text(0.5*(1-val) - 0.015, (1-val)*np.sqrt(3)/2 + 0.01, f"{val}", ha='right', va='bottom', fontsize=7, color='gray')

    ax.text(0.0, -0.06, disp_names[0], ha='center', va='top', fontsize=11, weight='bold')
    ax.text(1.0, -0.06, disp_names[1], ha='center', va='top', fontsize=11, weight='bold')
    ax.text(0.5, np.sqrt(3)/2 + 0.03, disp_names[2], ha='center', va='bottom', fontsize=11, weight='bold')
    
    ax.axis('off')
    plt.title(f"Ternary Phase Diagram: {disp_names[0]} System", pad=35)
    plt.tight_layout()
    safe_fn = f'{disp_names[0].replace(" ", "_")}_Ternary.png'
    plt.savefig(safe_fn, bbox_inches='tight', dpi=500)
    print(f"\nTernary Diagram saved as {safe_fn}!")

# ROUTE 2: SOLUBILITY (Pure or Mixed)
elif args.sol:
    if len(args.hbd) == 1:
        _, disp_a, _, _ = get_molecule_info(hbd_files[0], thermo_db)
        disp_names = [hba_disp, disp_a]
        final_sol = solve_pure_solubility(hba_file, hbd_files[0], tm_hba, hf_hba, disp_names, T_sys=args.temp, tol=args.tol)
        if args.csv:
            write_csv(args.csv, ['System', 'T_sys_K', 'Solubility_xS'], [[f"{hba_disp} in {disp_a}", args.temp, final_sol]])
            print(f"\nData appended to {args.csv}")

    elif len(args.hbd) == 2:
        _, disp_a, _, _ = get_molecule_info(hbd_files[0], thermo_db)
        _, disp_b, _, _ = get_molecule_info(hbd_files[1], thermo_db)
        disp_names = [hba_disp, disp_a, disp_b]
        x_solvent, y_solubility, ideal_sol = solve_solubility_curve(hba_file, hbd_files[0], hbd_files[1], tm_hba, hf_hba, disp_names, T_sys=args.temp, tol=args.tol)
        
        if args.csv:
            header = ['System', 'T_sys_K', 'x_' + disp_a, 'x_' + disp_b, 'Solubility_xS']
            rows = [[f"{hba_disp} in {disp_a}+{disp_b}", args.temp, x_solvent[i], 1.0 - x_solvent[i], y_solubility[i]] for i in range(len(x_solvent))]
            write_csv(args.csv, header, rows)
            print(f"\nData appended to {args.csv}")

        fig, ax = plt.subplots(figsize=(6, 4))
        plt.style.use('seaborn-v0_8-white')
        ax.axhline(ideal_sol, color='k', linestyle='-.', alpha=0.5, label='Ideal Solubility')
        ax.plot(x_solvent, y_solubility, color=cmap[0], lw=2.5, marker='o', label=f'Real Solubility')
        
        ax.set_xlim(0, 1.0)
        ax.set_yscale('log')
        ax.set_xlabel(f'Mole Fraction of {disp_a} in Solvent Mixture (Free of Solute)')
        ax.set_ylabel(f'Solubility of {hba_disp} ($x_S$)')
        ax.set_title(f"Solubility in {disp_a} + {disp_b} at {args.temp} K")
        
        ax.grid(alpha=0.4, which='both')
        ax.legend(frameon=True, fancybox=True, shadow=True)
        
        safe_fn = f'{hba_disp.replace(" ", "_")}_Solubility.png'
        plt.tight_layout()
        plt.savefig(safe_fn, bbox_inches='tight', dpi=500)
        print(f"\nSolubility curve saved as {safe_fn}!")
        
    else:
        print("\nERROR: Solubility evaluation requires exactly 1 or 2 Solvents.")
        sys.exit(1)

# ROUTE 3: PARTITION COEFFICIENTS (logP & dG)
elif args.logp:
    if len(args.hbd) != 2:
        print("\nERROR: logP evaluation requires exactly 2 Solvents to transfer between.")
        sys.exit(1)
        
    _, disp_a, _, _ = get_molecule_info(hbd_files[0], thermo_db)
    _, disp_b, _, _ = get_molecule_info(hbd_files[1], thermo_db)
    disp_names = [hba_disp, disp_a, disp_b]
    dG_transfer, logP_x = solve_logp_and_dg(hba_file, hbd_files[0], hbd_files[1], disp_names, T_sys=args.temp)
    
    if args.csv:
        header = ['Solute', 'System', 'T_sys_K', 'dG_transfer_kJ_mol', 'logP_x']
        system_name = f"{disp_names[1]} -> {disp_names[2]}"
        write_csv(args.csv, header, [[hba_disp, system_name, args.temp, dG_transfer, logP_x]])
        print(f"\nData appended to {args.csv}")

# ROUTE 4: EXTRACTION TO DES
elif args.extract:
    if len(args.hbd) != 3:
        print("\nERROR: DES Extraction requires exactly 3 Solvents (e.g., h2o des_a des_b).")
        sys.exit(1)
        
    _, disp_w, _, _ = get_molecule_info(hbd_files[0], thermo_db)
    _, disp_des_a, _, _ = get_molecule_info(hbd_files[1], thermo_db)
    _, disp_des_b, _, _ = get_molecule_info(hbd_files[2], thermo_db)
    disp_names = [hba_disp, disp_w, disp_des_a, disp_des_b]
    dG_transfer, logP_x = solve_extraction(hba_file, hbd_files[0], hbd_files[1], hbd_files[2], disp_names, T_sys=args.temp, des_ratio=args.ratio)
    
    if args.csv:
        header = ['Solute', 'System', 'T_sys_K', 'DES_Ratio', 'dG_transfer_kJ_mol', 'logP_x']
        system_name = f"{disp_names[1]} -> DES[{disp_names[2]}:{disp_names[3]}]"
        ratio_str = f"{args.ratio[0]}:{args.ratio[1]}"
        write_csv(args.csv, header, [[hba_disp, system_name, args.temp, ratio_str, dG_transfer, logP_x]])
        print(f"\nData appended to {args.csv}")

# ROUTE 5: BINARY EXECUTION (Default)
else:
    results_ideal, results_real, successful_hbds = [], [], []

    for hbd_file in hbd_files:
        hbd_key, hbd_disp, tm_hbd, hf_hbd = get_molecule_info(hbd_file, thermo_db)
        if tm_hbd is None or not os.path.exists(hbd_file):
            print(f"\n[SKIPPED] >> {hbd_disp} >> Missing DB entry or ORCA file!")
            continue

        x_id, T_id, eut_id = get_ideal_curve(tm_hba, hf_hba, tm_hbd, hf_hbd)
        results_ideal.append((x_id, T_id, eut_id))
        
        x_real, T_real = solve_real_curve(hba_file, hbd_file, tm_hba, hf_hba, tm_hbd, hf_hbd, hba_disp, hbd_disp, tol=args.tol)
        results_real.append((x_real, T_real))
        successful_hbds.append(hbd_disp)

    num_plots = len(successful_hbds)
    if num_plots > 0:
        fig, axs = plt.subplots(1, num_plots, figsize=(4.5 * num_plots, 4.0))
        if num_plots == 1: axs = [axs] 
        plt.style.use('seaborn-v0_8-white')
        plt.suptitle(f"Phase Diagrams for {hba_disp}")

        for i in range(num_plots):
            x_id, T_id, eut_id = results_ideal[i]
            x_real, T_real = results_real[i]
            hbd_name = successful_hbds[i]
            
            min_T = min(np.nanmin(T_id), np.nanmin(T_real))
            max_T = max(np.nanmax(T_id), np.nanmax(T_real))
            y_buffer = (max_T - min_T) * 0.15 
            
            axs[i].axhspan(285, 298, alpha=0.35, color='skyblue', label='Room Temp (285-298 K)' if i==0 else "")
            axs[i].plot(x_id, T_id, c='k', linestyle='-.', alpha=0.55, label='Ideal SLE' if i==0 else "")
            axs[i].axvline(eut_id, ls='--', alpha=0.55, c=cmap[1], label='Ideal Eutectic' if i==0 else "")
            axs[i].plot(x_real, T_real, c=cmap[i*2], lw=2.25, label=hbd_name)
            
            axs[i].set_xlim(0.00, 1.00)
            axs[i].set_ylim(min_T - y_buffer, max_T + y_buffer) 
            axs[i].set_xlabel(rf'$\chi_{{{hba_disp.replace(" ", r"~")}}}$')
            if i == 0: axs[i].set_ylabel('Temperature (K)')
            axs[i].grid(alpha=0.7)
            axs[i].set_title(f"{hba_disp} + {hbd_name}")

        plt.figlegend(frameon=True, fancybox=True, framealpha=1, bbox_to_anchor=(0.92, 0.85), shadow=True, loc='upper left', borderpad=0.5, prop={'size': 7.5})
        plt.tight_layout()
        safe_fn = f'{hba_key}_Phase_Diagrams.png'
        plt.savefig(safe_fn, bbox_inches='tight')
        print(f"\nPlot saved successfully as {safe_fn}!")

# ---------------------------------------------------------
# ATLAS SIGN-OFF & EXECUTION TIMER
# ---------------------------------------------------------
if not args.silent:
    elapsed_time = time.time() - start_time
    quote, speaker = random.choice(ATLAS_QUOTES)
    print(f"\n{'-'*85}")
    print(f"ATLAS v{__version__} Execution Completed in {elapsed_time:.2f} seconds.")
    print(f"\"{quote}\"\n — {speaker}")
    print(f"{'-'*85}\n")
