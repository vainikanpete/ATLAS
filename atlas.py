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
# ==============================================================================

__version__ = "1.1.1"

import os
import sys

# ---------------------------------------------------------
# THREAD MANAGEMENT (MUST BE BEFORE NUMPY/SCIPY)
# ---------------------------------------------------------
num_threads = "1" # Safe default for HPC OpenBLAS thrashing
for arg in sys.argv:
    if arg.startswith("--threads="):
        num_threads = arg.split("=")[1]
if "--threads" in sys.argv:
    try:
        idx = sys.argv.index("--threads")
        num_threads = str(int(sys.argv[idx + 1]))
    except (ValueError, IndexError):
        pass

os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

# --- NOW WE CAN SAFELY IMPORT THE HEAVY LIBRARIES ---
import time
import re
import random
import csv
import argparse
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import seaborn as sns
import matplotlib
from typing import List, Tuple, Dict, Optional, Any
import tempfile

# ---------------------------------------------------------
# ATLAS Exceptions
# ---------------------------------------------------------
class AtlasError(Exception):
    """Base exception class for ATLAS."""
    pass

class AtlasDatabaseError(AtlasError):
    """Raised when there is an issue with the thermodynamic database or files."""
    pass

class AtlasCalculationError(AtlasError):
    """Raised when a thermodynamic calculation fails to converge."""
    pass

# ---------------------------------------------------------
# ATLAS Constants & Styling
# ---------------------------------------------------------
T_MIN = 180.0         # Minimum allowed temperature (miscibility gap mask)
T_MAX = 600.0         # Maximum allowed temperature ceiling
X_INF = 1e-7          # Infinite dilution mole fraction
MAX_UNITY_ITER = 50   # Max iterations for bit-nudging unity loop
R = 8.3145            # Thermodynamic Gas Constant [J / (mol K)]

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['text.usetex'] = True
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
cmap = sns.color_palette("husl", 7)

try:
    ATLAS_STYLE = 'seaborn-v0_8-white'
    plt.style.use(ATLAS_STYLE)
except OSError:
    ATLAS_STYLE = 'seaborn-white'
    plt.style.use(ATLAS_STYLE)

# ---------------------------------------------------------
# COSMO-RS Setup & Hotfixes
# ---------------------------------------------------------
from opencosmorspy import COSMORS
import opencosmorspy.molecules as mols

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

# ---------------------------------------------------------
# The ATLAS Quote Vault
# ---------------------------------------------------------
ATLAS_QUOTES = [
    ("Contradictions do not exist. Whenever you think that you are facing a contradiction, check your premises. You will find that one of them is wrong.", "Francisco d'Anconia"),
    ("I swear by my life and my love of it that I will never live for the sake of another man, nor ask another man to live for mine.", "John Galt"),
    ("There are no evil thoughts except one: the refusal to think.", "Dagny Taggart"),
    ("I'm a simple man. I like simple orders and instructions.", "Petteri Vainikka"),
    ("Do not let your fire go out, spark by irreplaceable spark in the hopeless swamps of the not-quite, the not-yet, and the not-at-all.", "John Galt"),
    ("It is not advisable, James, to venture unsolicited opinions. You should spare yourself the embarrassing discovery of their exact value to your listener.", "Francisco d'Anconia"),
    ("Man? What is man? He's just a collection of chemicals with delusions of grandeur.", "Dr. Simon Pritchett"),
    ("Don't ever get angry at a man for stating the truth.", "Dagny Taggart"),
    ("You see, Dr. Stadler, people don't want to think. And the deeper they get into trouble, the less they want to think.", "Dr. Floyd Ferris"),
    ("Physics is like sex. Sure it may have some practical uses, but that's not why we do it.", "Richard Feynman"),
    ("I know poetry is not dead, nor genius lost; nor has Mammon gained power over either, to bind or slay: they will both assert their existence, their presence, their liberty and strength again one day.", "Jane Eyre"),
    ("To arrive at a contradiction is to confess an error in one's thinking; to maintain a contradiction is to abdicate one's mind and to evict oneself from the realm of reality.", "John Galt"),
    ("Devotion to truth is the hallmark of morality; there is no greater, nobler, more heroic form of devotion than the act of a man who assumes the responsibility of thinking.", "John Galt"),
    ("When I disagree with a rational man, I let reality be our final arbiter; if I am right, he will learn; if I am wrong, I will; one of us will win, but both will profit.", "John Galt"),
    ("Power-lust is a weed that grows only in the vacant lots of an abandoned mind.", "John Galt"),
    ("Live and act within the limit of your knowledge and keep expanding it to the limit of your life.", "John Galt"),
    ("Never, I said never, compare with experiment", "Magnus Bergh"),
    ("Theoretical chemistry has of course always been important and useful ... at least to theoretical chemists", "Sven Lidin"),
    ("In this house, we OBEY the laws of thermodynamics!", "Homer Simpson"),
    ("Sie haben also recht gehabt, Sie Spitzbube.","Albert Einstein (letter to Wolfgang Pauli)"),
    ("Computers are incredibly fast, accurate and stupid. Humans are incredibly slow, inaccurate and... also stupid.","Anonymous"),
    ("Everything what mathematicians were saying for the last 50 years is slowly catching up with us.", "David van der Spoel"),
    ("The first principle is that you must not fool yourself and you are the easiest person to fool.", "Richard Feynman"),
    ("I would rather have questions that can't be answered than answers that can't be questioned.", "Richard Feynman"),
    ("Man is unique not because he does science, and he is unique not because he does art, but because science and art equally are expressions of his marvellous plasticity of mind.", "Jacob Bronowski"),
    ("Man masters nature not by force, but by understanding.", "Jacob Bronowski"),
    ("There is no absolute knowledge. And those who claim it, whether they are scientists or dogmatists, open the door to tragedy.", "Jacob Bronowski") 
]

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------
def write_csv(filepath: str, header: List[str], row_data_list: List[List[Any]]) -> None:
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        for row in row_data_list:
            writer.writerow(row)

def load_thermo_database(filepath: str) -> Dict[str, Dict[str, float]]:
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
        raise AtlasDatabaseError(f"Database file not found at {filepath}. Please check your --db_file path.")
    return db

def get_molecule_info(filepath: str, db: Dict[str, Dict[str, float]]) -> Tuple[str, str, Optional[float], Optional[float]]:
    filename = os.path.basename(filepath)
    clean_name = filename.replace('_c000.orcacosmo', '').replace('.orcacosmo', '')
    display_name = clean_name.replace('_', ' ').title()
    if clean_name in db:
        return clean_name, display_name, db[clean_name]['T_m'], db[clean_name]['dH_fus']
    return clean_name, display_name, None, None

def enforce_unity(x_raw: List[float]) -> np.ndarray:
    clean_x = np.array(x_raw, dtype=np.float64)
    clean_x /= np.sum(clean_x)
    clean_x[-1] = 1.0 - np.sum(clean_x[:-1])
    iterations = 0
    # Relaxed to 1e-15 to prevent precision ghosting
    while np.abs(1.0 - clean_x.sum()) > 1e-15 and iterations < MAX_UNITY_ITER:
        if clean_x.sum() < 1.0:
            clean_x[-1] = np.nextafter(clean_x[-1], 2.0)
        else:
            clean_x[-1] = np.nextafter(clean_x[-1], -2.0)
        iterations += 1
    return clean_x

# ---------------------------------------------------------
# ENGINE 1: Binary SLE
# ---------------------------------------------------------
def get_ideal_curve(tm_a: float, hf_a: float, tm_b: float, hf_b: float, x_array: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float, float]:
    x_a = x_array if x_array is not None else np.linspace(0.01, 0.99, 99)
    x_b = 1.0 - x_a
    T_ideal_a = 1 / (1/tm_a - (R/hf_a)*np.log(x_a))
    T_ideal_b = 1 / (1/tm_b - (R/hf_b)*np.log(x_b))
    T_ideal = np.maximum(T_ideal_a, T_ideal_b)
    eutectic_idx = np.argmin(T_ideal)
    return x_a, T_ideal, x_a[eutectic_idx], T_ideal[eutectic_idx]

def solve_real_curve(crs: Any, hba_path: str, hbd_path: str, tm_hba: float, hf_hba: float, tm_hbd: float, hf_hbd: float, hba_disp: str, hbd_disp: str, relax_t: float = 0.15, max_iter: int = 40, tol: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_hba = np.linspace(0.05, 0.95, 20) 
    x_hbd = 1.0 - x_hba
    print(f"\n--- Starting Iterations for {hba_disp} + {hbd_disp} ---")
    _, T_guess, _, _ = get_ideal_curve(tm_hba, hf_hba, tm_hbd, hf_hbd)
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
      
        # --- THE TRUST REGION CLAMP ---
        intended_step = relax_t * (T_raw - T_guess)
        clamped_step = np.clip(intended_step, -15.0, 15.0) 
        
        T_safe = np.clip(T_guess + clamped_step, T_MIN, T_MAX)
        max_diff = np.max(np.abs(T_safe - T_guess))
        
        eutectic_print = f"{np.nanmin(T_safe):>6.2f}" if not np.all(np.isnan(T_safe)) else "   NaN"
        print(f"  Cycle {iteration + 1:>2}/{max_iter} | Max ΔT: {max_diff:>7.4f} K | Est. Eutectic: {eutectic_print} K")
        
        if max_diff < tol:
            print(f"  -> SUCCESS: Converged!\n")
            T_safe[T_safe <= (T_MIN + 0.1)] = np.nan 
            return x_hba, T_safe, np.exp(lng_hba), np.exp(lng_hbd)
        T_guess = T_safe 
        
    print(f"  -> WARNING: Hit max iterations. Final Max ΔT: {max_diff:.4f} K\n")
    T_guess[T_guess <= (T_MIN + 0.1)] = np.nan 
    return x_hba, T_guess, np.exp(lng_hba), np.exp(lng_hbd)

# ---------------------------------------------------------
# ENGINE 2: Ternary SLE
# ---------------------------------------------------------
def get_ternary_grid(steps: int = 15, eps: float = 0.01) -> np.ndarray:
    grid = []
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            k = steps - i - j
            x_a, x_b, x_c = max(i/steps, eps), max(j/steps, eps), max(k/steps, eps)
            tot = x_a + x_b + x_c
            grid.append([x_a/tot, x_b/tot, (1.0 - x_a/tot - x_b/tot)])
    return np.array(grid)

def solve_ternary_curve(crs: Any, mol_paths: List[str], thermo_props: List[Tuple[float, float]], disp_names: List[str], relax_t: float = 0.15, max_iter: int = 40, tol: float = 0.1, progress_callback=None) -> Tuple[np.ndarray, np.ndarray]:
    grid = get_ternary_grid(steps=15) 
    x_a, x_b, x_c = grid[:, 0], grid[:, 1], grid[:, 2]
    print(f"\n--- Starting Ternary Iterations for {disp_names[0]} + {disp_names[1]} + {disp_names[2]} ---")
    
    T_id_a = 1 / (1/thermo_props[0][0] - (R/thermo_props[0][1])*np.log(x_a))
    T_id_b = 1 / (1/thermo_props[1][0] - (R/thermo_props[1][1])*np.log(x_b))
    T_id_c = 1 / (1/thermo_props[2][0] - (R/thermo_props[2][1])*np.log(x_c))
    T_guess = np.maximum.reduce([T_id_a, T_id_b, T_id_c])
    
    for iteration in range(max_iter):
        if progress_callback:
            progress_callback(iteration, max_iter, f"Trust-Region Cycle {iteration + 1}/{max_iter}")
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
        intended_step = relax_t * (T_raw - T_guess)
        clamped_step = np.clip(intended_step, -15.0, 15.0) 
        
        T_safe = np.clip(T_guess + clamped_step, T_MIN, T_MAX) 
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
def solve_pure_solubility(crs: Any, solute_path: str, solv_path: str, tm_s: float, hf_s: float, disp_names: List[str], T_sys: float, relax_x: float = 0.3, max_iter: int = 50, tol: float = 1e-5) -> float:
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
    converged = False
    
    for iteration in range(max_iter):
        x_solv = 1.0 - x_s
        clean_x = enforce_unity([x_s, x_solv])
        
        crs.clear_jobs()
        crs.clear_molecules()
        for path in [solute_path, solv_path]: crs.add_molecule([path])
            
        crs.add_job(x=clean_x, T=T_sys, refst='pure_component')
        try:
            results = crs.calculate()
            lng_s = results['tot']['lng'][0, 0] 
        except Exception as e:
            print(f"  -> FATAL BACKEND CRASH: {e}")
            lng_s = np.nan
            
        if np.isnan(lng_s) or np.abs(lng_s) < 1e-12:
            print(f"  -> ENGINE ABORT: Matrix collapse detected (ln γ_s = {lng_s}). Flagging as NaN.")
            return np.nan
            
        x_s_new_raw = np.exp(k_ideal - lng_s)
        x_s_new = min((relax_x * x_s_new_raw) + ((1.0 - relax_x) * x_s), 0.9999) 
        
        if np.abs(x_s_new - x_s) < tol: 
            print(f"  -> SUCCESS: Converged in {iteration + 1} cycles!")
            print(f"  -> Final Solubility (x_s): {x_s_new:.4e}\n")
            converged = True
            return x_s_new
        x_s = x_s_new
        
    if not converged:
        print(f"  -> WARNING: Failed to converge after {max_iter} iterations. Matrix is oscillating. Flagging as NaN.\n")
        return np.nan


def solve_solubility_curve(crs: Any, solute_path: str, solv_a_path: str, solv_b_path: str, tm_s: float, hf_s: float, disp_names: List[str], T_sys: float, relax_x: float = 0.3, max_iter: int = 50, tol: float = 1e-5, progress_callback=None) -> Tuple[np.ndarray, np.ndarray, float]:
    solv_a_fracs = np.linspace(0.0, 1.0, 21) 
    solubilities = []
    total_steps = len(solv_a_fracs)
    
    print(f"\n--- Calculating Solubility of {disp_names[0]} in {disp_names[1]} + {disp_names[2]} at {T_sys} K ---")
    k_ideal = (hf_s / R) * (1.0 / tm_s - 1.0 / T_sys)
    ideal_solubility = np.exp(k_ideal)
    
    if ideal_solubility >= 1.0:
        print(f"  -> WARNING: System Temp ({T_sys} K) exceeds Solute Melting Point. Capping at x = 0.9999.")
        ideal_solubility = 0.9999
    else:
        print(f"  -> Ideal Solubility limit: {ideal_solubility:.4e} mol fraction")
    
    for idx, frac_A in enumerate(solv_a_fracs):
        if progress_callback:
            progress_callback(idx, total_steps, f"Ratio [{frac_A:.2f} A / {1.0-frac_A:.2f} B]")
        
        safe_frac_A = np.clip(frac_A, 1e-5, 1.0 - 1e-5)
        safe_frac_B = 1.0 - safe_frac_A
        
        x_s = ideal_solubility 
        converged = False
        
        for iteration in range(max_iter):
            x_a = safe_frac_A * (1.0 - x_s)
            x_b = safe_frac_B * (1.0 - x_s)
            clean_x = enforce_unity([x_s, x_a, x_b])
            
            crs.clear_jobs()
            crs.clear_molecules()
            for path in [solute_path, solv_a_path, solv_b_path]: crs.add_molecule([path])
                
            crs.add_job(x=clean_x, T=T_sys, refst='pure_component')
            try:
                results = crs.calculate()
                lng_s = results['tot']['lng'][0, 0]
            except Exception as e:
                print(f"  -> FATAL BACKEND CRASH at Ratio [{frac_A:.2f} A]: {e}")
                lng_s = np.nan
            
            if np.isnan(lng_s) or np.abs(lng_s) < 1e-12:
                print(f"  -> ENGINE ABORT: Matrix collapse at Ratio [{frac_A:.2f} A]. Flagging as NaN.")
                x_s = np.nan
                break
                
            x_s_new_raw = np.exp(k_ideal - lng_s)
            x_s_new = min((relax_x * x_s_new_raw) + ((1.0 - relax_x) * x_s), 0.9999) 
            
            if np.abs(x_s_new - x_s) < tol: 
                x_s = x_s_new
                converged = True
                break
            x_s = x_s_new
            
        if not converged and not np.isnan(x_s):
            print(f"  -> WARNING: Failed to converge after {max_iter} iterations at Ratio [{frac_A:.2f} A]. Flagging as NaN.")
            x_s = np.nan
            
        solubilities.append(x_s)
        if np.isnan(x_s):
            print(f"  Solvent Ratio [{frac_A:.2f} A / {1.0-frac_A:.2f} B] -> Solubility (x_s): [UNRESOLVABLE]")
        else:
            print(f"  Solvent Ratio [{frac_A:.2f} A / {1.0-frac_A:.2f} B] -> Solubility (x_s): {x_s:.4e}")
        
    if progress_callback:
        progress_callback(total_steps, total_steps, "Solubility grid complete!")
        
    return solv_a_fracs, np.array(solubilities), ideal_solubility

# ---------------------------------------------------------
# ENGINE 4: Partition Coefficient (logP) & Transfer dG
# ---------------------------------------------------------
def solve_logp_and_dg(crs: Any, solute_path: str, solv_a_path: str, solv_b_path: str, disp_names: List[str], T_sys: float) -> Tuple[float, float]:
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
def solve_extraction(crs: Any, solute_path: str, water_path: str, des_a_path: str, des_b_path: str, disp_names: List[str], T_sys: float, des_ratio: List[float]) -> Tuple[float, float]:
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
# ENGINE 6: Molecular Fingerprint Scanner
# ---------------------------------------------------------
def parse_cosmo_segments(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    areas, uncorrected_charges, corrected_charges = [], [], []
    in_surface_block, in_corrected_block = False, False
    BOHR_TO_A2 = 0.28002852 
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("# SURFACE POINTS"):
                in_surface_block, in_corrected_block = True, False
                continue
            elif line.startswith("#COSMO_corrected"):
                in_surface_block, in_corrected_block = False, True
                continue
            
            if not line or line.startswith("#") or line.startswith("X") or line.startswith("Cor") or line.startswith("Tot") or line.startswith("C-PCM") or line.startswith("FINAL"):
                continue
                
            if in_surface_block:
                parts = line.split()
                if len(parts) >= 10:
                    try:
                        areas.append(float(parts[3]) * BOHR_TO_A2)
                        uncorrected_charges.append(float(parts[5]))
                    except ValueError:
                        continue
            elif in_corrected_block:
                try: corrected_charges.append(float(line))
                except ValueError: continue
                    
    if not areas:
        raise AtlasError(f"Failed to extract COSMO segments from {os.path.basename(filepath)}.")
        
    areas = np.array(areas)
    charges = np.array(corrected_charges) if len(corrected_charges) == len(areas) else np.array(uncorrected_charges)
    return areas, charges / areas

def generate_sigma_profile(areas: np.ndarray, sigmas: np.ndarray, num_bins: int = 100, sigma_max: float = 0.03) -> Tuple[np.ndarray, np.ndarray]:
    sigmas_clipped = np.clip(sigmas, -sigma_max, sigma_max)
    bins = np.linspace(-sigma_max, sigma_max, num_bins + 1)
    sigma_grid = 0.5 * (bins[:-1] + bins[1:])
    p_sigma, _ = np.histogram(sigmas_clipped, bins=bins, weights=areas)
    p_sigma_smooth = scipy.ndimage.gaussian_filter1d(p_sigma, sigma=2.0)
    return sigma_grid, p_sigma_smooth

def calculate_fingerprint(sigma_grid: np.ndarray, p_sigma: np.ndarray, hbd_cutoff: float = -0.0084, hba_cutoff: float = 0.0084) -> Dict[str, float]:
    hbd_mask = sigma_grid < hbd_cutoff
    np_mask = (sigma_grid >= hbd_cutoff) & (sigma_grid <= hba_cutoff)
    hba_mask = sigma_grid > hba_cutoff
    
    return {
        "HBD_Area": np.sum(p_sigma[hbd_mask]), 
        "NonPolar_Area": np.sum(p_sigma[np_mask]), 
        "HBA_Area": np.sum(p_sigma[hba_mask]), 
        "Total_Area": np.sum(p_sigma)
    }

# ---------------------------------------------------------
# ENGINE 7: Mathematical 2D & 3D Sigma-Map
# ---------------------------------------------------------
COVALENT_RADII = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05, 'P': 1.07, 'F': 0.57, 'Cl': 1.02}

def parse_xyz_block(filepath: str) -> Tuple[List[str], np.ndarray]:
    elements, coords = [], []
    in_xyz, start_idx = False, 0
    with open(filepath, 'r') as f: lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("#XYZ_FILE"):
            in_xyz, start_idx = True, i + 3
            break
    if in_xyz:
        for line in lines[start_idx:]:
            if not line.strip() or line.startswith("#") or line.startswith("$"): break
            parts = line.split()
            if len(parts) >= 4:
                elements.append(parts[0].capitalize())
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return elements, np.array(coords)

def parse_cosmo_segments_3d(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    areas, uncorrected, corrected, coords = [], [], [], []
    in_surf, in_corr = False, False
    B_TO_A2, B_TO_A = 0.28002852, 0.52917721
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("# SURFACE POINTS"):
                in_surf, in_corr = True, False; continue
            elif line.startswith("#COSMO_corrected"):
                in_surf, in_corr = False, True; continue
            if not line or line.startswith("#") or line.startswith("X") or line.startswith("Cor") or line.startswith("Tot") or line.startswith("C-PCM") or line.startswith("FINAL"): continue
            
            if in_surf:
                parts = line.split()
                if len(parts) >= 10:
                    try:
                        coords.append([float(parts[0])*B_TO_A, float(parts[1])*B_TO_A, float(parts[2])*B_TO_A])
                        areas.append(float(parts[3])*B_TO_A2)
                        uncorrected.append(float(parts[5]))
                    except ValueError: continue
            elif in_corr:
                try: corrected.append(float(line))
                except ValueError: continue
                    
    areas, coords = np.array(areas), np.array(coords)
    charges = np.array(corrected) if len(corrected) == len(areas) else np.array(uncorrected)
    return areas, charges / areas, coords

def map_sigma_to_atoms(atom_coords: np.ndarray, surf_coords: np.ndarray, areas: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    atom_sigmas = np.zeros(len(atom_coords))
    atom_areas = np.zeros(len(atom_coords))
    for i, sc in enumerate(surf_coords):
        dists = np.linalg.norm(atom_coords - sc, axis=1)
        closest_atom = np.argmin(dists)
        atom_sigmas[closest_atom] += sigmas[i] * areas[i]
        atom_areas[closest_atom] += areas[i]
        
    mask = atom_areas > 0
    atom_sigmas[mask] = atom_sigmas[mask] / atom_areas[mask]
    return atom_sigmas

def infer_bonds(elements: List[str], coords: np.ndarray, tolerance: float = 1.25) -> List[Tuple[int, int]]:
    bonds = []
    n_atoms = len(elements)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            r_i = COVALENT_RADII.get(elements[i], 0.76)
            r_j = COVALENT_RADII.get(elements[j], 0.76)
            if dist < (r_i + r_j) * tolerance:
                bonds.append((i, j))
    return bonds

def project_to_2d(coords: np.ndarray) -> np.ndarray:
    centered = coords - np.mean(coords, axis=0)
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    return np.dot(centered, evecs[:, idx][:, :2])

def render_publication_surface(elements: List[str], coords: np.ndarray, atom_sigmas: np.ndarray, 
                               out_file: str, disp_name: str, args, animate: bool = False) -> bool:
    try:
        from xyzrender import render, render_gif, load
    except ImportError:
        return False

    # --- CLOUD-SAFE PATH ROUTING ---
    tmp_dir = tempfile.gettempdir()
    safe_prefix = disp_name.replace(' ', '_')
    temp_dens = os.path.join(tmp_dir, f"temp_{safe_prefix}_dens.cube")
    temp_esp = os.path.join(tmp_dir, f"temp_{safe_prefix}_esp.cube")
       
    A_TO_BOHR = 1.8897259886
    canvas_size = 1200
    
    try:
        # 1. High-Res Grid Synthesis
        margin = 4.5 
        min_c = np.min(coords, axis=0) - margin
        max_c = np.max(coords, axis=0) + margin
        N_grid = 100 
        x, y, z = [np.linspace(min_c[i], max_c[i], N_grid) for i in range(3)]
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        dens_grid, esp_grid, weight_sum = [np.zeros(N_grid**3) for _ in range(2)] + [np.zeros(N_grid**3) + 1e-12]
        for i, c in enumerate(coords):
            r2 = np.sum((grid_coords - c)**2, axis=1)
            dens_grid += np.exp(-0.8 * r2)
            weight = np.exp(-1.5 * r2) 
            esp_grid += (-atom_sigmas[i]) * weight
            weight_sum += weight
        esp_grid /= weight_sum

        # 2. Write Cube Files (Bohr Scaling)
        steps = (max_c - min_c) / (N_grid - 1)
        periodic = {'H':1, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'SI':14, 'P':15, 'S':16, 'CL':17, 'BR':35, 'I':53}
        
        def write_cube(fname, flat_grid):
            grid_3d = flat_grid.reshape((N_grid, N_grid, N_grid))
            with open(fname, 'w') as f:
                f.write("ATLAS 3D Sigma-Map\nGenerated via xyzrender\n")
                f.write(f"{len(coords):5d} {min_c[0]*A_TO_BOHR:12.6f} {min_c[1]*A_TO_BOHR:12.6f} {min_c[2]*A_TO_BOHR:12.6f}\n")
                for i in range(3):
                    vec = [0.0]*3; vec[i] = steps[i]*A_TO_BOHR
                    f.write(f"{N_grid:5d} {vec[0]:12.6f} {vec[1]:12.6f} {vec[2]:12.6f}\n")
                for i, el in enumerate(elements):
                    Z = periodic.get(el.upper(), 6)
                    f.write(f"{Z:5d} {float(Z):12.6f} {coords[i,0]*A_TO_BOHR:12.6f} {coords[i,1]*A_TO_BOHR:12.6f} {coords[i,2]*A_TO_BOHR:12.6f}\n")
                for ix in range(N_grid):
                    for iy in range(N_grid):
                        row = grid_3d[ix, iy, :]
                        for i_c in range(0, N_grid, 6):
                            f.write("".join([f" {v:12.5E}" for v in row[i_c:i_c+6]]) + "\n")

        write_cube(temp_dens, dens_grid); write_cube(temp_esp, esp_grid)

        # 3. Generate ESP Atom Colors (used if not in cloud mode)
        sigma_max = args.sigma_max if hasattr(args, 'sigma_max') else 0.015
        norm = mcolors.TwoSlopeNorm(vmin=-sigma_max, vcenter=0.0, vmax=sigma_max)
        sigma_cmap = mcolors.LinearSegmentedColormap.from_list('sigma_cmap', ['royalblue', 'lightgray', 'crimson'])
        
        esp_highlights = []
        for i, sigma in enumerate(atom_sigmas):
            hex_color = mcolors.to_hex(sigma_cmap(norm(sigma)))
            esp_highlights.append(([i + 1], hex_color))

        # 4. Build Base Render Configuration (No Vector Arrow)
        mol_obj = load(temp_dens)
        render_kwargs = {
            "molecule": mol_obj, "canvas_size": canvas_size,
            "iso": 0.05, "opacity": 0.75, "idx": "s", "label_font_size": 18.0,
            "hy": True, "orient": True
        }

        # 5. Fork Engine Mode (Cloud vs Colored Atoms)
        use_cloud = hasattr(args, 'cloud') and args.cloud
        
        if use_cloud:
            render_kwargs["esp"] = temp_esp
        else:
            render_kwargs["highlight"] = esp_highlights
            render_kwargs["mol_color"] = "gainsboro" 

        # 6. Execute Render
        if animate:
            # Strip incompatible variables to prevent render_gif from crashing
            excluded_kwargs = ["molecule", "canvas_size", "idx", "orient", "esp", "opacity"] 
            safe_kwargs = {k: v for k, v in render_kwargs.items() if k not in excluded_kwargs}
            
            render_gif(molecule=temp_dens, output=out_file.replace('.png', '.gif'), 
                       gif_rot="y", rot_frames=80, **safe_kwargs)
            return True

        else:
            result = render(**render_kwargs)
            from xyzrender.export import svg_to_png
            svg_to_png(str(result), out_file, size=canvas_size)
            
            # 7. Fast NumPy Auto-Crop & Matplotlib Bar Overlay
            try:
                img = mpimg.imread(out_file)
                
                if img.shape[-1] == 4:
                    bg_mask = (img[:, :, 3] < 0.1) | (np.min(img[:, :, :3], axis=2) > 0.98)
                else:
                    bg_mask = np.min(img, axis=2) > 0.98
                    
                content_mask = ~bg_mask
                coords_mask = np.argwhere(content_mask)
                
                if coords_mask.size > 0:
                    y0, x0 = coords_mask.min(axis=0)
                    y1, x1 = coords_mask.max(axis=0)
                    pad = 40
                    img = img[max(0, y0 - pad):min(img.shape[0], y1 + pad), 
                              max(0, x0 - pad):min(img.shape[1], x1 + pad)]

                h, w = img.shape[:2]
                dpi = 300
                cbar_pad_px = 180 
                
                fig_w_inch = w / dpi
                fig_h_inch = (h + cbar_pad_px) / dpi
                
                fig = plt.figure(figsize=(fig_w_inch, fig_h_inch), dpi=dpi)
                
                bottom_frac = cbar_pad_px / (h + cbar_pad_px)
                ax_img = fig.add_axes([0, bottom_frac, 1, 1 - bottom_frac])
                ax_img.imshow(img)
                ax_img.axis('off')
                
                sm = plt.cm.ScalarMappable(cmap=sigma_cmap, norm=norm)
                sm.set_array([])
                
                cbar_h_frac = (40 / dpi) / fig_h_inch 
                cbar_y_frac = (70 / dpi) / fig_h_inch 
                cbar_ax = fig.add_axes([0.15, cbar_y_frac, 0.7, cbar_h_frac]) 
                
                cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
                cbar.set_label(r'Screening Charge Density, $\sigma$ ($e$/Å$^2$)', fontsize=12, labelpad=8)
                
                cbar.outline.set_linewidth(0.75)
                cbar.ax.tick_params(labelsize=9, width=0.75, length=4)
                
                ticks = [-sigma_max, args.hbd_cutoff, 0.0, args.hba_cutoff, sigma_max]
                cbar.set_ticks(ticks)
                cbar.ax.set_xticklabels([f"{t:+.3f}" if t != 0 else "0.000" for t in ticks])
                
                plt.savefig(out_file, dpi=dpi, facecolor='white')
                plt.close(fig)
                
            except Exception as e:
                print(f"  -> WARNING: Matplotlib colorbar overlay failed: {e}")
                
            return True

    except Exception as e:
        print(f"  -> WARNING: Rendering failed: {e}")
        return False
    finally:
        for f in [temp_dens, temp_esp]: 
            if os.path.exists(f): os.remove(f)

# ---------------------------------------------------------
# ENGINE 8: Molecular Similarity Matcher (ASIS)
# ---------------------------------------------------------
def calculate_asis(sigma_grid: np.ndarray, p_target: np.ndarray, p_bench: np.ndarray, hbd_cutoff: float = -0.0084, hba_cutoff: float = 0.0084) -> Dict[str, Any]:
    d_sigma = sigma_grid[1] - sigma_grid[0]
    norm_target = p_target / (np.sum(p_target) * d_sigma)
    norm_bench = p_bench / (np.sum(p_bench) * d_sigma)
    
    intersection = np.minimum(norm_target, norm_bench)
    
    hbd_mask = sigma_grid < hbd_cutoff
    np_mask = (sigma_grid >= hbd_cutoff) & (sigma_grid <= hba_cutoff)
    hba_mask = sigma_grid > hba_cutoff
    
    return {
        "Global_ASIS": np.sum(intersection) * d_sigma * 100.0,
        "HBD_Contribution": np.sum(intersection[hbd_mask]) * d_sigma * 100.0,
        "NP_Contribution": np.sum(intersection[np_mask]) * d_sigma * 100.0,
        "HBA_Contribution": np.sum(intersection[hba_mask]) * d_sigma * 100.0,
        "norm_target": norm_target,
        "norm_bench": norm_bench,
        "intersection": intersection
    }

# ---------------------------------------------------------
# ROUTING CONTROLLERS (Isolated Execution Logic)
# ---------------------------------------------------------
def run_fingerprint_cli(args, thermo_db: dict) -> None:
    target_file = os.path.join(args.orcacosmo_dir, args.mol + '_c000.orcacosmo')
    if not os.path.exists(target_file): raise AtlasDatabaseError(f"Target '{args.mol}' missing.")
    
    _, disp_name, _, _ = get_molecule_info(target_file, thermo_db)
    print(f"\n--- Scanning Molecular Fingerprint: {disp_name} ---")
    
    areas, sigmas = parse_cosmo_segments(target_file)
    sigma_grid, p_sigma = generate_sigma_profile(areas, sigmas, num_bins=args.sigma_bins, sigma_max=args.sigma_max)
    fingerprint = calculate_fingerprint(sigma_grid, p_sigma, hbd_cutoff=args.hbd_cutoff, hba_cutoff=args.hba_cutoff)
    
    print(f"  -> Total Surface Area : {fingerprint['Total_Area']:>7.2f} Å²")
    print(f"  -> H-Bond Donor (HBD) : {fingerprint['HBD_Area']:>7.2f} Å²  [Blue]")
    print(f"  -> Non-Polar Bulk (NP): {fingerprint['NonPolar_Area']:>7.2f} Å²  [Grey]")
    print(f"  -> H-Bond Acceptor(HBA): {fingerprint['HBA_Area']:>7.2f} Å²  [Red]\n")
    
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.style.use(ATLAS_STYLE)
    ax.plot(sigma_grid, p_sigma, color='black', lw=1.5)
    
    ax.fill_between(sigma_grid, p_sigma, where=(sigma_grid < args.hbd_cutoff), color='royalblue', alpha=0.5, label='HB Donor')
    ax.fill_between(sigma_grid, p_sigma, where=((sigma_grid >= args.hbd_cutoff) & (sigma_grid <= args.hba_cutoff)), color='gray', alpha=0.3, label='Non-Polar')
    ax.fill_between(sigma_grid, p_sigma, where=(sigma_grid > args.hba_cutoff), color='crimson', alpha=0.5, label='HB Acceptor')
    
    ax.axvline(args.hbd_cutoff, color='k', linestyle='--', lw=1, alpha=0.5)
    ax.axvline(args.hba_cutoff, color='k', linestyle='--', lw=1, alpha=0.5)
    ax.set_xlim(-args.sigma_max, args.sigma_max) 
    ax.set_ylim(0, np.max(p_sigma) * 1.15)
    
    ax.set_xlabel(r'Screening Charge Density, $\sigma$ ($e$/Å$^2$)')
    ax.set_ylabel(r'Area, $P(\sigma)$ (Å$^2$)')
    ax.set_title(f"$\sigma$-Profile Fingerprint: {disp_name}")
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    safe_fn = f'{disp_name.replace(" ", "_")}_Fingerprint.png'
    plt.savefig(safe_fn, bbox_inches='tight', dpi=500)
    print(f"  -> Visual Fingerprint saved to {safe_fn}")

def run_render_cli(args, thermo_db: dict) -> None:
    target_file = os.path.join(args.orcacosmo_dir, args.mol + '_c000.orcacosmo')
    if not os.path.exists(target_file): raise AtlasDatabaseError(f"Target '{args.mol}' missing.")
    _, disp_name, _, _ = get_molecule_info(target_file, thermo_db)
    print(f"\n--- Generating Thermodynamic σ-Map: {disp_name} ---")
    
    elements, atom_coords_3d = parse_xyz_block(target_file)
    areas, sigmas, surf_coords = parse_cosmo_segments_3d(target_file)
    atom_sigmas = map_sigma_to_atoms(atom_coords_3d, surf_coords, areas, sigmas)
    
    safe_fn = f'{disp_name.replace(" ", "_")}_SigmaMap.png'
    
    # 1. Attempt High-Fidelity 3D Render
    success = render_publication_surface(elements, atom_coords_3d, atom_sigmas, safe_fn, disp_name, args, animate=args.animate)

    # 2. Graceful Fallback to 2D PCA Projection
    if success:
        print(f"  -> SUCCESS: High-fidelity 3D Sigma-Map rendered via xyzrender to {safe_fn}\n")
    else:
        print("  -> Falling back to 3D PCA projection (Matplotlib)...")
        coords_2d = project_to_2d(atom_coords_3d)
        bonds = infer_bonds(elements, atom_coords_3d)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.style.use(ATLAS_STYLE)
        
        x_min, x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
        y_min, y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()
        x_pad, y_pad = max((x_max - x_min) * 0.15, 1.0), max((y_max - y_min) * 0.15, 1.0)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        
        for idx1, idx2 in bonds:
            if idx1 < len(coords_2d) and idx2 < len(coords_2d):
                c1, c2 = coords_2d[idx1], coords_2d[idx2]
                ax.plot([c1[0], c2[0]], [c1[1], c2[1]], color='dimgray', lw=4, zorder=1)
           
        norm = mcolors.TwoSlopeNorm(vmin=-0.015, vcenter=0.0, vmax=0.015)
        cmap = mcolors.LinearSegmentedColormap.from_list('sigma_cmap', ['royalblue', 'lightgray', 'crimson'])
        sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], s=1200, c=atom_sigmas, cmap=cmap, norm=norm, edgecolors='black', linewidths=2.0, zorder=2)
        
        for i, el in enumerate(elements):
            text_color = 'white' if abs(atom_sigmas[i]) > 0.006 else 'black'
            ax.text(coords_2d[i, 0], coords_2d[i, 1], el, ha='center', va='center', fontsize=14, fontweight='bold', color=text_color, zorder=3)
                   
        cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04, aspect=40)
        cbar.set_label(r'Screening Charge Density, $\sigma$ ($e$/Å$^2$)', fontsize=14, labelpad=10)
        
        ax.set_title(f"Thermodynamic $\sigma$-Map: {disp_name}", fontsize=20, pad=20, fontweight='bold')
        ax.axis('off')
        
        plt.savefig(safe_fn, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  -> SUCCESS: Mathematical 2D Sigma-Map saved to {safe_fn}\n")

def run_match_cli(args, thermo_db: dict) -> None:
    target_path = os.path.join(args.orcacosmo_dir, args.target + '_c000.orcacosmo')
    if not os.path.exists(target_path): raise AtlasDatabaseError(f"Target '{args.target}' missing.")
        
    _, target_disp, _, _ = get_molecule_info(target_path, thermo_db)
    areas_t, sigmas_t = parse_cosmo_segments(target_path)
    grid_t, p_target = generate_sigma_profile(areas_t, sigmas_t, num_bins=args.sigma_bins, sigma_max=args.sigma_max)
    
    print(f"\n--- ASIS Similarity Screening: Target [{target_disp}] ---")
    
    results = []
    for bench in args.benchmarks:
        bench_path = os.path.join(args.orcacosmo_dir, bench + '_c000.orcacosmo')
        if not os.path.exists(bench_path):
            print(f"  [SKIPPED] Benchmark '{bench}' not found.")
            continue
            
        _, bench_disp, _, _ = get_molecule_info(bench_path, thermo_db)
        areas_b, sigmas_b = parse_cosmo_segments(bench_path)
        _, p_bench = generate_sigma_profile(areas_b, sigmas_b, num_bins=args.sigma_bins, sigma_max=args.sigma_max)
        
        asis_data = calculate_asis(grid_t, p_target, p_bench, hbd_cutoff=args.hbd_cutoff, hba_cutoff=args.hba_cutoff)
        asis_data['bench_disp'] = bench_disp
        results.append(asis_data)
        
    results = sorted(results, key=lambda x: x['Global_ASIS'], reverse=True)
    
    for i, res in enumerate(results):
        print(f"\n  [{i+1}] {res['bench_disp']:<20} | Global ASIS: {res['Global_ASIS']:>5.1f}%")
        print(f"      ├─ HBD Match: {res['HBD_Contribution']:>4.1f}%")
        print(f"      ├─ NP  Match: {res['NP_Contribution']:>4.1f}%")
        print(f"      └─ HBA Match: {res['HBA_Contribution']:>4.1f}%")
        
    num_plots = len(results)
    if num_plots > 0:
        fig, axs = plt.subplots(num_plots, 1, figsize=(7, 3.5 * num_plots), squeeze=False)
        plt.style.use(ATLAS_STYLE)
        
        for i, res in enumerate(results):
            ax = axs[i, 0]
            ax.plot(grid_t, res['norm_bench'], color='gray', lw=2, linestyle='--', label=f"{res['bench_disp']} (Bench)")
            ax.plot(grid_t, res['norm_target'], color='black', lw=2, label=f"{target_disp} (Target)")
            ax.fill_between(grid_t, res['intersection'], color='mediumpurple', alpha=0.4, label=f"ASIS Overlap ({res['Global_ASIS']:.1f}%)")
            
            ax.axvline(args.hbd_cutoff, color='k', linestyle=':', lw=1, alpha=0.5)
            ax.axvline(args.hba_cutoff, color='k', linestyle=':', lw=1, alpha=0.5)
            ax.set_xlim(-args.sigma_max, args.sigma_max)
            ax.set_ylim(0, max(np.max(res['norm_target']), np.max(res['norm_bench'])) * 1.15)
            
            ax.set_xlabel(r'Screening Charge Density, $\sigma$ ($e$/Å$^2$)')
            ax.set_ylabel('Probability Density')
            ax.set_title(f"ASIS Matching: {target_disp} vs {res['bench_disp']}")
            ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
            ax.grid(alpha=0.3)
            
        plt.tight_layout()
        safe_fn = f'{target_disp.replace(" ", "_")}_ASIS_Screen.png'
        plt.savefig(safe_fn, bbox_inches='tight', dpi=500)
        print(f"\n  -> ASIS Validation Plot saved to {safe_fn}")

def run_binary_cli(args, thermo_db: dict, crs: Any, hba_file: str, hbd_files: List[str], hba_key: str, hba_disp: str, tm_hba: float, hf_hba: float) -> None:
    results_ideal, results_real, successful_hbds = [], [], []

    for hbd_file in hbd_files:
        hbd_key, hbd_disp, tm_hbd, hf_hbd = get_molecule_info(hbd_file, thermo_db)
        if tm_hbd is None or not os.path.exists(hbd_file):
            print(f"\n[SKIPPED] >> {hbd_disp} >> Missing DB entry or ORCA file!")
            continue

        x_id, T_id, eut_id_x, eut_id_T = get_ideal_curve(tm_hba, hf_hba, tm_hbd, hf_hbd)
        results_ideal.append((x_id, T_id, eut_id_x))
        
        x_real, T_real, gamma_hba, gamma_hbd = solve_real_curve(
            crs, hba_file, hbd_file, tm_hba, hf_hba, tm_hbd, hf_hbd, hba_disp, hbd_disp, relax_t=args.relax_t, tol=args.tol)
        results_real.append((x_real, T_real))
        successful_hbds.append(hbd_disp)

        valid_idx = np.where(~np.isnan(T_real))[0]
        if len(valid_idx) > 0:
            eut_real_idx = valid_idx[np.argmin(T_real[valid_idx])]
            T_e_real = T_real[eut_real_idx]
            x_e_real = x_real[eut_real_idx]
            delta_T = T_e_real - eut_id_T
            print(f"\n--- Eutectic Summary: {hba_disp} + {hbd_disp} ---")
            print(f"  -> Ideal Eutectic: T = {eut_id_T:.2f} K at x_HBA = {eut_id_x:.4f}")
            print(f"  -> Real Eutectic : T = {T_e_real:.2f} K at x_HBA = {x_e_real:.4f}")
            print(f"  -> Eutectic Depth (ΔTe): {delta_T:.2f} K")
            print(f"  -> Activity Coeffs at Te: γ_HBA = {gamma_hba[eut_real_idx]:.4f}, γ_HBD = {gamma_hbd[eut_real_idx]:.4f}\n")
        else:
            eut_real_idx = None
            print(f"\n--- Eutectic Summary: {hba_disp} + {hbd_disp} ---")
            print("  -> WARNING: No valid liquidus points found.\n")

        if args.csv:
            _, T_id_matched, _, _ = get_ideal_curve(tm_hba, hf_hba, tm_hbd, hf_hbd, x_array=x_real)
            header = ['System', 'x_HBA', 'x_HBD', 'T_ideal_K', 'T_real_K', 'gamma_HBA', 'gamma_HBD', 'Note']
            rows = []
            for i in range(len(x_real)):
                note = "Eutectic Minimum" if (eut_real_idx is not None and i == eut_real_idx) else ""
                rows.append([f"{hba_disp} + {hbd_disp}", x_real[i], 1.0 - x_real[i], T_id_matched[i], T_real[i], gamma_hba[i], gamma_hbd[i], note])
            write_csv(args.csv, header, rows)
            print(f"  -> Phase diagram exported to CSV: {args.csv}\n")

    num_plots = len(successful_hbds)
    if num_plots > 0:
        fig, axs = plt.subplots(1, num_plots, figsize=(4.5 * num_plots, 4.0))
        if num_plots == 1: axs = [axs] 
        plt.style.use(ATLAS_STYLE)
        plt.suptitle(f"Phase Diagrams for {hba_disp}")

        for i in range(num_plots):
            x_id, T_id, eut_id_x = results_ideal[i]
            x_real, T_real = results_real[i]
            hbd_name = successful_hbds[i]
            
            # --- THE NAN SHIELD (BINARY) ---
            valid_T_real = T_real[~np.isnan(T_real)]
            if len(valid_T_real) > 0:
                min_T = min(np.nanmin(T_id), np.nanmin(valid_T_real))
                max_T = max(np.nanmax(T_id), np.nanmax(valid_T_real))
            else:
                min_T = np.nanmin(T_id)
                max_T = np.nanmax(T_id)
                
            y_buffer = (max_T - min_T) * 0.15 
            
            axs[i].axhspan(285, 298, alpha=0.35, color='skyblue', label='Room Temp (285-298 K)' if i==0 else "")
            axs[i].plot(x_id, T_id, c='k', linestyle='-.', alpha=0.55, label='Ideal SLE' if i==0 else "")
            axs[i].axvline(eut_id_x, ls='--', alpha=0.55, c=cmap[1], label='Ideal Eutectic' if i==0 else "")
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

def run_ternary_cli(args, thermo_db: dict, crs: Any, hba_file: str, hbd_files: List[str], hba_key: str, hba_disp: str, tm_hba: float, hf_hba: float) -> None:
    if len(hbd_files) != 2:
        raise AtlasError("Ternary evaluation requires exactly 2 HBDs.")
        
    mol_paths = [hba_file, hbd_files[0], hbd_files[1]]
    thermo_props = [(tm_hba, hf_hba)]
    disp_names = [hba_disp]
    
    for path in [hbd_files[0], hbd_files[1]]:
        _, disp, tm, hf = get_molecule_info(path, thermo_db)
        if tm is None or not os.path.exists(path):
            raise AtlasDatabaseError(f"Missing database entry or ORCA file for {disp}. Aborting ternary.")
        thermo_props.append((tm, hf))
        disp_names.append(disp)
        
    grid, T_real = solve_ternary_curve(crs, mol_paths, thermo_props, disp_names, relax_t=args.relax_t, tol=args.tol)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.style.use(ATLAS_STYLE)
    X = grid[:, 1] + 0.5 * grid[:, 2]
    Y = grid[:, 2] * (np.sqrt(3.0) / 2.0)
    
    triang = tri.Triangulation(X, Y)
    mask = np.any(np.isnan(T_real[triang.triangles]), axis=1)
    triang.set_mask(mask)
    
    # --- THE NAN SHIELD (TERNARY) ---
    valid_T_real = T_real[~np.isnan(T_real)]
    if len(valid_T_real) > 0:
        T_min = np.nanmin(valid_T_real)
        T_max_plot = np.nanpercentile(valid_T_real, 90)
        levels = np.linspace(T_min, T_max_plot + 1e-5, 50) 
        contour = ax.tricontourf(triang, T_real, levels=levels, cmap='magma', extend='max')
        fig.colorbar(contour, ax=ax, label='Liquidus Temperature (K)', pad=0.12)
    else:
        print("\n  [WARNING] Entire Ternary grid hit T_MIN. Plotting empty simplex.")
    
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

def run_solubility_cli(args, thermo_db: dict, crs: Any, hba_file: str, hbd_files: List[str], hba_key: str, hba_disp: str, tm_hba: float, hf_hba: float) -> None:
    if len(hbd_files) == 1:
        _, disp_a, _, _ = get_molecule_info(hbd_files[0], thermo_db)
        disp_names = [hba_disp, disp_a]
        # Tol falls back to 1e-5 natively
        final_sol = solve_pure_solubility(crs, hba_file, hbd_files[0], tm_hba, hf_hba, disp_names, T_sys=args.temp, relax_x=args.relax_x) 
        if args.csv:
            write_csv(args.csv, ['System', 'T_sys_K', 'Solubility_xS'], [[f"{hba_disp} in {disp_a}", args.temp, final_sol]])
            print(f"\nData appended to {args.csv}")

    elif len(hbd_files) == 2:
        _, disp_a, _, _ = get_molecule_info(hbd_files[0], thermo_db)
        _, disp_b, _, _ = get_molecule_info(hbd_files[1], thermo_db)
        disp_names = [hba_disp, disp_a, disp_b]
        # Tol falls back to 1e-5 natively
        x_solvent, y_solubility, ideal_sol = solve_solubility_curve(crs, hba_file, hbd_files[0], hbd_files[1], tm_hba, hf_hba, disp_names, T_sys=args.temp, relax_x=args.relax_x)
        
        if args.csv:
            header = ['System', 'T_sys_K', 'x_' + disp_a, 'x_' + disp_b, 'Solubility_xS']
            rows = [[f"{hba_disp} in {disp_a}+{disp_b}", args.temp, x_solvent[i], 1.0 - x_solvent[i], y_solubility[i]] for i in range(len(x_solvent))]
            write_csv(args.csv, header, rows)
            print(f"\nData appended to {args.csv}")

        fig, ax = plt.subplots(figsize=(6, 4))
        plt.style.use(ATLAS_STYLE)
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
        raise AtlasError("Solubility evaluation requires exactly 1 or 2 Solvents.")

def run_logp_cli(args, thermo_db: dict, crs: Any, hba_file: str, hbd_files: List[str], hba_key: str, hba_disp: str) -> None:
    if len(hbd_files) != 2:
        raise AtlasError("logP evaluation requires exactly 2 Solvents to transfer between.")
        
    _, disp_a, _, _ = get_molecule_info(hbd_files[0], thermo_db)
    _, disp_b, _, _ = get_molecule_info(hbd_files[1], thermo_db)
    disp_names = [hba_disp, disp_a, disp_b]
    dG_transfer, logP_x = solve_logp_and_dg(crs, hba_file, hbd_files[0], hbd_files[1], disp_names, T_sys=args.temp)
    
    if args.csv:
        header = ['Solute', 'System', 'T_sys_K', 'dG_transfer_kJ_mol', 'logP_x']
        system_name = f"{disp_names[1]} -> {disp_names[2]}"
        write_csv(args.csv, header, [[hba_disp, system_name, args.temp, dG_transfer, logP_x]])
        print(f"\nData appended to {args.csv}")

def run_extraction_cli(args, thermo_db: dict, crs: Any, hba_file: str, hbd_files: List[str], hba_key: str, hba_disp: str) -> None:
    if len(hbd_files) != 3:
        raise AtlasError("DES Extraction requires exactly 3 Solvents (e.g., h2o des_a des_b).")
        
    _, disp_w, _, _ = get_molecule_info(hbd_files[0], thermo_db)
    _, disp_des_a, _, _ = get_molecule_info(hbd_files[1], thermo_db)
    _, disp_des_b, _, _ = get_molecule_info(hbd_files[2], thermo_db)
    disp_names = [hba_disp, disp_w, disp_des_a, disp_des_b]
    dG_transfer, logP_x = solve_extraction(crs, hba_file, hbd_files[0], hbd_files[1], hbd_files[2], disp_names, T_sys=args.temp, des_ratio=args.ratio)
    
    if args.csv:
        header = ['Solute', 'System', 'T_sys_K', 'DES_Ratio', 'dG_transfer_kJ_mol', 'logP_x']
        system_name = f"{disp_names[1]} -> DES[{disp_names[2]}:{disp_names[3]}]"
        ratio_str = f"{args.ratio[0]}:{args.ratio[1]}"
        write_csv(args.csv, header, [[hba_disp, system_name, args.temp, ratio_str, dG_transfer, logP_x]])
        print(f"\nData appended to {args.csv}")

# ---------------------------------------------------------
# MAIN EXECUTION BLOCK (REFACTORED FOR SUBPARSERS)
# ---------------------------------------------------------
def main():
    start_time = time.time()
    
    # 1. Create a Base Parser for GLOBAL arguments so they work anywhere in the command
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--threads', type=int, default=10, help="Number of OpenBLAS/OMP threads (default: 10)")
    base_parser.add_argument('--silent', action='store_true', help="Suppress final quote and timing output")
    base_parser.add_argument('--orcacosmo_dir', type=str, default='./molecules', help="Path to directory containing .orcacosmo files")
    base_parser.add_argument('--db_file', type=str, default='./database/thermo_db.dat', help="Path to thermodynamic database CSV")
    
    # 2. Main Parser
    parser = argparse.ArgumentParser(description="ATLAS: Advanced Thermodynamic Liquid & Aqueous Solver", parents=[base_parser])
    parser.add_argument('--status', action='store_true', help="Check thermodynamic database status")
    
    subparsers = parser.add_subparsers(dest='command', help='Available ATLAS operations')

    def add_thermo_args(p):
        p.add_argument('--relax_t', type=float, default=0.15, help="Temperature relaxation factor (default 0.15)")
        p.add_argument('--relax_x', type=float, default=0.3, help="Mole fraction relaxation factor (default 0.3)")
        p.add_argument('--tol', type=float, default=0.1, help="Convergence tolerance in Kelvin (default 0.1)")
        p.add_argument('--csv', type=str, metavar='FILE', help="Save numerical output to CSV")

    def add_sigma_args(p):
        p.add_argument('--sigma_max', type=float, default=0.03, help="Maximum boundary for sigma grid")
        p.add_argument('--sigma_bins', type=int, default=100, help="Number of bins for profile resolution")
        p.add_argument('--hbd_cutoff', type=float, default=-0.0084, help="Klamt H-Bond Donor threshold")
        p.add_argument('--hba_cutoff', type=float, default=0.0084, help="Klamt H-Bond Acceptor threshold")

    # 3. Add `parents=[base_parser]` to EVERY subparser
    p_bin = subparsers.add_parser('binary', parents=[base_parser], help='Calculate Binary SLE phase diagram')
    p_bin.add_argument('--hba', type=str, required=True, help="Solute/HBA molecule")
    p_bin.add_argument('--hbd', nargs='+', type=str, required=True, help="List of Solvent/HBD molecules")
    add_thermo_args(p_bin)

    p_tern = subparsers.add_parser('ternary', parents=[base_parser], help='Calculate Ternary SLE phase diagram')
    p_tern.add_argument('--hba', type=str, required=True)
    p_tern.add_argument('--hbd', nargs=2, type=str, required=True)
    add_thermo_args(p_tern)

    p_sol = subparsers.add_parser('solubility', parents=[base_parser], help='Calculate Solute solubility')
    p_sol.add_argument('--hba', type=str, required=True)
    p_sol.add_argument('--hbd', nargs='+', type=str, required=True)
    p_sol.add_argument('--temp', type=float, default=298.15, help="System Temp (K)")
    add_thermo_args(p_sol)

    p_logp = subparsers.add_parser('logp', parents=[base_parser], help='Calculate partition coefficient (logP)')
    p_logp.add_argument('--hba', type=str, required=True)
    p_logp.add_argument('--hbd', nargs=2, type=str, required=True)
    p_logp.add_argument('--temp', type=float, default=298.15)
    p_logp.add_argument('--csv', type=str)

    p_ext = subparsers.add_parser('extract', parents=[base_parser], help='Calculate extraction logP to Mixed DES')
    p_ext.add_argument('--hba', type=str, required=True)
    p_ext.add_argument('--hbd', nargs=3, type=str, required=True)
    p_ext.add_argument('--ratio', nargs=2, type=float, default=[1.0, 2.0])
    p_ext.add_argument('--temp', type=float, default=298.15)
    p_ext.add_argument('--csv', type=str)

    p_fp = subparsers.add_parser('fingerprint', parents=[base_parser], help='Generate Sigma-Profile fingerprint')
    p_fp.add_argument('--mol', type=str, required=True, help="Molecule name to scan")
    add_sigma_args(p_fp)

    p_map = subparsers.add_parser('sigma-map', parents=[base_parser], help='Calculate High-Fidelity 3D Sigma-Map')
    p_map.add_argument('--mol', type=str, required=True)
    p_map.add_argument('--animate', action='store_true', help="Generate an animated GIF rotation")
    p_map.add_argument('--cloud', action='store_true', help="Draw the translucent ESP cloud instead of coloring atoms")
    add_sigma_args(p_map) 

    p_match = subparsers.add_parser('match', parents=[base_parser], help='Calculate ASIS Similarity between molecules')
    p_match.add_argument('--target', type=str, required=True, help="Candidate molecule to score")
    p_match.add_argument('--benchmarks', nargs='+', type=str, required=True, help="List of benchmark archetypes")
    add_sigma_args(p_match)

    # 4. Parse arguments after all parsers are defined
    args = parser.parse_args()
    
    # 5. Run safety intercepts
    if getattr(args, 'command', None) == 'sigma-map':
        if getattr(args, 'cloud', False) and getattr(args, 'animate', False):
            print("\n[ATLAS EXCEPTION] The '--cloud' and '--animate' flags are mutually exclusive.")
            sys.exit(1)    

    try:
        thermo_db = load_thermo_database(args.db_file)
        
        if args.status:
            print(f"\n{'='*85}\n{'Molecule Name':<30} | {'T_m (K)':<10} | {'dH_fus (J/mol)':<15} | {'Orcacosmo File'}\n{'='*85}")
            for name, props in sorted(thermo_db.items()):
                expected_file = os.path.join(args.orcacosmo_dir, f"{name}_c000.orcacosmo")
                display_name = name.title().replace('_', ' ')
                file_status = "\033[92m[FOUND]\033[0m" if os.path.exists(expected_file) else "\033[91m[MISSING]\033[0m"
                print(f"{display_name:<30} | {props['T_m']:<10.2f} | {props['dH_fus']:<15.2f} | {file_status}")
            print(f"{'='*85}\n")
            return

        if not args.command:
            parser.print_help()
            return

        if args.command == 'fingerprint':
            run_fingerprint_cli(args, thermo_db)
        elif args.command == 'sigma-map':
            run_render_cli(args, thermo_db)
        elif args.command == 'match':
            run_match_cli(args, thermo_db)
            
        elif args.command in ['binary', 'ternary', 'solubility', 'logp', 'extract']:
            crs = COSMORS(par='default_orca')
            crs.par.calculate_contact_statistics_molecule_properties = True

            hba_file = os.path.join(args.orcacosmo_dir, args.hba + '_c000.orcacosmo')
            hbd_files = [os.path.join(args.orcacosmo_dir, hbd + '_c000.orcacosmo') for hbd in args.hbd]
            hba_key, hba_disp, tm_hba, hf_hba = get_molecule_info(hba_file, thermo_db)
            
            if args.command in ['binary', 'ternary', 'solubility']:
                if tm_hba is None or not os.path.exists(hba_file):
                    raise AtlasDatabaseError(f"Solute '{hba_key}' missing from database or ORCA folder.")
            else:
                if not os.path.exists(hba_file):
                    raise AtlasDatabaseError(f"Solute '{hba_key}' ORCA file missing.")

            if args.command == 'binary':
                run_binary_cli(args, thermo_db, crs, hba_file, hbd_files, hba_key, hba_disp, tm_hba, hf_hba)
            elif args.command == 'ternary':
                run_ternary_cli(args, thermo_db, crs, hba_file, hbd_files, hba_key, hba_disp, tm_hba, hf_hba)
            elif args.command == 'solubility':
                run_solubility_cli(args, thermo_db, crs, hba_file, hbd_files, hba_key, hba_disp, tm_hba, hf_hba)
            elif args.command == 'logp':
                run_logp_cli(args, thermo_db, crs, hba_file, hbd_files, hba_key, hba_disp)
            elif args.command == 'extract':
                run_extraction_cli(args, thermo_db, crs, hba_file, hbd_files, hba_key, hba_disp)

        if not args.silent:
            quote, speaker = random.choice(ATLAS_QUOTES)
            print(f"\n{'-'*85}\nATLAS v{__version__} Execution Completed in {time.time() - start_time:.2f} seconds.")
            print(f"\"{quote}\"\n — {speaker}\n{'-'*85}\n")

    except AtlasError as e:
        print(f"\n[ATLAS EXCEPTION] {e}\nExecution terminated safely.")
        sys.exit(1)

if __name__ == "__main__":
    main()
