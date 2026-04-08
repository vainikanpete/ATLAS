import os
import tempfile
import shutil

# ==========================================
# 0. HARDCORE THREAD CLAMPS (MUST BE BEFORE NUMPY)
# ==========================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import random
import base64
import threading
import json
import streamlit as st
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.colors as mcolors
import numpy as np

from atlas import (
    load_thermo_database, get_molecule_info, get_ideal_curve, solve_real_curve,
    solve_ternary_curve, solve_pure_solubility, solve_solubility_curve, solve_logp_and_dg,
    parse_cosmo_segments, generate_sigma_profile, calculate_fingerprint, calculate_asis,
    parse_xyz_block, parse_cosmo_segments_3d, map_sigma_to_atoms, render_publication_surface, 
    ATLAS_STYLE, ATLAS_QUOTES, __version__
)
from opencosmorspy import COSMORS

# ==========================================
# 0.1 CLOUD-SAFE TEMP DIR & PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def get_temp_dir():
    """Create a persistent temp folder that works on Streamlit Cloud"""
    tmp = tempfile.mkdtemp(prefix="atlas_")
    return tmp

TEMP_DIR = get_temp_dir()

# ==========================================
# 0.2 COMPUTE LOCKS & TELEMETRY ENGINE
# ==========================================
@st.cache_resource
def get_compute_queue():
    # Allows exactly 20 concurrent threads to run math engines simultaneously.
    return threading.BoundedSemaphore(20)

@st.cache_resource
def get_analytics_lock():
    return threading.Lock()

compute_queue = get_compute_queue()
analytics_lock = get_analytics_lock()

# Safely route analytics to the Cloud's temp folder
ANALYTICS_FILE = os.path.join(tempfile.gettempdir(), 'atlas_analytics.json')

def init_analytics():
    if not os.path.exists(ANALYTICS_FILE):
        try:
            with open(ANALYTICS_FILE, 'w') as f:
                json.dump({"sessions": 0, "computations": {
                    "Binary": 0, "Ternary": 0, "Solubility": 0, "logP": 0, 
                    "Fingerprint": 0, "SigmaMap": 0, "ASIS": 0
                }}, f)
        except Exception:
            pass

def log_event(event_type, category=None):
    with analytics_lock:
        try:
            with open(ANALYTICS_FILE, 'r') as f:
                data = json.load(f)
            if event_type == "session":
                data["sessions"] += 1
            elif event_type == "compute" and category:
                data["computations"][category] += 1
            with open(ANALYTICS_FILE, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception:
            pass # Fail silently so analytics never crash the app

def get_analytics():
    with analytics_lock:
        try:
            with open(ANALYTICS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {"sessions": 0, "computations": {}}

init_analytics()

def apply_watermark(ax):
    ax.text(0.5, 0.5, f"ATLAS v{__version__}\n© 2026 P. Vainikka", 
            transform=ax.transAxes, fontsize=26, color='gray', 
            alpha=0.2, ha='center', va='center', rotation=25, zorder=0, fontweight='bold')

# ==========================================
# 0.3 Remove logging issues from SL UI
# ==========================================
# Silence the harmless 'missing ScriptRunContext' log spam
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)

# ==========================================
# 1. UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="ATLAS", page_icon="atlas.png", layout="wide")

# --- THE SESSION TRIPWIRE ---
if 'session_logged' not in st.session_state:
    st.session_state.session_logged = True
    log_event("session")
# ==========================================
# 2. MAIN UI & SIDEBAR
# ==========================================
with st.sidebar:
    try: st.image(os.path.join(BASE_DIR, "atlas.png"), width=120)
    except FileNotFoundError: pass
        
    st.markdown(f"## ATLAS v{__version__}")
    st.markdown("**Advanced Thermodynamic Liquid & Aqueous Solver**")
    st.markdown("© 2026 Petteri Vainikka, PhD\n\n*Licensed under GPL-3.0*")
    
    st.divider()
    stats = get_analytics()
    total_comps = sum(stats["computations"].values())
    st.markdown("### Live Analytics")
    st.markdown(f"**Unique Sessions:** `{stats['sessions']}`")
    st.markdown(f"**Computations Routed:** `{total_comps}`")
    
    st.divider()
    quote, speaker = random.choice(ATLAS_QUOTES)
    st.markdown(f"*{quote}*")
    st.markdown(f"— **{speaker}**")

st.title("ATLAS")
st.markdown("**Advanced Thermodynamic Liquid & Aqueous Solver**")
st.link_button("⭐ View Source & Documentation on GitHub", "https://github.com/vainikanpete/ATLAS")
st.markdown("---")

# ==========================================
# 3. DATABASE LOADING
# ==========================================
db_path = os.path.join(BASE_DIR, 'database', 'thermo_db.dat')
mol_dir = os.path.join(BASE_DIR, 'molecules')

@st.cache_data
def load_backend():
    db = load_thermo_database(db_path)
    available_mols = [f.replace('_c000.orcacosmo', '') for f in os.listdir(mol_dir) if f.endswith('.orcacosmo')]
    return db, sorted(available_mols)

try: db, available_mols = load_backend()
except Exception as e:
    st.error(f"System Error: Cannot load database. {e}")
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Binary SLE", "Ternary SLE", "Solubility", "logP (Transfer)", 
    "Sigma Fingerprint", "Sigma-Map", "ASIS Matcher", "Read Me", "Add Molecule"
])

# ------------------------------------------
# TAB 1: BINARY SLE
# ------------------------------------------
with tab1:
    st.markdown("Calculate Solid-Liquid Equilibrium for a Binary system.")
    with st.form("binary_inputs"):
        c1, c2 = st.columns(2)
        hba_bin = c1.selectbox("Select Solute (HBA)", available_mols, index=0, key="bin_hba")
        hbd_bin = c2.selectbox("Select Solvent (HBD)", available_mols, index=1 if len(available_mols)>1 else 0, key="bin_hbd")
        run_binary = st.form_submit_button("Compute Binary Phase Diagram")

    if run_binary:
        if compute_queue.acquire(blocking=False):
            try:
                log_event("compute", "Binary")
                with st.spinner(f"Solving thermodynamics for {hba_bin} + {hbd_bin}..."):
                    crs = COSMORS(par='default_orca')
                    crs.par.calculate_contact_statistics_molecule_properties = True
                    hba_path = os.path.join(mol_dir, f"{hba_bin}_c000.orcacosmo")
                    hbd_path = os.path.join(mol_dir, f"{hbd_bin}_c000.orcacosmo")
                    _, hba_disp, tm_hba, hf_hba = get_molecule_info(hba_path, db)
                    _, hbd_disp, tm_hbd, hf_hbd = get_molecule_info(hbd_path, db)
                    
                    x_id, T_id, eut_id_x, eut_id_T = get_ideal_curve(tm_hba, hf_hba, tm_hbd, hf_hbd)
                    x_real, T_real, gamma_hba, gamma_hbd = solve_real_curve(
                        crs, hba_path, hbd_path, tm_hba, hf_hba, tm_hbd, hf_hbd, hba_disp, hbd_disp
                    )
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    plt.style.use(ATLAS_STYLE)
                    ax.plot(x_id, T_id, 'k-.', alpha=0.5, label='Ideal SLE')
                    ax.plot(x_real, T_real, color='#e74c3c', lw=2.5, marker='o', label='ATLAS Prediction')
                    ax.set_xlim(0, 1.0)
                    ax.set_xlabel(f"Mole Fraction ({hba_disp})")
                    ax.set_ylabel("Temperature (K)")
                    ax.set_title(f"Phase Diagram: {hba_disp} + {hbd_disp}")
                    ax.grid(alpha=0.4)
                    ax.legend()
                    apply_watermark(ax)
                    st.pyplot(fig)
                    
                    valid_idx = np.where(~np.isnan(T_real))[0]
                    if len(valid_idx) > 0:
                        eut_real_idx = valid_idx[np.argmin(T_real[valid_idx])]
                        st.success("Mathematical convergence achieved.")
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Ideal Eutectic", f"{eut_id_T:.2f} K", f"x = {eut_id_x:.2f}", delta_color="off")
                        mc2.metric("COSMO Eutectic", f"{T_real[eut_real_idx]:.2f} K", f"x = {x_real[eut_real_idx]:.2f}", delta_color="off")
                        mc3.metric("Eutectic Depth", f"{T_real[eut_real_idx] - eut_id_T:.2f} K", "Departure from Ideality", delta_color="inverse")
            except Exception as e:
                st.error(f"Execution Failed: {e}")
            finally:
                compute_queue.release()
        else:
            st.error("**SERVER BUSY:** ATLAS is currently processing a computation for another user. Please wait a moment and try again.")

# ------------------------------------------
# TAB 2: TERNARY SLE
# ------------------------------------------
with tab2:
    st.markdown("Calculate the phase envelope for a multi-component Ternary system.")
    st.warning("**Execution Time:** This takes multiple minutes. Do not refresh.")
    
    with st.form("ternary_inputs"):
        c1, c2, c3 = st.columns(3)
        hba_tern = c1.selectbox("Select Solute (HBA)", available_mols, index=0, key="tern_hba")
        hbd1_tern = c2.selectbox("Select Solvent 1 (HBD)", available_mols, index=1 if len(available_mols)>1 else 0, key="tern_hbd1")
        hbd2_tern = c3.selectbox("Select Solvent 2 (HBD)", available_mols, index=2 if len(available_mols)>2 else 0, key="tern_hbd2")
        run_ternary = st.form_submit_button("Compute Ternary Simplex")

    if run_ternary:
        if compute_queue.acquire(blocking=False):
            try:
                log_event("compute", "Ternary")
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                def cb_ternary(current, total, msg):
                    progress_bar.progress(min(current / total, 1.0))
                    status_text.markdown(f"**Status:** {msg}...")
                    
                with st.spinner("Initializing 136-point ternary matrix..."):
                    crs = COSMORS(par='default_orca')
                    crs.par.calculate_contact_statistics_molecule_properties = True
                    paths = [os.path.join(mol_dir, f"{m}_c000.orcacosmo") for m in [hba_tern, hbd1_tern, hbd2_tern]]
                    props, names = [], []
                    for p in paths:
                        _, disp, tm, hf = get_molecule_info(p, db)
                        props.append((tm, hf))
                        names.append(disp)
                
                grid, T_real = solve_ternary_curve(crs, paths, props, names, progress_callback=cb_ternary)
                
                progress_bar.empty()
                status_text.empty()
                
                fig, ax = plt.subplots(figsize=(6, 5))
                plt.style.use(ATLAS_STYLE)
                X = grid[:, 1] + 0.5 * grid[:, 2]
                Y = grid[:, 2] * (np.sqrt(3.0) / 2.0)
                triang = tri.Triangulation(X, Y)
                mask = np.any(np.isnan(T_real[triang.triangles]), axis=1)
                triang.set_mask(mask)
                
                contour = ax.tricontourf(triang, T_real, levels=50, cmap='magma', extend='max')
                fig.colorbar(contour, ax=ax, label='Liquidus Temperature (K)', pad=0.12)
                ax.text(0.0, -0.06, names[0], ha='center', va='top', fontsize=11, weight='bold')
                ax.text(1.0, -0.06, names[1], ha='center', va='top', fontsize=11, weight='bold')
                ax.text(0.5, np.sqrt(3)/2 + 0.03, names[2], ha='center', va='bottom', fontsize=11, weight='bold')
                ax.axis('off')
                ax.set_title(f"Ternary Phase Diagram: {names[0]} System", pad=20)
                apply_watermark(ax)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Execution Failed: {e}")
            finally:
                compute_queue.release()
        else:
            st.error("**SERVER BUSY:** ATLAS is currently mapping a ternary grid for another user. This takes several minutes. Please try again later.")

# ------------------------------------------
# TAB 3: SOLUBILITY
# ------------------------------------------
with tab3:
    st.markdown("Calculate solubility limits in pure or mixed solvents.")
    
    sweep_mode = st.radio("Analysis Mode", ["Sweep Solvent Ratio (Fixed T)", "Sweep Temperature (Pure Solvent A)"])
    
    with st.form("sol_inputs"):
        c1, c2, c3 = st.columns(3)
        solute_sol = c1.selectbox("Select Solute", available_mols, index=0, key="sol_solute")
        solv1_sol = c2.selectbox("Solvent A", available_mols, index=1 if len(available_mols)>1 else 0, key="sol_a")
        
        solv2_sol = None
        temp_val = 298.15
        if sweep_mode == "Sweep Solvent Ratio (Fixed T)":
            solv2_sol = c3.selectbox("Solvent B", available_mols, index=2 if len(available_mols)>2 else 0, key="sol_b")
            temp_val = st.number_input("System Temperature (K)", value=298.15, step=1.0)
        else:
            c3.markdown("*(Solvent B ignored in Temp Sweep)*")
            col_t1, col_t2 = st.columns(2)
            t_start = col_t1.number_input("Start Temp (K)", value=280.0, step=5.0)
            t_end = col_t2.number_input("End Temp (K)", value=350.0, step=5.0)

        run_sol = st.form_submit_button("Compute Solubility")

    if run_sol:
        if compute_queue.acquire(blocking=False):
            try:
                log_event("compute", "Solubility")
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                def cb_solubility(current, total, msg):
                    progress_bar.progress(min(current / total, 1.0))
                    status_text.markdown(f"**Math Engine:** {msg}...")

                with st.spinner("Initializing quantum segments..."):
                    crs = COSMORS(par='default_orca')
                    crs.par.calculate_contact_statistics_molecule_properties = True
                    solute_path = os.path.join(mol_dir, f"{solute_sol}_c000.orcacosmo")
                    solv1_path = os.path.join(mol_dir, f"{solv1_sol}_c000.orcacosmo")
                    _, sol_disp, tm_s, hf_s = get_molecule_info(solute_path, db)
                    _, solv1_disp, _, _ = get_molecule_info(solv1_path, db)

                    fig, ax = plt.subplots(figsize=(6, 4))
                    plt.style.use(ATLAS_STYLE)

                if sweep_mode == "Sweep Solvent Ratio (Fixed T)":
                    solv2_path = os.path.join(mol_dir, f"{solv2_sol}_c000.orcacosmo")
                    _, solv2_disp, _, _ = get_molecule_info(solv2_path, db)
                    
                    x_solvent, y_solubility, ideal_sol = solve_solubility_curve(
                        crs, solute_path, solv1_path, solv2_path, tm_s, hf_s, 
                        [sol_disp, solv1_disp, solv2_disp], temp_val, progress_callback=cb_solubility
                    )
                    
                    ax.axhline(ideal_sol, color='k', linestyle='-.', alpha=0.5, label='Ideal')
                    ax.plot(x_solvent, y_solubility, color='#2ecc71', lw=2.5, marker='o', label='Real')
                    ax.set_yscale('log')
                    ax.set_xlabel(f'Mole Fraction of {solv1_disp} in Solvent Mixture')
                    ax.set_title(f"Solubility in {solv1_disp} + {solv2_disp} at {temp_val} K")
                else:
                    T_range = np.linspace(t_start, t_end, 12)
                    solubilities, ideal_sols = [], []
                    total_t_steps = len(T_range)
                    
                    for idx, T in enumerate(T_range):
                        cb_solubility(idx, total_t_steps, f"Evaluating Temp: {T:.2f} K")
                        s = solve_pure_solubility(crs, solute_path, solv1_path, tm_s, hf_s, [sol_disp, solv1_disp], T)
                        k_id = (hf_s / 8.3145) * (1.0 / tm_s - 1.0 / T)
                        ideal_sols.append(min(np.exp(k_id), 0.9999))
                        solubilities.append(s)
                    cb_solubility(total_t_steps, total_t_steps, "Solubility grid complete!")

                    ax.plot(T_range, ideal_sols, 'k-.', alpha=0.5, label='Ideal')
                    ax.plot(T_range, solubilities, color='#3498db', lw=2.5, marker='o', label='Real')
                    ax.set_yscale('log')
                    ax.set_xlabel('Temperature (K)')
                    ax.set_title(f"Temperature Sweep: {sol_disp} in pure {solv1_disp}")

                progress_bar.empty()
                status_text.empty()

                ax.set_ylabel(f'Solubility of {sol_disp} ($x_S$)')
                ax.grid(alpha=0.4, which='both')
                ax.legend(frameon=True, fancybox=True, shadow=True)
                apply_watermark(ax)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Execution Failed: {e}")
            finally:
                compute_queue.release()
        else:
            st.error("**SERVER BUSY:** ATLAS is currently running another user's job. Please wait.")

# ------------------------------------------
# TAB 4: logP (TRANSFER)
# ------------------------------------------
with tab4:
    st.markdown("Calculate partition coefficients (logPx) and Transfer Free Energy (ΔG).")
    with st.form("logp_inputs"):
        c1, c2, c3 = st.columns(3)
        solute_lp = c1.selectbox("Select Solute", available_mols, index=0, key="lp_sol")
        phase1 = c2.selectbox("Phase 1 (Origin)", available_mols, index=available_mols.index('h2o') if 'h2o' in available_mols else 0, key="lp_p1")
        phase2 = c3.selectbox("Phase 2 (Destination)", available_mols, index=available_mols.index('octanol') if 'octanol' in available_mols else 1, key="lp_p2")
        sweep_mode_lp = st.radio("Evaluation Mode", ["Single Temperature", "Temperature Sweep"])
        
        if sweep_mode_lp == "Single Temperature":
            t_lp = st.number_input("System Temperature (K)", value=298.15, step=1.0)
        else:
            col_t1, col_t2 = st.columns(2)
            t_start_lp = col_t1.number_input("Start Temp (K)", value=280.0, step=5.0)
            t_end_lp = col_t2.number_input("End Temp (K)", value=350.0, step=5.0)

        run_logp = st.form_submit_button("Compute Partition Coefficient")

    if run_logp:
        if compute_queue.acquire(blocking=False):
            try:
                log_event("compute", "logP")
                with st.spinner("Computing transfer thermodynamics..."):
                    crs = COSMORS(par='default_orca')
                    crs.par.calculate_contact_statistics_molecule_properties = True
                    sol_path = os.path.join(mol_dir, f"{solute_lp}_c000.orcacosmo")
                    p1_path = os.path.join(mol_dir, f"{phase1}_c000.orcacosmo")
                    p2_path = os.path.join(mol_dir, f"{phase2}_c000.orcacosmo")
                    _, sol_disp, _, _ = get_molecule_info(sol_path, db)
                    _, p1_disp, _, _ = get_molecule_info(p1_path, db)
                    _, p2_disp, _, _ = get_molecule_info(p2_path, db)

                    if sweep_mode_lp == "Single Temperature":
                        dG, logPx = solve_logp_and_dg(crs, sol_path, p1_path, p2_path, [sol_disp, p1_disp, p2_disp], t_lp)
                        st.success("Calculation Complete.")
                        m1, m2 = st.columns(2)
                        m1.metric(f"logPx ({p1_disp} → {p2_disp})", f"{logPx:.2f}")
                        m2.metric(f"ΔG Transfer", f"{dG:.2f} kJ/mol")
                    else:
                        T_range = np.linspace(t_start_lp, t_end_lp, 10)
                        logps = []
                        for T in T_range:
                            dG, logPx = solve_logp_and_dg(crs, sol_path, p1_path, p2_path, [sol_disp, p1_disp, p2_disp], T)
                            logps.append(logPx)
                            
                        fig, ax = plt.subplots(figsize=(6, 4))
                        plt.style.use(ATLAS_STYLE)
                        ax.plot(T_range, logps, color='#9b59b6', lw=2.5, marker='s')
                        ax.set_xlabel('Temperature (K)')
                        ax.set_ylabel(f'logPx ({p1_disp} → {p2_disp})')
                        ax.set_title(f"Partition Coefficient T-Sweep: {sol_disp}")
                        ax.grid(alpha=0.4)
                        apply_watermark(ax)
                        st.pyplot(fig)
            except Exception as e:
                st.error(f"Execution Failed: {e}")
            finally:
                compute_queue.release()
        else:
            st.error("**SERVER BUSY:** ATLAS is currently running another computation. Please try again in a few seconds.")

# ------------------------------------------
# TAB 5: FINGERPRINT SCANNER
# ------------------------------------------
with tab5:
    st.markdown("Scan a molecule to reveal its underlying COSMO-RS charge distribution.")
    with st.form("fp_inputs"):
        target_mol = st.selectbox("Select Molecule", available_mols, index=0, key="fp_target")
        run_fp = st.form_submit_button("Scan Fingerprint")
        
    if run_fp:
        if compute_queue.acquire(blocking=False):
            try:
                log_event("compute", "Fingerprint")
                with st.spinner("Parsing segments..."):
                    target_path = os.path.join(mol_dir, f"{target_mol}_c000.orcacosmo")
                    _, disp_name, _, _ = get_molecule_info(target_path, db)
                    areas, sigmas = parse_cosmo_segments(target_path)
                    sigma_grid, p_sigma = generate_sigma_profile(areas, sigmas)
                    fp = calculate_fingerprint(sigma_grid, p_sigma)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("H-Bond Donor (Blue)", f"{fp['HBD_Area']:.2f} Å²")
                    c2.metric("Non-Polar (Grey)", f"{fp['NonPolar_Area']:.2f} Å²")
                    c3.metric("H-Bond Acceptor (Red)", f"{fp['HBA_Area']:.2f} Å²")

                    fig, ax = plt.subplots(figsize=(7, 4))
                    plt.style.use(ATLAS_STYLE)
                    ax.plot(sigma_grid, p_sigma, color='black', lw=1.5)
                    ax.fill_between(sigma_grid, p_sigma, where=(sigma_grid < -0.0084), color='royalblue', alpha=0.5, label='HB Donor')
                    ax.fill_between(sigma_grid, p_sigma, where=((sigma_grid >= -0.0084) & (sigma_grid <= 0.0084)), color='gray', alpha=0.3, label='Non-Polar')
                    ax.fill_between(sigma_grid, p_sigma, where=(sigma_grid > 0.0084), color='crimson', alpha=0.5, label='HB Acceptor')
                    ax.axvline(-0.0084, color='k', linestyle='--', lw=1, alpha=0.5)
                    ax.axvline(0.0084, color='k', linestyle='--', lw=1, alpha=0.5)
                    ax.set_xlim(-0.03, 0.03)
                    ax.set_ylim(0, np.max(p_sigma) * 1.15)
                    
                    ax.set_xlabel(r'Screening Charge Density, $\sigma$ ($e$/Å$^2$)')
                    ax.set_ylabel(r'Area, $P(\sigma)$ (Å$^2$)')
                    ax.set_title(rf"$\sigma$-Profile Fingerprint: {disp_name}")
                    ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
                    ax.grid(alpha=0.3)
                    
                    apply_watermark(ax)
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Scanner Failed: {e}")
            finally:
                compute_queue.release()
        else:
            st.error("**SERVER BUSY:** Please wait.")

# ------------------------------------------
# TAB 6: SIGMA-MAP
# ------------------------------------------
with tab6:
    st.markdown(r"Generate a high-fidelity 3D Topological $\sigma$-Map directly from quantum segment data.")
    with st.form("map_inputs"):
        map_mol = st.selectbox("Select Molecule", available_mols, index=0, key="map_mol")
        
        c1, c2 = st.columns(2)
        use_cloud = c1.checkbox("Draw Volumetric ESP Cloud", value=False)
        use_animate = c2.checkbox("Generate Animated GIF", value=False)
        
        # We need a mock args object to pass to render_publication_surface
        class MockArgs:
            def __init__(self, cloud, animate):
                self.cloud = cloud
                self.animate = animate
                self.sigma_max = 0.03
                self.hbd_cutoff = -0.0084
                self.hba_cutoff = 0.0084
                
        run_map = st.form_submit_button(r"Generate 3D $\sigma$-Map")
        
    if run_map:
        if use_cloud and use_animate:
            st.error("**ATLAS EXCEPTION:** The ESP Cloud and Animation flags are mutually exclusive. Please select only one.")
        elif compute_queue.acquire(blocking=False):
            try:
                log_event("compute", "SigmaMap")
                status_msg = "Generating High-Fidelity 3D Render..." if not use_animate else "Rendering 80-Frame Animation (This will take a minute)..."
                with st.spinner(status_msg):
                    target_path = os.path.join(mol_dir, f"{map_mol}_c000.orcacosmo")
                    _, disp_name, _, _ = get_molecule_info(target_path, db)
                    
                    elements, atom_coords_3d = parse_xyz_block(target_path)
                    areas, sigmas, surf_coords = parse_cosmo_segments_3d(target_path)
                    atom_sigmas = map_sigma_to_atoms(atom_coords_3d, surf_coords, areas, sigmas)
                    
                    args = MockArgs(use_cloud, use_animate)
                    out_ext = ".gif" if use_animate else ".png"
                    
                    # Safely route the output to our persistent temp directory
                    out_file = os.path.join(TEMP_DIR, f"temp_render_{map_mol}{out_ext}")
                    
                    success = render_publication_surface(
                        elements, atom_coords_3d, atom_sigmas, 
                        out_file, disp_name, args, animate=use_animate
                    )
                    
                    if success and os.path.exists(out_file):
                        st.image(out_file, width="stretch")
                        # Clean up
                        try: os.remove(out_file)
                        except: pass
                    else:
                        st.error("Rendering failed. Ensure xyzrender is installed correctly.")
                        
            except Exception as e:
                st.error(f"Render Failed: {e}")
            finally:
                compute_queue.release()
        else:
            st.error("**SERVER BUSY:** Please wait.")

# ------------------------------------------
# TAB 7: ASIS MATCHER
# ------------------------------------------
with tab7:
    st.markdown("Calculate Szymkiewicz-Simpson structural overlap against benchmark archetypes.")
    with st.form("asis_inputs"):
        c1, c2 = st.columns([1, 2])
        target_asis = c1.selectbox("Target Molecule", available_mols, index=0, key="asis_target")
        benchmarks = c2.multiselect("Select Benchmarks", available_mols, default=[available_mols[1]] if len(available_mols)>1 else [])
        run_asis = st.form_submit_button("Run ASIS Evaluation")

    if run_asis and benchmarks:
        if compute_queue.acquire(blocking=False):
            try:
                log_event("compute", "ASIS")
                with st.spinner("Scoring continuous overlap..."):
                    t_path = os.path.join(mol_dir, f"{target_asis}_c000.orcacosmo")
                    _, t_disp, _, _ = get_molecule_info(t_path, db)
                    a_t, s_t = parse_cosmo_segments(t_path)
                    grid_t, p_target = generate_sigma_profile(a_t, s_t)
                    
                    for bench in benchmarks:
                        b_path = os.path.join(mol_dir, f"{bench}_c000.orcacosmo")
                        _, b_disp, _, _ = get_molecule_info(b_path, db)
                        a_b, s_b = parse_cosmo_segments(b_path)
                        _, p_bench = generate_sigma_profile(a_b, s_b)
                        res = calculate_asis(grid_t, p_target, p_bench)
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("HBD Overlap Contribution", f"{res['HBD_Contribution']:.1f}%")
                        m2.metric("NP Overlap", f"{res['NP_Contribution']:.1f}%")
                        m3.metric("HBA Overlap", f"{res['HBA_Contribution']:.1f}%")
                        
                        fig, ax = plt.subplots(figsize=(7, 3.5))
                        plt.style.use(ATLAS_STYLE)
                        ax.plot(grid_t, res['norm_bench'], color='gray', lw=2, linestyle='--', label=f"{b_disp} (Bench)")
                        ax.plot(grid_t, res['norm_target'], color='black', lw=2, label=f"{t_disp} (Target)")
                        ax.fill_between(grid_t, res['intersection'], color='mediumpurple', alpha=0.4, label=rf"ASIS Overlap ({res['Global_ASIS']:.1f}\%)")
                        ax.set_xlim(-0.03, 0.03)
                        
                        ax.set_xlabel(r'Screening Charge Density, $\sigma$ ($e$/Å$^2$)')
                        ax.set_ylabel('Probability Density')
                        ax.set_title(f"ASIS Matching: {t_disp} vs {b_disp}")
                        ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
                        ax.grid(alpha=0.3)
                        
                        apply_watermark(ax)
                        st.pyplot(fig)
            except Exception as e:
                st.error(f"Evaluation Failed: {e}")
            finally:
                compute_queue.release()
        else:
            st.error("**SERVER BUSY:** Please wait.")

# ------------------------------------------
# TAB 8: READ ME / DOCUMENTATION
# ------------------------------------------
with tab8:
    st.markdown(r"""
    ### Welcome to ATLAS
    **ATLAS (Advanced Thermodynamic Liquid & Aqueous Solver)** acts as an intelligent, high-performance routing engine, bridging the gap between quantum theory and applied laboratory screening.

    By translating molecular-level screening charge densities ($\sigma$-profiles) directly into macroscopic thermodynamic properties, ATLAS enables rapid evaluation of complex phase equilibria, solubility limits, and liquid-liquid extraction efficiencies.

    #### Graphical Web Interface
    This Streamlit interface serves as an accessible visual dashboard for:
    * **Phase Diagrams:** Autonomous Solid-Liquid Equilibrium (SLE) generation for Binary and Ternary mixtures with Trust-Region clamping.
    * **Solubility & Partitioning:** Rapid evaluation of solubility limits and partition coefficients (logP).
    * **Molecular Analysis:** Native parsing of ORCA geometries to generate integrated HBD/NP/HBA capacity profiles and evaluate continuous ASIS structural overlaps.
    * **3D Topological $\sigma$-Maps:** High-fidelity 3D rendering of screening charge densities directly from quantum segments.

    #### The Command-Line Engine (CLI)
    For advanced use-cases, the ATLAS Command-Line Interface offers significantly more power. If you require multi-component mixed DES extraction limits, high-throughput batch screening (CSV outputs), hyper-parameter tuning, or automated GIF generation, please utilize the CLI by pulling the full repository.

    #### Known Theoretical Limitations
    Users evaluating strongly associating Deep Eutectic Solvents (e.g., Choline Chloride + Urea) will note that standard continuum models accurately converge to the mathematical root of the liquidus equation (often ~200 K), which is significantly deeper than the experimental eutectic (~305 K). 
    
    Standard SLE models omit the heat capacity difference ($\Delta C_p$) between solid and liquid states and struggle to capture the complex speciation penalty of separating ion pairs in the melt. ATLAS faithfully computes and reports the unvarnished mathematical reality of the underlying quantum continuum models without arbitrary empirical fitting.

    #### Further reading
    For more details on the underlying openCOSMO-RS implementation, see the [official ORCA documentation](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/solvationmodels.html?q=openCOSMO&n=0#opencosmo-rs).

    ---

    #### References
    If you use ATLAS in your research, please formally cite the following underlying technologies:

    **The ORCA Program System**
    > Neese, F. (2012). The ORCA program system. *WIRES Comput. Molec. Sci.*, 2(1), 73-78. doi:10.1002/wcms.81
    
    **openCOSMO-RS Implementation**
    > Gerlach, T., Müller, S., de Castilla, A. G., & Smirnova, I. (2022). An open source COSMO-RS implementation and parameterization supporting the efficient implementation of multiple segment descriptors. *Fluid Phase Equilibria*, 560, 113472. doi:10.1016/j.fluid.2022.113472

    > GitHub: [TUHH-TVT/openCOSMO-RS_py](https://github.com/TUHH-TVT/openCOSMO-RS_py)
    
    **xyzrender Visualization Engine**
    > Goodfellow, A.S., & Nguyen, B.N. (2026). Graph-Based Internal Coordinate Analysis for Transition State Characterization. *J. Chem. Theory Comput.* doi:10.1021/acs.jctc.5c02073

    > GitHub: [aligfellow/xyzrender](https://github.com/aligfellow/xyzrender)
    """)





# ------------------------------------------
# TAB 9: ADD MOLECULE
# ------------------------------------------
with tab9:
    st.markdown("### Want *your* molecule(s) in ATLAS?")
    st.markdown("""
    The ATLAS database is constantly expanding. If you require specific compounds for your research, you can request them to be parameterized via quantum chemistry (ORCA) and added to the official public repository. 
    """)
    
    st.divider()
    
    with st.form("molecule_request_form"):
        st.markdown("**Submit a Parameterization Request**")
        
        c1, c2 = st.columns([3, 1])
        mol_name = c1.text_input("Full Chemical Name", placeholder="e.g., L-Menthol")
        mol_charge = c2.number_input("Formal Charge", value=0, step=1)
        
        smiles = st.text_input("Canonical SMILES", placeholder="e.g., CC1CCC(C(C1)O)C(C)C")
        
        st.markdown("##### Thermodynamic Constants (Optional but Recommended)")
        st.markdown("*Note: Purely ionic components do not require exact melting thermodynamics. Leaving these at 0.0 restricts the compound to logP, ASIS, and Solubility engines only.*")
        
        c3, c4 = st.columns(2)
        t_melt = c3.number_input(r"Melting Temperature, $T_{melt}$ (K)", value=0.0, step=0.1)
        dh_fus = c4.number_input(r"Enthalpy of Fusion, $\Delta H_{fus}$ (kJ/mol)", value=0.0, step=0.1)
        
        submit_req = st.form_submit_button("Generate Request")
        
if submit_req:
        # 1. Check for empty fields
        if not mol_name or not smiles:
            st.error("⚠️ Please provide at least the Chemical Name and Canonical SMILES.")
        else:
            # 2. Check the existing database
            safe_name = mol_name.strip().lower().replace(' ', '_').replace('-', '_')
            
            if any(safe_name in mol for mol in available_mols):
                st.warning(f"🛑 **Already Exists:** A molecule closely matching '{mol_name}' appears to already be in the ATLAS database. Please check the dropdown menus.")
            else:
                # 3. RDKit SMILES Sanity Check
                smiles_is_valid = True
                try:
                    from rdkit import Chem
                    mol = Chem.MolFromSmiles(smiles.strip())
                    if mol is None:
                        smiles_is_valid = False
                except ImportError:
                    pass # Bypass if RDKit fails to load on cloud
                
                if not smiles_is_valid:
                    st.error("🛑 **Invalid SMILES:** The RDKit backend could not parse the provided string. Please check for syntax or valency errors.")
                else:
                    # 4. Generate the thing
                    body_text = f"""ATLAS Parameterization Request:

Name: {mol_name}
Charge: {mol_charge}
SMILES: {smiles}
T_melt (K): {t_melt if t_melt > 0 else 'N/A'}
dH_fus (kJ/mol): {dh_fus if dh_fus > 0 else 'N/A'}

Additional Notes:
"""
                    st.success("✅ Valid SMILES detected. Request generated successfully!")
                    
                    target_email = "vainikanpete@gmail.com"
                    
                    st.markdown(f"**Step 1:** Copy the request data below.")
                    st.code(body_text, language="text")
                    
                    st.markdown(f"**Step 2:** Email the data to the developer.")
                    
                    # We keep the mailto link, but ONLY for the subject line. 
                    # This ensures it opens the email client without breaking.
                    import urllib.parse
                    safe_subject = urllib.parse.quote(f"ATLAS Molecule Request: {mol_name}")
                    mailto_url = f"mailto:{target_email}?subject={safe_subject}"
                    
                    html_btn = f'''
                    <a href="{mailto_url}" style="text-decoration:none;">
                        <div style="background-color:#FF4B4B; color:white; padding:10px 20px; border-radius:5px; text-align:center; font-weight:bold; display:inline-block; margin-top:10px;">
                            ✉️ Open Email Client
                        </div>
                    </a>
                    '''
                    st.markdown(html_btn, unsafe_allow_html=True)
                    st.markdown("*Just paste the copied data into the body of the email!*")
