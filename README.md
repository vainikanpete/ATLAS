# ATLAS: Advanced Thermodynamic Liquid & Aqueous Solver

**Version:** 1.1.2  
**Author:** Petteri Vainikka, PhD  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18826724.svg)](https://doi.org/10.5281/zenodo.18826724)

**ATLAS** is a high-performance computational suite for evaluating complex thermodynamic phase equilibria, solubility limits, molecular screening-charge descriptors, partitioning, and mixed-phase transfer/extraction efficiencies. Built as a specialized routing engine on top of the Conductor-like Screening Model for Real Solvents (COSMO-RS), ATLAS translates molecular-level screening charge densities ($\sigma$-profiles) directly into macroscopic thermodynamic properties.

Version **1.1.2** updates the COSMO-RS backend to the `openCOSMORS24a` parameterization, extends partition and extraction calculations with optional concentration-basis volume corrections, restores the command-line extraction controller, and expands the Streamlit interface with a dedicated mixed-phase extraction tab.

ATLAS is designed for multicomponent systems, including molecular solvent mixtures, Deep Eutectic Solvents (DES), and other structured liquid phases. It features custom numerical solvers for Solid-Liquid Equilibrium (SLE) phase boundaries, solubility curves, partition coefficients, liquid-liquid transfer calculations, high-fidelity 3D topological $\sigma$-maps, and ASIS molecular matching.

---

## What is new in v1.1.2

### COSMO-RS backend update

ATLAS now initializes openCOSMO-RS using:

```python
COSMORS(par=openCOSMORS24a())
```

instead of the previous `default_orca` parameter string.
The new parameterization is based on: https://doi.org/10.1016/j.fluid.2024.114250

### Mole-fraction and concentration-basis partitioning

Partition calculations now distinguish native mole-fraction-basis quantities from optional concentration-basis quantities:

| Quantity | Meaning |
|---|---|
| `logP_x` | Native mole-fraction-basis partition coefficient |
| `dG_x` | Native mole-fraction-basis transfer free energy |
| `logP_c` | Concentration-basis partition coefficient, available when a volume correction is supplied |
| `dG_c` | Concentration-basis transfer free energy, available when a volume correction is supplied |

ATLAS first computes the native COSMO-RS mole-fraction-basis result:

```math
\log P_x = \frac{\ln \gamma_A - \ln \gamma_B}{\ln 10}
```

When a volume quotient is supplied, ATLAS applies:

```math
\log P_c = \log P_x - \log_{10}\left(\frac{V_B}{V_A}\right)
```

and:

```math
\Delta G_c = \Delta G_x + RT \ln\left(\frac{V_B}{V_A}\right)
```

### Volume-correction models for `logp`

The `logp` command now supports several mutually exclusive volume-correction modes:

| Mode | CLI arguments |
|---|---|
| Direct volume quotient | `--vol_q` / `--volume_quotient` |
| Effective molar volumes | `--mvol_a` and `--mvol_b` |
| Molecular weights and densities | `--mw_a`, `--rho_a`, `--mw_b`, `--rho_b` |
| Octanol/water preset | `--ow_298` |

The octanol/water preset assumes:

```text
Phase A = water
Phase B = octanol
T = 298.15 K
V_B / V_A = 8.72
```

### Generalized extraction / mixed-phase transfer

The extraction engine now reports the same native and concentration-corrected thermodynamic quantities:

| Quantity | Meaning |
|---|---|
| `logP_x` | Native mole-fraction-basis transfer coefficient from source phase to target mixture |
| `dG_x` | Native mole-fraction-basis transfer free energy |
| `logP_c` | Concentration-basis transfer coefficient, when a volume correction is supplied |
| `dG_c` | Concentration-basis transfer free energy, when a volume correction is supplied |

Although the CLI command is still named `extract`, the calculation is not restricted to water or DES systems. It evaluates transfer from a selected source phase into a two-component target phase at a defined target mixture ratio.

### Streamlit interface update

The graphical interface now includes:

- a dedicated **Extraction** tab,
- optional concentration-basis correction controls for `logP` and extraction,
- explanatory “More information” panels for mole-fraction vs concentration-basis quantities,
- live telemetry backfilling for the new `Extraction` category,
- v1.1.2 backend initialization through `openCOSMORS24a`,
- sidebar attribution links for openCOSMO-RS and ORCA.

---

## Repository Structure

ATLAS separates the execution engine, graphical interface, thermodynamic constants, quantum screening profiles, and static interface assets.

```text
ATLAS/
├── atlas.py                  # Main ATLAS execution engine and CLI
├── app.py                    # Streamlit graphical web interface
├── requirements.txt
├── assets/
│   ├── logo.png              # ATLAS logo
│   ├── tuhh_logo.png         # openCOSMO-RS attribution logo
│   └── orca_logo.png         # ORCA attribution logo
├── database/
│   └── thermo_db.dat         # Validated physical constants (Tm, dH_fus)
└── molecules/
    ├── h2o_c000.orcacosmo    # .orcacosmo quantum screening profiles
    └── ...                   # Molecular screening-profile database
```

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/vainikanpete/ATLAS.git
cd ATLAS
pip install -r requirements.txt
```

ATLAS requires pre-computed `.orcacosmo` files generated via ORCA 6 or an equivalent compatible workflow.

---

## Graphical Web Interface

For rapid visual screening and prototyping without the command line, ATLAS includes a fully featured Streamlit interface with strict concurrency locking for safer multi-user deployments.

```bash
streamlit run app.py
```

The web interface supports:

- Binary SLE phase diagrams,
- Ternary SLE phase contours,
- solubility calculations,
- logP / transfer thermodynamics,
- mixed-phase extraction / transfer calculations,
- $\sigma$-profile fingerprints,
- 3D topological $\sigma$-maps,
- ASIS molecular matching,
- molecule parameterization request generation.

---

## CLI Usage Examples

ATLAS operates via a subparser-driven command-line interface.

### 1. Binary Solid-Liquid Equilibrium (SLE)

Generate a temperature-composition phase diagram for a binary system:

```bash
python atlas.py binary --hba thymol --hbd menthol
```

### 2. Ternary Phase Contours

Generate liquidus contours across a discrete ternary simplex grid:

```bash
python atlas.py ternary --hba thymol --hbd menthol h2o
```

### 3. Solubility in a Mixed Solvent

Calculate solubility across a binary solvent-ratio sweep at fixed temperature:

```bash
python atlas.py solubility --hba trifluoroacetic_acid --hbd thymol menthol --temp 298.15
```

Append solubility results to a CSV file:

```bash
python atlas.py solubility --hba trifluoroacetic_acid --hbd thymol menthol --temp 298.15 --csv screening_results.csv
```

### 4. Partition Coefficient and Transfer Free Energy

Calculate native mole-fraction-basis partition thermodynamics:

```bash
python atlas.py logp --hba aniline --hbd h2o octanol --temp 298.15
```

Apply the built-in octanol/water concentration-basis correction:

```bash
python atlas.py logp --hba aniline --hbd h2o octanol --temp 298.15 --ow_298
```

Use a direct volume quotient:

```bash
python atlas.py logp --hba aniline --hbd solvent_a solvent_b --temp 298.15 --vol_q 8.72
```

Use effective molar volumes:

```bash
python atlas.py logp --hba aniline --hbd solvent_a solvent_b --temp 298.15 --mvol_a 0.018 --mvol_b 0.157
```

Use molecular weights and densities for pure liquid phases:

```bash
python atlas.py logp --hba aniline --hbd h2o octanol --temp 298.15 \
  --mw_a 18.015 --rho_a 997.0 \
  --mw_b 130.23 --rho_b 827.0
```

### 5. Mixed-Phase Extraction / Transfer

Calculate transfer of a solute from a source phase into a two-component target phase:

```bash
python atlas.py extract --hba aniline --hbd h2o menthol thymol --ratio 1 2
```

Apply a direct concentration-basis volume correction:

```bash
python atlas.py extract --hba aniline --hbd h2o menthol thymol --ratio 1 2 --vol_q 9.1
```

Use effective source and target molar volumes:

```bash
python atlas.py extract --hba aniline --hbd h2o menthol thymol --ratio 1 2 --mvol_a 0.018 --mvol_b 0.165
```

In the `extract` command, the three `--hbd` entries are interpreted as:

```text
source_phase target_component_A target_component_B
```

The target mixture composition is controlled by `--ratio`.

### 6. High-Fidelity Topological $\sigma$-Maps

Generate a publication-quality, color-mapped 3D screening density projection directly from quantum segment data:

```bash
python atlas.py sigma-map --mol thymol
```

Generate a rotating GIF:

```bash
python atlas.py sigma-map --mol thymol --animate
```

Render a translucent ESP volumetric cloud:

```bash
python atlas.py sigma-map --mol thymol --cloud
```

### 7. ASIS Similarity Matcher

Evaluate the continuous Szymkiewicz-Simpson structural overlap of a target molecule against benchmark archetypes:

```bash
python atlas.py match --target hexanoic_acid --benchmarks thymol menthol
```

### 8. Database Status

Check database coverage and `.orcacosmo` file availability:

```bash
python atlas.py --status
```

---

## CSV Output

Several engines support CSV export through `--csv`.

### `logp` CSV schema

```text
Solute,System,T_sys_K,dG_x,logP_x,dG_c,logP_c
```

When no concentration-basis correction is applied, `dG_c` and `logP_c` are left blank.

### `extract` CSV schema

```text
Solute,System,T_sys_K,Target_Ratio,dG_x,logP_x,dG_c,logP_c
```

When no concentration-basis correction is applied, `dG_c` and `logP_c` are left blank.

---

## Key Engine Features

- **openCOSMORS24a backend initialization:** ATLAS v1.1.2 initializes openCOSMO-RS with the `openCOSMORS24a` parameterization.
- **Mole-fraction and concentration-basis partitioning:** Native `logP_x` / `dG_x` outputs can be supplemented with volume-corrected `logP_c` / `dG_c` outputs.
- **Generalized mixed-phase transfer:** The extraction engine supports transfer from any source phase into a two-component target phase, not only water-to-DES cases.
- **Singularity rejection:** The solubility algorithm rejects mathematical singularities caused by highly associative mixtures where the continuum model breaks down, preventing silent backend failures.
- **Thread thrashing mitigation:** ATLAS forces controlled thread usage for small iterative thermodynamic matrices, preventing OpenBLAS/OMP oversubscription.
- **Precision-enforced unity:** ATLAS applies dynamic bit-level mole-fraction correction to enforce strict unity in composition vectors.
- **Miscibility gap masking:** Highly repulsive systems are safely handled through a thermal floor in the extended Schröder-van Laar extrapolation.
- **High-fidelity $\sigma$-maps:** ORCA geometries and COSMO segment data are converted into 2D/3D molecular screening-charge visualizations.
- **ASIS molecular matching:** Continuous $\sigma$-profile overlap is used to compare molecules against selected benchmark archetypes.
- **Streamlit graphical interface:** Web-based workflows expose the main engines for interactive screening and prototyping.

---

## Theoretical Notes and Limitations

ATLAS reports the mathematical consequences of the underlying quantum continuum models without empirical fitting.

Strongly associating systems, ionic systems, and structured liquid phases may show large deviations from experimental behavior when the continuum model omits effects such as:

- heat-capacity differences between solid and liquid phases,
- explicit ion-pairing or speciation penalties,
- composition-dependent phase volumes,
- structural reorganization of complex liquid networks,
- non-ideal mixture density behavior.

For concentration-basis partitioning and extraction calculations, the quality of `logP_c` and `dG_c` depends directly on the quality of the supplied phase-volume model. For mixtures, DES-like systems, ionic liquids, or strongly non-ideal phases, experimentally measured or otherwise justified effective molar volumes are preferred over pure-component estimates.

---

## References

ATLAS would not be possible were it not for ORCA, openCOSMO-RS, and the visualization tools it builds upon. If you use ATLAS in your research, please cite and acknowledge the following works.

**ORCA**

> Neese, F. (2012). The ORCA program system. *WIRES Computational Molecular Science*, 2(1), 73–78. doi:10.1002/wcms.81

**openCOSMO-RS**

> Gerlach, T., Müller, S., de Castilla, A. G., & Smirnova, I. (2022). An open source COSMO-RS implementation and parameterization supporting the efficient implementation of multiple segment descriptors. *Fluid Phase Equilibria*, 560, 113472. doi:10.1016/j.fluid.2022.113472

> GitHub: [TUHH-TVT/openCOSMO-RS_py](https://github.com/TUHH-TVT/openCOSMO-RS_py)

**xyzrender Visualization Engine**

> Goodfellow, A. S., & Nguyen, B. N. (2026). Graph-Based Internal Coordinate Analysis for Transition State Characterization. *Journal of Chemical Theory and Computation*. doi:10.1021/acs.jctc.5c02073

> GitHub: [aligfellow/xyzrender](https://github.com/aligfellow/xyzrender)

---

## License

ATLAS is distributed under the GNU General Public License v3.0 (GPLv3). See the `LICENSE` file for more information.
