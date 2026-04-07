# ATLAS: Advanced Thermodynamic Liquid & Aqueous Solver

**Version:** 1.1.1  
**Author:** Petteri Vainikka, PhD  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18826724.svg)](https://doi.org/10.5281/zenodo.18826724)

**ATLAS** is a high-performance computational suite designed to evaluate complex thermodynamic phase equilibria and liquid-liquid extraction efficiencies. Built as a specialized routing engine upon the Conductor-like Screening Model for Real Solvents (COSMO-RS), ATLAS translates molecular-level screening charge densities ($\sigma$-profiles) directly into macroscopic thermodynamic properties.

With version 1.1.1, ATLAS has expanded beyond pure thermodynamics to include **High-Fidelity 3D Topological $\sigma$-Maps**, **ASIS Molecular Matching**, and a fully threaded **Streamlit Web Interface** for rapid prototyping. It is specifically engineered for multicomponent systems, including Deep Eutectic Solvents (DES), and features custom numerical solvers for Solid-Liquid Equilibrium (SLE) phase boundaries, solubility curves, and Liquid-Liquid Extraction (LLE) partition coefficients.

## Repository Structure

To ensure maximum stability, ATLAS separates quantum screening profiles from the physical thermodynamic constants database.

```text
ATLAS/
├── atlas.py               # The main ATLAS execution engine (CLI)
├── app.py                 # The Streamlit Web User Interface
├── requirements.txt
├── database/
│   └── thermo_db.dat      # Validated physical constants (Tm, dH_fus)
└── molecules/
    ├── h2o_c000.orcacosmo # .orcacosmo quantum screening profiles
    └── ...                # Database expanded to 82 molecules
```

## Installation

Clone the repository and install the required dependencies (including the TUHH openCOSMO-RS backend directly from GitHub):

```bash
git clone https://github.com/vainikanpete/ATLAS.git
cd ATLAS
pip install -r requirements.txt
```

*Note: ATLAS requires pre-computed `.orcacosmo` files generated via ORCA 6 (or equivalent) to function.*

## Graphical Web Interface (New in v1.1.0)

For rapid visual screening and prototyping without the command line, ATLAS now includes a fully featured web interface. It includes strict concurrency locking for safe multi-user deployments.

```bash
streamlit run app.py
```

## CLI Usage Examples

ATLAS operates via a robust, subparser-driven Command Line Interface (CLI). Below are examples of its core engines.

### 1\. Liquid-Liquid Extraction (LLE) into a DES

Calculate the partition coefficient (logP) of a solute transferring from an aqueous phase into a 1:2 molar ratio Deep Eutectic Solvent:

```bash
python atlas.py extract --hba perfluorohexanoic_acid --hbd h2o choline_chloride urea --ratio 1 2
```

### 2\. Binary Solid-Liquid Equilibrium (SLE)

Generate the complete temperature-composition phase diagram for a binary system:

```bash
python atlas.py binary --hba thymol --hbd menthol
```

### 3\. Ternary Phase Contours

Generate high-resolution liquidus contours across a discrete two-dimensional simplex grid for a 3-component system:

```bash
python atlas.py ternary --hba thymol --hbd menthol h2o
```

### 4\. High-Throughput Batch Output

Append solubility results to a CSV file for programmatic batch screening:

```bash
python atlas.py solubility --hba trifluoroacetic_acid --hbd thymol menthol --temp 298.15 --csv screening_results.csv
```

### 5\. High-Fidelity Topological $\sigma$-Maps (New)

Generate publication-quality, color-mapped 3D screening density projections directly from quantum segment data:

```bash
python atlas.py sigma-map --mol thymol
```

*(Add `--animate` to generate a rotating GIF, or `--cloud` to render the translucent ESP volumetric cloud).*

### 6\. ASIS Similarity Matcher (New)

Evaluate the continuous Szymkiewicz-Simpson structural overlap of a target molecule against a list of benchmark archetypes:

```bash
python atlas.py match --target hexanoic_acid --benchmarks thymol menthol
```

## Key Engine Features

  * **Singularity Rejection (Updated):** The solubility algorithm now actively rejects mathematical singularities caused by highly associative mixtures (e.g., Polyols + Urea) where the continuum model breaks down, preventing silent failures occurring in the COSMO-RS backend.
  * **Thread Thrashing Mitigation:** Forces single-core sequential execution on small combinatorial matrices, yielding order-of-magnitude speedups for iterative thermodynamic loops.
  * **Precision-Enforced Unity:** Implements dynamic "bit-nudging" to enforce strict mathematical unity (down to 1e-15) on mole fraction arrays, preventing backend floating-point crashes.
  * **Miscibility Gap Masking:** Safely handles highly repulsive systems by applying an adaptive thermal floor to the extended Schröder-van Laar extrapolation.
  * **Liquid-Liquid Extraction Engine:** Bypasses solid-state data gaps by simulating infinite-dilution partitioning into multicomponent, structured DES environments to calculate logP and transfer free energies.

## References

ATLAS would not be possible were it not for ORCA and openCOSMO-RS. 
If you use ATLAS in your research, please cite and pay your respects to the following works:

**ORCA**

> Neese, F. (2012). The ORCA program system. *WIRES Comput. Molec. Sci.*, 2(1), 73-78. doi:10.1002/wcms.81

**openCOSMO-RS**

> Gerlach, T., Müller, S., de Castilla, A. G., & Smirnova, I. (2022). An open source COSMO-RS implementation and parameterization supporting the efficient implementation of multiple segment descriptors. *Fluid Phase Equilibria*, 560, 113472. doi:10.1016/j.fluid.2022.113472

> GitHub: [TUHH-TVT/openCOSMO-RS\_py](https://github.com/TUHH-TVT/openCOSMO-RS_py)

**xyzrender Visualization Engine**
> Goodfellow, A.S., & Nguyen, B.N. (2026). Graph-Based Internal Coordinate Analysis for Transition State Characterization. *J. Chem. Theory Comput.* doi:10.1021/acs.jctc.5c02073

> GitHub: [aligfellow/xyzrender](https://github.com/aligfellow/xyzrender)

## License

ATLAS is distributed under the GNU General Public License v3.0 (GPLv3). See the `LICENSE` file for more information.

