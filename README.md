# ATLAS: Advanced Thermodynamic Liquid & Aqueous Solver 

**Version:** 1.0.4  
**Author:** Petteri Vainikka, PhD  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18826724.svg)](https://doi.org/10.5281/zenodo.18826724)

**ATLAS** is a high-performance computational suite designed to evaluate complex thermodynamic phase equilibria and liquid-liquid extraction efficiencies. Built as a specialized routing engine upon the Conductor-like Screening Model for Real Solvents (COSMO-RS), ATLAS translates molecular-level screening charge densities ($\sigma$-profiles) directly into macroscopic thermodynamic properties.

It is specifically engineered for multicomponent systems, including Deep Eutectic Solvents (DES), and features custom numerical solvers for Solid-Liquid Equilibrium (SLE) phase boundaries, solubility curves, and Liquid-Liquid Extraction (LLE) partition coefficients.

## Repository Structure

To ensure maximum stability, ATLAS separates quantum screening profiles from the physical thermodynamic constants database.

```text
ATLAS/
├── atlas.py               # The main ATLAS execution engine
├── requirements.txt
├── database/
│   └── thermo_db.dat      # Validated physical constants (Tm, dH_fus)
└── molecules/
    ├── h2o_c000.orcacosmo # .orcacosmo quantum screening profiles
    └── ... 
```

## Installation

Clone the repository and install the required dependencies (including the TUHH openCOSMO-RS backend directly from GitHub):

```bash
git clone [https://github.com/vainikanpete/ATLAS.git](https://github.com/vainkanpete/ATLAS.git)
cd ATLAS
pip install -r requirements.txt
```

*Note: ATLAS requires pre-computed `.orcacosmo` files generated via ORCA 6 (or equivalent) to function.*

## Usage Examples

ATLAS operates via a multi-route Command Line Interface (CLI). Below are examples of its core engines.

### 1. Liquid-Liquid Extraction (LLE) into a DES
Calculate the partition coefficient (logP) of a solute transferring from an aqueous phase into a 1:2 molar ratio Deep Eutectic Solvent:

```bash
python atlas.py --extract --hba perfluorohexanoic_acid --hbd h2o choline_chloride urea --ratio 1 2
```

### 2. Binary Solid-Liquid Equilibrium (SLE)
Generate the complete temperature-composition phase diagram for a binary system:

```bash
python atlas.py --hba thymol --hbd menthol
```

### 3. Ternary Phase Contours
Generate high-resolution liquidus contours across a discrete two-dimensional simplex grid for a 3-component system:

```bash
python atlas.py --tern --hba thymol --hbd menthol h2o
```

### 4. High-Throughput Batch Output
Append results to a CSV file for programmatic batch screening:

```bash
python atlas.py --sol --hba trifluoroacetic_acid --hbd thymol menthol --temp 298.15 --csv screening_results.csv
```

## Other Features

* **Thread Thrashing Mitigation:** Forces single-core sequential execution on small combinatorial matrices, yielding order-of-magnitude speedups for iterative thermodynamic loops.
* **Precision-Enforced Unity:** Implements dynamic "bit-nudging" to enforce strict mathematical unity (down to 1e-16) on mole fraction arrays, preventing C++ backend floating-point crashes.
* **Miscibility Gap Masking:** Safely handles highly repulsive systems by applying an adaptive thermal floor to the extended Schröder-van Laar extrapolation.
* **Liquid-Liquid Extraction Engine:** Bypasses solid-state data gaps by simulating infinite-dilution partitioning into multicomponent, structured DES environments to calculate logP and transfer free energies.


## License
ATLAS is distributed under the GNU General Public License v3.0 (GPLv3). See the `LICENSE` file for more information.
