# Drug Molecular Structure and Topological Index Calculator

This project provides a Streamlit application for parsing a SMILES string, validating it, generating 2D and 3D molecular structures, calculating degree-based topological indices from the hydrogen-suppressed molecular graph, and retrieving available physicochemical properties from documented PubChem REST APIs.

## Scientific calculation convention

- Heavy atoms are treated as vertices.
- Bonds are treated as edges.
- Bond order is ignored during index calculation.
- Explicit hydrogens are removed before graph-based calculations.
- The 3D conformer is generated separately and does not affect the graph-based index values.

## Features

- Optional drug-name input
- SMILES validation and canonical SMILES generation
- 2D structure rendering with a white background
- Interactive 3D structure rendering with py3Dmol
- Molecular descriptors and graph metrics
- 16 degree-based topological indices
- Edge-degree distribution table
- Excel workbook export with multiple worksheets
- 2D PNG and 3D SDF downloads
- Best-effort PubChem physicochemical-property retrieval with source records and verification status
- RDKit InChI/InChIKey generation and clearly-labelled calculated MolLogP/MolMR values

## Screenshots

Placeholder for screenshots.

## Installation on Windows

### Option 1: Miniforge / Anaconda

```bash
conda env create -f environment.yml
conda activate drug-topology
```

### Option 2: pip

```bash
pip install -r requirements.txt
```

## VS Code interpreter selection

In VS Code, select the interpreter from the created environment:

- Windows: `Python: Select Interpreter`
- Choose the environment at `drug_topology_app/.venv` if using the local virtual environment

## Run the application

```bash
streamlit run app.py
```

## Run tests

```bash
pytest -v
```

## Deploy to Streamlit Community Cloud

This repository is ready for deployment with `app.py` as its entry point.

1. Create a GitHub repository and push this `drug_topology_app` directory to it.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Select **Create app**, choose the repository and branch, and set the entry-point file to `app.py`.
4. Choose a subdomain such as `drug-topology-calculator` and deploy.

Community Cloud installs the dependencies from `requirements.txt`. The app remains usable when PubChem or EPA CompTox is unavailable; those requests are optional and time-limited.

## Example input

Drug name: Atenolol

SMILES:

```text
CC(C)NCC(COC1=CC=C(C=C1)CC(=O)N)O
```

## Expected Atenolol validation values

- Molecular formula: `C14H22N2O3`
- Edge-degree distribution:
  - `(1,3)` = 5
  - `(2,2)` = 4
  - `(2,3)` = 10
  - Total edges = 19
- Topological indices:
  - `M1 = 86`
  - `M2 = 91`
  - `F = 212`
  - `R = 8.9692`
  - `ABC = 13.9820`
  - `GA = 18.1281`
  - `H = 8.5000`
  - `HM = 394`
  - `SCI = 8.9721`
  - `ISI = 19.7500`
  - `SO = 63.1806`
  - `RSO = 38.0175`
  - `MSO = 5.7689`
  - `NSO = 3.3253`
  - `DPSO = 309.0221`
  - `DSSO = 288.7780`

## Excel report

The Excel workbook contains the following worksheets:

- Molecule Summary
- Topological Indices
- Edge Distribution
- Structure
- 3D Coordinates
- Physicochemical Properties
- Raw Property Records
- Data Sources

## Troubleshooting

- If RDKit is missing, install it via Conda Forge.
- If the 3D viewer fails, the app will still show the 2D structure and indices.
- If Streamlit does not detect your environment, restart VS Code after selecting the interpreter.

## Limitations

- PubChem lookup results depend on source availability and are never fabricated; optional external retrieval can be unavailable.
- 3D generation is heuristic and may fail for unusual structures.
- Physicochemical properties beyond the requested descriptors are not inferred from external models.

## Disclaimer

These descriptors are provided for academic and research use. They are calculated locally from the supplied structure and should not be interpreted as experimentally measured values.
