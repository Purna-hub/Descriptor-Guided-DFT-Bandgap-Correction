# Descriptor-Guided-DFT-Bandgap-Correction Using Machine Learning
Interpretable Machine Learning Modelling of DFT Band-Gap Underestimation from Composition-Derived Descriptors


========================================================


This repository contains the code and processed dataset supporting the manuscript submitted to Materials Letters.

Study summary:

2928 experimentally validated compounds

Modeling of semilocal DFT band-gap underestimation

Nonlinear ensemble modeling (XGBoost)

MAE reduced from ~1.1 eV to ~0.52 eV

SHAP-based interpretability

REPRODUCIBILITY

Python version: 3.10

Install dependencies:

pip install -r requirements.txt

Run full pipeline:

python src/main.py

This will generate all tables and figures automatically in the results/ directory.

DATA SOURCES

Experimental band gaps: Juho dataset
DFT band gaps: Materials Project database

The repository includes the processed dataset used for modeling.
No external API calls or internet access are required.

DATA AND CODE AVAILABILITY

The dataset and source code will be made publicly available upon acceptance of the manuscript.
This repository is provided for peer-review reproducibility.

========================================================
