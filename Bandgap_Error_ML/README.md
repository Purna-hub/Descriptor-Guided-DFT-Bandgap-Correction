# Descriptor-Guided ML Correction of DFT Band Gap Underestimation

## Installation

python -m venv venv


source venv/bin/activate  (Linux/Mac)


venv\Scripts\activate     (Windows)

pip install -r requirements.txt

## Execution

Place dataset in:
data/raw/bandgap_dataset.csv

Run:

python src/main.py

Outputs:
- figures/Figure1_Parity_Residual.png
- figures/Figure3_SHAP_summary.png
- results/shap_feature_importance.csv
