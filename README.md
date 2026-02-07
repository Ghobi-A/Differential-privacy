# Differential Privacy in Machine Learning

This repository demonstrates how to apply **differential privacy** techniques to a tabular health‑insurance dataset and measure the impact of privacy on downstream machine‑learning models.  It grew out of an academic project, but all institution specific content has been removed so that the work can be shared freely on GitHub.

For a deeper theoretical background on differential privacy concepts, see the
[findings report](reports/findings_report.md).

## Overview

The included Jupyter notebook (`notebooks/dp_privacy_insurance.ipynb`) walks
through the following steps:

1. **Data loading and exploration** –  We start with a sample insurance dataset containing demographic and health‑related features (age, sex, BMI, number of children, smoker status, region and charges).  Basic exploratory analysis is performed on numeric features.
2. **Noise injection** –  Laplace and Gaussian noise are added to the dataset at
   both random and fixed magnitudes to provide differential‑privacy guarantees.
   The noise functions accept a privacy budget ``epsilon`` (and optional
   ``sensitivity``), computing any required scale internally.
3. **Anonymisation** –  We anonymise certain quasi‑identifiers by grouping continuous features into buckets (e.g. converting raw age into age groups) and binarising the number of children.  This improves privacy through *k‑anonymity*, *l‑diversity* and *t‑closeness* without destroying utility.
4. **Machine‑learning models** –  Three classifiers are trained on the original and noised datasets: Support Vector Machines, a basic Neural Network and a Decision Tree.  The goal is to understand how privacy noise affects predictive performance.
5. **Fairness metrics** –  For each trained model we compute demographic‑parity and equal‑opportunity metrics to see how the addition of noise impacts fairness across sensitive groups.

The notebook is fully executable and makes no assumptions about prior infrastructure—open it in JupyterLab or VS Code and run the cells from top to bottom.

## Quickstart

1. **Clone this repository** and change into the directory:

   ```bash
   git clone https://github.com/your‑username/your‑repo.git
   cd your‑repo
   ```

2. **Install the project** in editable mode so that its modules and CLI are available system‑wide:

   ```bash
   pip install -e .
   ```

   This installs the package along with the required dependencies.

3. **Add the dataset**.  The notebook expects a CSV file named
   `data/insurance.csv` in the repository root.  You can supply your own dataset
   or download a public insurance dataset such as the one from the Kaggle
   medical‑cost dataset.

4. **Launch Jupyter** and run the notebook:

   ```bash
   jupyter notebook notebooks/dp_privacy_insurance.ipynb
   ```

Alternatively, you can execute the notebook from the command line using `nbconvert`:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/dp_privacy_insurance.ipynb
```

## Privacy Model and Guarantees

This project uses additive Laplace and Gaussian noise as standard
differential-privacy mechanisms. The threat model assumes an attacker with
auxiliary information who attempts re-identification or attribute inference
from released data or model outputs. Under the stated sensitivity and epsilon
choices, the Laplace/Gaussian mechanisms provide formal DP guarantees. Other
anonymisation steps in this repo (e.g., k-anonymity, l-diversity, t-closeness)
are heuristic, non-DP baselines and should not be treated as formal
privacy guarantees.

## Pipeline entry point

Reusable pipeline components live in `src/dp/pipeline.py`. These functions load
the dataset, split it into train/test partitions, and build a preprocessing
transformer that is fit only on the training data to avoid leakage.

## Continuous integration

For convenience, this repository includes a GitHub Actions workflow that will
run the notebook on every push and pull request.  It sets up a Python
environment, installs the dependencies and executes
`notebooks/dp_privacy_insurance.ipynb` using `nbconvert`.  If the notebook runs
successfully, the workflow will finish with a green check mark.  You can find
the workflow definition in `.github/workflows/run‑notebook.yml`.

## Removing private materials

The original academic report (PDF) and any institution‑specific content are intentionally **not** tracked in this repository.  The `.gitignore` file excludes the uploaded dissertation to avoid committing it.  If you add your own private files (e.g. PDFs, datasets with personal information), update `.gitignore` accordingly.

## License

This project is licensed under the [MIT License](LICENSE).
