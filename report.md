# Refactored Differential Privacy Analysis

## Overview

This report documents improvements made to the original differential‑privacy notebook.  The analysis retains the core objective—examining how adding privacy‑preserving noise affects machine‑learning models trained on a health‑insurance dataset—but introduces a modular pipeline, cleaner code and more rigorous evaluation.

The refactored code is implemented in `dp_refactored.py`.  It defines reusable functions for loading and preprocessing the data, applying noise mechanisms, anonymising quasi‑identifiers, training models and computing both predictive and fairness metrics.

## Dataset and Preprocessing

The dataset contains attributes like age, sex, BMI, number of children, smoker status, region and charges.  Missing values are dropped.  A helper function `preprocess_data` uses a `ColumnTransformer` to standardise numeric features and one‑hot encode categorical features.  The target (`smoker`) is binarised.

An additional function `anonymise_dataset` groups continuous variables into coarse categories and binarises the `children` attribute.  This approach follows k‑anonymity principles: individuals are indistinguishable from at least *k* others with respect to quasi‑identifiers【12286000783039†L272-L314】.  By ensuring a minimum group size and grouping variables, the data also facilitates l‑diversity and t‑closeness: it encourages variety in sensitive attributes within each group【12286000783039†L325-L384】【12286000783039†L390-L401】.

## Differential‑Privacy Noise Mechanisms

The module defines several functions for adding noise:

- **Laplace noise**: drawn from a Laplacian distribution scaled by sensitivity/ε, suitable for ε‑differential privacy.
- **Gaussian noise**: implements the analytic Gaussian mechanism, adding noise proportional to \(\sqrt{\log (1.25/\delta)}\) per ε【12286000783039†L272-L314】.
- **Exponential and geometric noise**: provide alternatives for count data.
- **Randomised response**: replaces a categorical value with a random alternative with probability \(1-p\).  This protects discrete sensitive attributes.

Each noise function operates only on numeric columns and returns a new DataFrame, leaving the original unchanged.  Randomised response is applied to categorical attributes such as `sex` and `region`.

## Modelling and Evaluation

The pipeline creates several variants of the dataset: original, Laplace‑noised, Gaussian‑noised, exponential‑noised, geometric‑noised, anonymised and randomised‑response.  For each variant it:

1. **Preprocesses** the features and target.
2. **Splits** the data into stratified training and test sets.
3. **Balances** the training set using SMOTE to address class imbalance.
4. **Trains three models**:
   - Support Vector Machine with a small hyper‑parameter grid search.
   - Shallow Decision Tree.
   - Simple feed‑forward Neural Network with two hidden layers.
5. **Evaluates** the models using accuracy and two fairness metrics:
   - *Demographic parity difference*: measures the difference in positive prediction rates between sensitive groups【163128011541164†L115-L123】.
   - *Equalised odds difference*: measures disparities in true‑positive and false‑positive rates.  
   Sensitive features are derived from one of the one‑hot encoded `sex` dummies.

Results are collected into a dictionary keyed by dataset and model name.  When the script is run via `python dp_refactored.py --data insurance.csv`, it prints each model’s accuracy along with its demographic‑parity and equalised‑odds differences.

## Improvements over the Original Notebook

1. **Modularity and Reuse** – Functions encapsulate each task (noise addition, anonymisation, preprocessing, training and evaluation).  This reduces duplication and clarifies data flow.
2. **Command‑line Interface** – The script can be executed from the terminal with a dataset argument, making it suitable for automation (e.g., CI pipelines).
3. **Fairness Evaluation** – Fairness metrics are systematically computed for every model/dataset combination, not just one model.  Demographic‑parity and equalised‑odds differences capture independence and separation notions of fairness【163128011541164†L115-L123】.
4. **Anonymisation Rationale** – The anonymisation function generalises quasi‑identifiers and enforces minimum group sizes to meet k‑anonymity and encourages diversity and closeness in the sensitive attribute【12286000783039†L272-L314】【12286000783039†L325-L384】【12286000783039†L390-L401】.
5. **Simplicity** – Explanatory comments and docstrings have been added to clarify the purpose of each function.  Unused code and duplicate imports have been removed.

## Usage in GitHub

Add this report and the `dp_refactored.py` script to your repository.  Update the GitHub Actions workflow to run the refactored script instead of the notebook if you wish to automate testing of the refactored pipeline.  You can still execute the original notebook for exploratory analysis, but the refactored code is better suited for reuse and integration.

## Conclusion

The refactored pipeline preserves the experimental goals of the original project—studying the effects of differential‑privacy mechanisms on machine‑learning models—while improving code organisation and evaluation depth.  Fairness metrics are computed consistently, and the anonymisation approach is grounded in established privacy definitions.  This structure will make it easier to extend the work, integrate into continuous‑integration workflows and share with collaborators.