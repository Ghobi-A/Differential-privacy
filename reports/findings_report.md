# Findings Report

## Dataset characteristics
The insurance dataset contains **1,338 rows** and **7 columns**. Features are mixed-type: **3 categorical (object) columns**, **2 integer columns**, and **2 floating-point columns**. The target class **smoker** is imbalanced: **1,064 “no” (79.52%)** and **274 “yes” (20.48%)**. These proportions establish a substantial majority class that any evaluation must consider when interpreting ROC-AUC outcomes.

## Baseline performance
Two baseline classifiers were trained with standard preprocessing (scaling numeric features and one-hot encoding categorical features). ROC-AUC on the test set is:
- **SVM:** 0.994793
- **Decision tree:** 0.917883

The SVM establishes a near-ceiling reference point for discrimination, while the decision tree provides a weaker but still strong baseline.

## Failure of feature-level Laplace/Gaussian noise
Feature-level perturbation was applied to clipped numeric features using Laplace (ε-DP) and Gaussian ((ε, δ)-DP) mechanisms across ε ∈ {0.01, 0.0398, 0.158, 0.631, 2.512, 10}. The observed ranges show a steep deterioration at low ε values:

- **Laplace (SVM):** ROC-AUC spans **0.4329 → 0.9780**.
- **Laplace (Decision tree):** ROC-AUC spans **0.3842 → 0.8853**.
- **Gaussian (SVM):** ROC-AUC spans **0.4732 → 0.8586**.
- **Gaussian (Decision tree):** ROC-AUC spans **0.4774 → 0.8267**.

At the most stringent privacy levels (e.g., ε = 0.01), Laplace perturbation yields ROC-AUC around **0.43–0.46**, which is effectively at or below chance. This constitutes a practical failure for feature-level noise at low ε: the injected noise overwhelms the clipped numeric signal and erodes discriminative structure. The same degradation pattern appears for Gaussian noise, with minimum ROC-AUC near **0.47**, again close to random performance. These results demonstrate that feature-level DP at tight ε values is not viable for maintaining baseline utility in this setting.

## Why Bernoulli / exponential / geometric noise were excluded
The notebook’s experimental scope explicitly restricts feature-level perturbation to **Laplace and Gaussian mechanisms** and compares them to **DP-SGD training-time privacy**. No outputs are reported for alternative distributions such as Bernoulli, exponential, or geometric noise. Accordingly, these mechanisms were excluded to keep the experiments limited to (i) continuous feature perturbations with standard DP mechanisms and (ii) training-time privacy via DP-SGD, all evaluated uniformly through ROC-AUC.

## DP-SGD results and interpretation
DP-SGD was run with Opacus using noise multipliers {0.5, 1.0, 1.5, 2.0}. The privacy accountant reports ε values and ROC-AUC as follows:

| Noise multiplier | ε (δ = 1e-5) | ROC-AUC |
| --- | --- | --- |
| 0.5 | 32.0865 | 0.993171 |
| 1.0 | 6.3774 | 0.992659 |
| 1.5 | 3.1909 | 0.992574 |
| 2.0 | 2.1290 | 0.994025 |

Across a wide ε range (≈2.13–32.09), **ROC-AUC remains effectively constant around 0.993–0.994**. This indicates that, within the tested configuration, DP-SGD delivers strong privacy guarantees with minimal degradation relative to the SVM baseline. The results also show the expected privacy-utility trade-off in terms of ε: larger noise multipliers yield smaller ε, while utility remains stable in this dataset and architecture.

## Fairness discussion
Fairness metrics were computed on the baseline SVM using **sex** as the protected attribute and a threshold that maximizes Youden’s J statistic. The results are:
- **Demographic parity (DP) difference:** 0.133690
- **Equalized odds (EO) difference:** 0.023996

A DP difference of **0.1337** indicates a 13.37 percentage-point gap in positive prediction rates between groups. The EO difference of **0.0240** indicates a smaller gap in error-rate parity (bounded by differences in TPR and FPR). These values show that, even with strong baseline discrimination, group-level disparities persist and should be considered when selecting operating thresholds or evaluating downstream policy impacts.

## Final conclusion
The dataset is moderately imbalanced and supports high baseline discrimination, particularly for SVM. Feature-level Laplace and Gaussian perturbation collapses utility at low ε, with ROC-AUC near chance, making tight ε feature-noise impractical in this setting. In contrast, DP-SGD achieves **ε as low as ≈2.13** while preserving **ROC-AUC ≈0.993–0.994**, indicating a substantially better privacy–utility trade-off for this task. Fairness metrics reveal non-trivial demographic parity gaps despite strong predictive performance, underscoring the need to evaluate privacy, utility, and fairness jointly rather than in isolation.
