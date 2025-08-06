# Metadata Bias Analysis and Mitigation

This project focuses on the detection and reduction of bias in metadata related to chest pathology diagnosis. The data includes metadata for 15,000 individuals. The work involves analyzing demographic and clinical attributes, identifying sources of statistical unfairness, and implementing correction strategies to build fairer predictive models.

## 1. Dataset Overview

- Input file: `JEMLI_ZEINEB.csv`
- After cleaning, columns such as `Image Index` and unnamed placeholders were removed.
- Binary labeling was applied: `0` for "No Finding", `1` for presence of pathology.
- Categorical features include `Patient Gender`, `View Position`, and discretized `Patient Age`.

## 2. Exploratory and Statistical Analysis

- Univariate and bivariate analysis to detect bias based on gender, age, and view position.
- Construction of an `Age Category` variable to group patients by age.
- Visualization of class distributions across categorical variables.
- Correlation analysis of numeric features with pathology label.

## 3. Bias Identification

- The dataset shows moderate imbalance:
  - Patients aged 50–70 are overrepresented.
  - Slight disparity in label distribution across gender and view positions.
- Age and gender show measurable differences in label proportions.

## 4. Model Training and Bias Measurement

- Trained a Decision Tree classifier on encoded features.
- Feature importance analysis highlights potential bias sources.
- Bias metrics computed:
  - **Disparate Impact**
  - **Demographic Parity**
  - **Fairness metrics per group (TPR, FPR, PPR)**

Findings:
- The trained model introduces bias against women and against patients aged 50–70.
- Calibration and odds ratios confirm these discrepancies.

## 5. Fairness Mitigation Techniques

- **Instance Reweighting** based on age category to balance group influence.
- **Disparate Impact Remover** strategy: trained models without sensitive attributes.
- **PCA-based representation learning** to reduce bias in latent space.
  
Each method was evaluated with fairness metrics and classification accuracy.

## 6. Conclusion

The analysis confirms that predictive models can amplify existing bias. Several mitigation techniques were applied, including sample reweighting and attribute removal. Among them, removing sensitive attributes and using latent representations via PCA led to a better trade-off between accuracy and fairness.

All code is implemented in a modular and reproducible Jupyter notebook.

## Dependencies

- Python 3
- pandas, numpy, matplotlib, seaborn
- scikit-learn, plotl
