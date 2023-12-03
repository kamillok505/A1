# Classification with SVM, BP, and MLR

## Overview
This project focuses on performing data classification using three algorithms: Support Vector Machines (SVM), Back-Propagation (BP), and Multiple Linear Regression (MLR). The primary objective is to explore and analyze the performance of these algorithms on three distinct datasets. The chosen datasets include the Ring datasets, the Bank Marketing dataset, and a custom Search dataset.

## Description of Implementation
### Selected Datasets
1. **Ring Datasets:**
   - Training Set 1: ring-separable.txt
   - Training Set 2: ring-merged.txt
   - Test Set (common for both): ring-test.txt
   - Two input features + 1 class identifier (0/1)
   - Visualization of the data is crucial for understanding the classification task, especially due to the limited two input features.

2. **Bank Marketing Dataset:**
   - Data: bank-additional.csv or bank-additional-full.csv
   - Training: First 80% patterns
   - Test: Last 20% patterns
   - 20 features, mainly categorical, requiring proper representation as numerical data.
   - Prediction Feature: Subscription to a term deposit (yes/no)

3. **Search Dataset:**
   - At least 6 features, one for classification
   - Binary or multivariate classification feature
   - Randomly selected 80% for training/validation, 20% for testing, with shuffling to eliminate sorting bias.

### Execution Instructions
1. **Data Preprocessing:**
   - Checked for missing values
   - Applied one-hot encoding for categorical variables
   - Conducted outlier analysis
   - Applied data normalization based on feature distributions

2. **Classification Models:**
   - **SVM:** Explored LibSVM and other libraries in various languages. Explored kernel and parameter options.
   - **BP (Back-Propagation):** Used libraries from A1, experimented with architecture, learning rate, momentum, activation function, and epochs based on cross-validation results.
   - **MLP (Multiple Linear Regression):** Utilized preferred libraries, following A1 recommendations.

### Discussion and Results
#### Parameter Selection (Part 2.1):
   - Applied cross-validation for SVM, BP, and MLR.
   - Reported expected classification errors from cross-validation and compared them with test set errors.

#### Evaluation of Results (Part 2.2):
   - Computed classification errors on Test and Validation sets for SVM, BP, and MLR.
   - Computed and compared confusion matrices for all three algorithms on each dataset.
   - Calculated ROC curve and AUC for BP and MLR.

#### Visualization:
   - Utilized visualization techniques, representing patterns in different colors or panels for each class.

## Conclusions
The analysis and implementation revealed insights into the performance variations of SVM, BP, and MLR across diverse datasets. Proper data preprocessing and parameter tuning are crucial for achieving optimal results. The inclusion of a new dataset highlighted the significance of dataset-specific considerations.

## Repository Link
[GitHub Repository](#)

## References
S. Moro, P. Cortez, and P. Rita. "A Data-Driven Approach to Predict the Success of Bank Telemarketing." Decision Support Systems (2014), doi:10.1016/j.dss.2014.03.001.
