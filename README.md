# A1
Back Propagation
Overview
This project focuses on predicting turbine performance using machine learning models, specifically a Backpropagation Neural Network (BP) and a benchmark Linear Regression model (MLR). The implementation is in Python, leveraging libraries such as pandas, numpy, scikit-learn, and matplotlib.

![Uploading slide_231.jpg…]()

Table of Contents
Description of Implementation
Execution Instructions
Selected Dataset
Implementation Decisions
Discussion and Results
Conclusions
Description of Implementation
The project employs two models, BP and MLR, for turbine performance prediction. Python is used, and data preprocessing includes handling missing values, categorical value representation, outlier detection, and data normalization for consistent scales.

Execution Instructions
Ensure the required libraries (pandas, numpy, scikit-learn, matplotlib) are installed. Run the code in a Python environment.

Selected Dataset
The Fitbit Fitness Tracker Dataset from Kaggle is used for this project. Two additional datasets, A1-synthetic and A1-turbine, are also employed. Data normalization techniques, such as Min-Max scaling, are applied for effective model training.

Implementation Decisions
Neural Network (BP)
Architecture: Input layer, hidden layer (4 neurons), output layer.
Activation Function: Sigmoid in the hidden layer.
Loss Function: Mean Absolute Percentage Error (MAPE).
Training: 1000 epochs, learning rate of 0.01.
Linear Regression (MLR)
Model: Utilizes scikit-learn's LinearRegression.
Evaluation Metric: Mean Absolute Error (MAE).
Discussion and Results
Both models are evaluated on synthetic and turbine datasets. MAPE and MAE values, along with scatter plots, provide insights into their performance. MLR outperforms BP on the synthetic dataset, while challenges are observed in turbine dataset predictions for both models.

Conclusions
The project highlights the importance of aligning model architecture with dataset characteristics. Further exploration of neural network architecture, parameter tuning, and dataset characteristics is recommended. Future work should focus on advanced models, ensemble methods, dataset enhancement, and systematic hyperparameter tuning. Special thanks to Kaggle for providing the Fitbit Fitness Tracker Dataset.

