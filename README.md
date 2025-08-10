# Hypertension-Prediction-Optimized-ML-DL-Model-Comparison

This project implements and compares three distinct approaches for binary hypertension classification:

1. Random Forest Classifier (95% Accuracy)
   GridSearchCV optimized with 5-fold cross-validation
   Best parameters: max_depth=20, n_estimators=100
   Full parameter grid tested for reproducibility

2. Sequential Neural Network (90.7% Accuracy)
   Architecture: 5 hidden layers (64-128-64-32-1) with ReLU activation
   Regularization: L2 weight decay (λ=0.01) + Dropout (30-40%)
   Training: Adam optimizer (lr=0.001) with early stopping
   Preprocessing: StandardScaler normalization

3. Functional API Neural Network (88.2% Accuracy)
   Identical architecture to Sequential NN but with functional implementation
   19,905 total parameters with batch normalization
   Same training protocol for fair comparison

Key Findings:
Random Forest outperformed both Neural Network (NN) architectures,
Deeper networks (128 units) didn't improve performance,
Regularization was critical for NN stability,
Tabular medical data favoured tree-based methods.

Tech Stack: Python, Scikit-learn, TensorFlow 2.x, Keras, Pandas

