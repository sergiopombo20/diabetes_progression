#!/usr/bin/env python3
"""
Diabetes Progression Predictor - Model Training Pipeline
=========================================================
Generates a realistic diabetes dataset, trains a Logistic Regression
classifier from scratch using only numpy, evaluates performance,
and exports model weights + metadata as JSON for the web app.

Author: Sergio (Vibe Coded with Claude)
License: MIT
"""

import numpy as np
import pandas as pd
import json
import os

np.random.seed(42)

# =============================================================================
# 1. GENERATE REALISTIC DIABETES DATASET
# =============================================================================
# Based on the scikit-learn diabetes dataset structure:
# 442 patients, 10 baseline variables, target = disease progression at 1 year

N_SAMPLES = 442

def generate_diabetes_dataset(n=N_SAMPLES):
    """
    Generate a synthetic diabetes dataset with realistic correlations.
    Features mirror the sklearn diabetes dataset with human-readable values.
    """
    # Generate correlated features using Cholesky decomposition
    # Feature order: age, sex, bmi, bp, s1(tc), s2(ldl), s3(hdl), s4(tch), s5(ltg), s6(glu)

    # Correlation matrix (approximate, based on known medical correlations)
    corr = np.array([
        [1.00, 0.17, 0.19, 0.34, 0.26, 0.22,-0.08, 0.20, 0.27, 0.30],  # age
        [0.17, 1.00, 0.09, 0.24, 0.04, 0.14,-0.38, 0.33, 0.15, 0.21],  # sex
        [0.19, 0.09, 1.00, 0.40, 0.25, 0.26,-0.37, 0.41, 0.45, 0.39],  # bmi
        [0.34, 0.24, 0.40, 1.00, 0.24, 0.19,-0.18, 0.26, 0.39, 0.39],  # bp
        [0.26, 0.04, 0.25, 0.24, 1.00, 0.90, 0.05, 0.54, 0.52, 0.33],  # s1 (tc)
        [0.22, 0.14, 0.26, 0.19, 0.90, 1.00,-0.20, 0.66, 0.32, 0.29],  # s2 (ldl)
        [-0.08,-0.38,-0.37,-0.18, 0.05,-0.20, 1.00,-0.74, 0.16,-0.07],  # s3 (hdl)
        [0.20, 0.33, 0.41, 0.26, 0.54, 0.66,-0.74, 1.00, 0.62, 0.42],  # s4 (tch)
        [0.27, 0.15, 0.45, 0.39, 0.52, 0.32, 0.16, 0.62, 1.00, 0.46],  # s5 (ltg)
        [0.30, 0.21, 0.39, 0.39, 0.33, 0.29,-0.07, 0.42, 0.46, 1.00],  # s6 (glu)
    ])

    # Make it positive definite
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 0.05)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalize to correlation matrix
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)

    # Add small regularization for numerical stability
    corr += np.eye(10) * 0.01

    L = np.linalg.cholesky(corr)
    Z = np.random.randn(n, 10)
    X_corr = Z @ L.T

    # Transform to realistic medical ranges
    feature_params = {
        'age':  (50, 12),      # years
        'sex':  (0.5, 0.5),    # will be binarized
        'bmi':  (26.4, 4.4),   # kg/m²
        'bp':   (95, 13),      # mm Hg (mean arterial pressure)
        'tc':   (200, 35),     # mg/dL total cholesterol (s1)
        'ldl':  (120, 30),     # mg/dL LDL cholesterol (s2)
        'hdl':  (50, 13),      # mg/dL HDL cholesterol (s3)
        'tch':  (4.1, 1.3),    # total cholesterol / HDL ratio (s4)
        'ltg':  (4.6, 0.5),    # log triglycerides (s5)
        'glu':  (92, 12),      # mg/dL fasting blood sugar (s6)
    }

    feature_names = list(feature_params.keys())
    X = np.zeros_like(X_corr)

    for i, (name, (mean, std)) in enumerate(feature_params.items()):
        X[:, i] = X_corr[:, i] * std + mean

    # Binarize sex
    X[:, 1] = (X[:, 1] > 0.5).astype(float)

    # Clip to realistic ranges
    X[:, 0] = np.clip(X[:, 0], 19, 79)   # age
    X[:, 2] = np.clip(X[:, 2], 18, 42)    # bmi
    X[:, 3] = np.clip(X[:, 3], 62, 133)   # bp
    X[:, 4] = np.clip(X[:, 4], 100, 320)  # tc
    X[:, 5] = np.clip(X[:, 5], 50, 240)   # ldl
    X[:, 6] = np.clip(X[:, 6], 22, 100)   # hdl
    X[:, 7] = np.clip(X[:, 7], 2.0, 9.0)  # tch
    X[:, 8] = np.clip(X[:, 8], 3.2, 6.1)  # ltg
    X[:, 9] = np.clip(X[:, 9], 58, 124)   # glu

    # Generate target: disease progression (quantitative, 25-346)
    # Weighted combination of features (based on known risk factors)
    weights = np.array([0.5, 3.0, 8.0, 4.0, 1.5, 2.0, -4.0, 5.0, 10.0, 3.0])

    # Standardize for target generation
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    target_continuous = X_std @ weights
    target_continuous += np.random.randn(n) * 8  # noise

    # Scale to realistic range (25-346)
    target_continuous = (target_continuous - target_continuous.min()) / (target_continuous.max() - target_continuous.min())
    target_continuous = target_continuous * 321 + 25

    return X, target_continuous, feature_names

print("=" * 60)
print("  DIABETES PROGRESSION PREDICTOR - Training Pipeline")
print("=" * 60)

X_raw, y_continuous, feature_names = generate_diabetes_dataset()

print(f"\n[1] Dataset generado: {X_raw.shape[0]} pacientes, {X_raw.shape[1]} variables")
print(f"    Target continuo: min={y_continuous.min():.1f}, max={y_continuous.max():.1f}")
print(f"    Mediana del target: {np.median(y_continuous):.1f}")

# =============================================================================
# 2. BINARIZE TARGET
# =============================================================================
median_val = np.median(y_continuous)
y_binary = (y_continuous > median_val).astype(int)
# 0 = "Diabetes Controlada" (progression <= median)
# 1 = "Diabetes Descontrolada" (progression > median)

n_controlled = np.sum(y_binary == 0)
n_uncontrolled = np.sum(y_binary == 1)
print(f"\n[2] Target binarizado (mediana = {median_val:.1f}):")
print(f"    Diabetes Controlada (0):     {n_controlled} ({100*n_controlled/len(y_binary):.1f}%)")
print(f"    Diabetes Descontrolada (1):  {n_uncontrolled} ({100*n_uncontrolled/len(y_binary):.1f}%)")

# =============================================================================
# 3. STANDARDIZE FEATURES
# =============================================================================
X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0) + 1e-8
X_scaled = (X_raw - X_mean) / X_std

# =============================================================================
# 4. TRAIN/TEST SPLIT (80/20)
# =============================================================================
indices = np.random.permutation(len(X_scaled))
split = int(0.8 * len(X_scaled))
train_idx, test_idx = indices[:split], indices[split:]

X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y_binary[train_idx], y_binary[test_idx]

print(f"\n[3] Split: {len(X_train)} train / {len(X_test)} test")

# =============================================================================
# 5. LOGISTIC REGRESSION FROM SCRATCH
# =============================================================================

def sigmoid(z):
    """Numerically stable sigmoid."""
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

def logistic_regression_train(X, y, lr=0.1, epochs=2000, reg_lambda=0.01):
    """Train logistic regression with L2 regularization via gradient descent."""
    n_samples, n_features = X.shape

    # Add bias column
    X_b = np.column_stack([np.ones(n_samples), X])

    # Initialize weights
    weights = np.zeros(n_features + 1)

    losses = []

    for epoch in range(epochs):
        # Forward pass
        z = X_b @ weights
        predictions = sigmoid(z)

        # Binary cross-entropy loss + L2 regularization
        eps = 1e-15
        loss = -np.mean(y * np.log(predictions + eps) + (1 - y) * np.log(1 - predictions + eps))
        loss += (reg_lambda / (2 * n_samples)) * np.sum(weights[1:] ** 2)
        losses.append(loss)

        # Gradient
        error = predictions - y
        gradient = (1 / n_samples) * (X_b.T @ error)
        gradient[1:] += (reg_lambda / n_samples) * weights[1:]

        # Update
        weights -= lr * gradient

        if epoch % 500 == 0:
            print(f"    Epoch {epoch:4d}: loss = {loss:.4f}")

    return weights, losses

print(f"\n[4] Entrenando Logistic Regression (L2 regularized)...")
weights, losses = logistic_regression_train(X_train, y_train, lr=0.15, epochs=3000, reg_lambda=0.05)

# =============================================================================
# 6. EVALUATE MODEL
# =============================================================================

def predict(X, weights):
    """Predict probabilities and classes."""
    X_b = np.column_stack([np.ones(X.shape[0]), X])
    probs = sigmoid(X_b @ weights)
    classes = (probs >= 0.5).astype(int)
    return probs, classes

def compute_metrics(y_true, y_pred):
    """Compute accuracy, precision, recall, F1, confusion matrix."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}
    }

# Training metrics
_, y_train_pred = predict(X_train, weights)
train_metrics = compute_metrics(y_train, y_train_pred)

# Test metrics
probs_test, y_test_pred = predict(X_test, weights)
test_metrics = compute_metrics(y_test, y_test_pred)

print(f"\n[5] Resultados del Modelo:")
print(f"\n    --- TRAIN ---")
print(f"    Accuracy:  {train_metrics['accuracy']:.4f}")
print(f"    Precision: {train_metrics['precision']:.4f}")
print(f"    Recall:    {train_metrics['recall']:.4f}")
print(f"    F1-Score:  {train_metrics['f1']:.4f}")

print(f"\n    --- TEST ---")
print(f"    Accuracy:  {test_metrics['accuracy']:.4f}")
print(f"    Precision: {test_metrics['precision']:.4f}")
print(f"    Recall:    {test_metrics['recall']:.4f}")
print(f"    F1-Score:  {test_metrics['f1']:.4f}")

cm = test_metrics['confusion_matrix']
print(f"\n    Confusion Matrix (Test):")
print(f"                    Predicted")
print(f"                  Ctrl  | Desctrl")
print(f"    Actual Ctrl  | {cm['tn']:3d}  |  {cm['fp']:3d}")
print(f"    Actual Desc  | {cm['fn']:3d}  |  {cm['tp']:3d}")

# =============================================================================
# 7. FEATURE IMPORTANCE (from coefficients)
# =============================================================================
coefficients = weights[1:]  # exclude bias
abs_importance = np.abs(coefficients)
importance_normalized = abs_importance / abs_importance.sum() * 100

print(f"\n[6] Importancia de Características:")
sorted_idx = np.argsort(importance_normalized)[::-1]
for i in sorted_idx:
    bar = "█" * int(importance_normalized[i] / 2)
    direction = "↑ riesgo" if coefficients[i] > 0 else "↓ protector"
    print(f"    {feature_names[i]:5s}: {importance_normalized[i]:5.1f}% {bar} ({direction})")

# =============================================================================
# 8. COMPUTE FEATURE STATISTICS FOR THE WEB APP
# =============================================================================
feature_stats = []
feature_labels = {
    'age': ('Edad', 'años', 'Edad del paciente'),
    'sex': ('Sexo', '0=F, 1=M', 'Sexo biológico'),
    'bmi': ('IMC', 'kg/m²', 'Índice de Masa Corporal'),
    'bp':  ('Presión Arterial', 'mmHg', 'Presión arterial media'),
    'tc':  ('Colesterol Total', 'mg/dL', 'Colesterol total en sangre (S1)'),
    'ldl': ('Colesterol LDL', 'mg/dL', 'Colesterol LDL "malo" (S2)'),
    'hdl': ('Colesterol HDL', 'mg/dL', 'Colesterol HDL "bueno" (S3)'),
    'tch': ('Ratio TC/HDL', 'ratio', 'Ratio colesterol total / HDL (S4)'),
    'ltg': ('Log Triglicéridos', 'log(mg/dL)', 'Logaritmo de triglicéridos séricos (S5)'),
    'glu': ('Glucosa', 'mg/dL', 'Nivel de glucosa en ayunas (S6)'),
}

for i, name in enumerate(feature_names):
    label, unit, desc = feature_labels[name]
    col = X_raw[:, i]

    if name == 'sex':
        stats = {'min': 0, 'max': 1, 'mean': float(np.mean(col)), 'std': float(np.std(col)), 'step': 1}
    else:
        stats = {
            'min': float(np.floor(col.min())),
            'max': float(np.ceil(col.max())),
            'mean': float(np.mean(col)),
            'std': float(np.std(col)),
            'step': 0.1 if name in ['tch', 'ltg'] else 1
        }

    feature_stats.append({
        'name': name,
        'label': label,
        'unit': unit,
        'description': desc,
        'coefficient': float(coefficients[i]),
        'importance_pct': float(importance_normalized[i]),
        **stats
    })

# =============================================================================
# 9. EXPORT MODEL AS JSON
# =============================================================================
model_export = {
    'model_type': 'Logistic Regression (L2 Regularized)',
    'bias': float(weights[0]),
    'weights': weights[1:].tolist(),
    'feature_mean': X_mean.tolist(),
    'feature_std': X_std.tolist(),
    'feature_names': feature_names,
    'features': feature_stats,
    'train_metrics': {k: v if not isinstance(v, np.floating) else float(v) for k, v in train_metrics.items()},
    'test_metrics': {k: v if not isinstance(v, np.floating) else float(v) for k, v in test_metrics.items()},
    'median_threshold': float(median_val),
    'n_train': int(len(X_train)),
    'n_test': int(len(X_test)),
    'n_total': int(len(X_scaled)),
    'training_loss_final': float(losses[-1]),
}

json_path = 'model_weights.json'
with open(json_path, 'w') as f:
    json.dump(model_export, f, indent=2)

print(f"\n[7] Modelo exportado a '{json_path}'")
print(f"    Bias: {weights[0]:.4f}")
print(f"    Weights: {[f'{w:.4f}' for w in weights[1:]]}")
print(f"\n{'=' * 60}")
print("  Pipeline completado exitosamente!")
print("=" * 60)
