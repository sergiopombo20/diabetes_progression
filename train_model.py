#!/usr/bin/env python3
"""
Diabetes Progression Predictor - Model Training Pipeline
=========================================================
Trains a Logistic Regression classifier using the official 
scikit-learn Diabetes Dataset, evaluates performance,
and exports model weights + metadata as JSON for the web app.

Author: Sergio
License: MIT
"""

import numpy as np
import json
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Fijar semilla para REPRODUCIBILIDAD
np.random.seed(42)

print("=" * 60)
print("  DIABETES PROGRESSION PREDICTOR - Training Pipeline (Scikit-Learn)")
print("=" * 60)

# =============================================================================
# 1. LOAD OFFICIAL DIABETES DATASET
# =============================================================================
# Usamos scaled=False para obtener los valores médicos reales (edad 19-79, etc)
data = load_diabetes(scaled=False)
X_raw = data.data
y_continuous = data.target
feature_names = data.feature_names

# En el dataset original, el sexo (índice 1) viene como 1.0 y 2.0.
# Lo mapeamos a 0 y 1 para que la aplicación web funcione correctamente.
X_raw[:, 1] = X_raw[:, 1] - 1

print(f"\n[1] Dataset oficial de scikit-learn cargado: {X_raw.shape[0]} pacientes, {X_raw.shape[1]} variables")
print(f"    Target continuo (progresión a 1 año): min={y_continuous.min():.1f}, max={y_continuous.max():.1f}")
print(f"    Mediana del target: {np.median(y_continuous):.1f}")

# =============================================================================
# 2. BINARIZE TARGET
# =============================================================================
median_val = np.median(y_continuous)
# 0 = "Diabetes Controlada" (progression <= median)
# 1 = "Diabetes Descontrolada" (progression > median)
y_binary = (y_continuous > median_val).astype(int)

n_controlled = np.sum(y_binary == 0)
n_uncontrolled = np.sum(y_binary == 1)
print(f"\n[2] Target binarizado (mediana = {median_val:.1f}):")
print(f"    Diabetes Controlada (0):     {n_controlled} ({100*n_controlled/len(y_binary):.1f}%)")
print(f"    Diabetes Descontrolada (1):  {n_uncontrolled} ({100*n_uncontrolled/len(y_binary):.1f}%)")

# =============================================================================
# 3. TRAIN/TEST SPLIT & STANDARDIZATION
# =============================================================================
# Usamos random_state=42 para reproducibilidad
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# Estandarizamos las variables (media=0, std=1) para la Regresión Logística
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Extraemos media y desviación típica para la inferencia en JavaScript
X_mean = scaler.mean_
X_std = scaler.scale_

print(f"\n[3] Split (80/20): {len(X_train)} train / {len(X_test)} test (Estratificado)")

# =============================================================================
# 4. TRAIN LOGISTIC REGRESSION (Scikit-Learn)
# =============================================================================
print(f"\n[4] Entrenando modelo LogisticRegression (scikit-learn)...")
# Usamos regularización L2 por defecto. random_state=42 para reproducibilidad total.
model = LogisticRegression(penalty='l2', C=1.0, random_state=42, solver='lbfgs')
model.fit(X_train, y_train)

# Extraemos los pesos y el sesgo del modelo entrenado
weights = model.coef_[0]
bias = model.intercept_[0]

# =============================================================================
# 5. EVALUATE MODEL
# =============================================================================
def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'cm': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}
    }

train_metrics = compute_metrics(y_train, model.predict(X_train))
test_metrics = compute_metrics(y_test, model.predict(X_test))

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

cm = test_metrics['cm']
print(f"\n    Matriz de Confusión (Test):")
print(f"                    Predicted")
print(f"                  Ctrl  | Desctrl")
print(f"    Actual Ctrl  | {cm['tn']:3d}  |  {cm['fp']:3d}")
print(f"    Actual Desc  | {cm['fn']:3d}  |  {cm['tp']:3d}")

# =============================================================================
# 6. FEATURE IMPORTANCE
# =============================================================================
abs_importance = np.abs(weights)
importance_normalized = abs_importance / abs_importance.sum() * 100

print(f"\n[6] Importancia de Características (Coeficientes absolutos):")
sorted_idx = np.argsort(importance_normalized)[::-1]
# feature_names original es: ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
for i in sorted_idx:
    bar = "#" * int(importance_normalized[i] / 2)
    direction = "(+) riesgo" if weights[i] > 0 else "(-) protector"
    print(f"    {feature_names[i]:5s}: {importance_normalized[i]:5.1f}% {bar} ({direction})")

# =============================================================================
# 7. EXPORT MODEL AS JSON FOR WEB APP
# =============================================================================
# La app web index.html original espera que las características s1-s6 
# tengan los nombres tc, ldl, hdl, tch, ltg, glu.
web_names = ['age', 'sex', 'bmi', 'bp', 'tc', 'ldl', 'hdl', 'tch', 'ltg', 'glu']


feature_stats = []
feature_labels = {
    'age': ('Edad', 'años', 'Edad del paciente'),
    'sex': ('Sexo', '0=M, 1=H', 'Sexo biológico'),
    'bmi': ('IMC', 'kg/m²', 'Índice de Masa Corporal'),
    'bp':  ('Presión Arterial', 'mmHg', 'Presión arterial media'),
    's1':  ('Colesterol Total', 'mg/dL', 'Colesterol total en sangre (S1)'),
    's2':  ('Colesterol LDL', 'mg/dL', 'Colesterol LDL "malo" (S2)'),
    's3':  ('Colesterol HDL', 'mg/dL', 'Colesterol HDL "bueno" (S3)'),
    's4':  ('Ratio TC/HDL', 'ratio', 'Ratio colesterol total / HDL (S4)'),
    's5':  ('Log Triglicéridos', 'log(mg/dL)', 'Logaritmo de triglicéridos séricos (S5)'),
    's6':  ('Glucosa', 'mg/dL', 'Nivel de glucosa en ayunas (S6)'),
}

for i, original_name in enumerate(feature_names):
    web_name = web_names[i]
    label, unit, desc = feature_labels[original_name]
    col = X_raw[:, i]

    if original_name == 'sex':
        stats = {'min': 0, 'max': 1, 'mean': float(np.mean(col)), 'std': float(np.std(col)), 'step': 1}
    else:
        stats = {
            'min': float(np.floor(col.min())),
            'max': float(np.ceil(col.max())),
            'mean': float(np.mean(col)),
            'std': float(np.std(col)),
            'step': 0.1 if original_name in ['s4', 's5'] else 1
        }

    feature_stats.append({
        'name': web_name,
        'label': label,
        'unit': unit,
        'description': desc,
        'coefficient': float(weights[i]),
        'importance_pct': float(importance_normalized[i]),
        **stats
    })

model_export = {
    'model_type': 'Logistic Regression (Scikit-Learn)',
    'bias': float(bias),
    'weights': weights.tolist(),
    'feature_mean': X_mean.tolist(),
    'feature_std': X_std.tolist(),
    'feature_names': web_names,
    'features': feature_stats,
    'train_metrics': {k: float(v) if not isinstance(v, dict) else v for k, v in train_metrics.items()},
    'test_metrics': {k: float(v) if not isinstance(v, dict) else v for k, v in test_metrics.items()},
    'median_threshold': float(median_val),
    'n_train': int(len(X_train)),
    'n_test': int(len(X_test)),
    'n_total': int(len(X_raw)),
}

json_path = 'model_weights.json'
with open(json_path, 'w') as f:
    json.dump(model_export, f, indent=2)

print(f"\n[7] Modelo y metadatos exportados a '{json_path}'")
print(f"    (El archivo index.html ahora debe incrustar este nuevo JSON)")
print(f"\n{'=' * 60}")
print("  Pipeline completado. ¡Resultados 100% reales y reproducibles!")
print("=" * 60)
