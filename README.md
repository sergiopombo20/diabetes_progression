# DiabetesAI — Predictor de Progresión de Diabetes

Proyecto final de la asignatura **Bioinformática y Medicina** del **Grado en Inteligencia Artificial** de la **Universidade da Coruña**.

**Demo en vivo:** [diabetes-progression.vercel.app](https://diabetes-progression.vercel.app/)

---

## Descripción

DiabetesAI es una aplicación web que predice si la progresión de la diabetes de un paciente a un año vista será **controlada** o **descontrolada**, a partir de 10 variables clínicas de base. El modelo de Machine Learning se entrena en Python con scikit-learn y sus pesos se exportan como JSON para que toda la inferencia ocurra directamente en el navegador, sin necesidad de servidor.

> **Aviso:** Esta herramienta es un proyecto educativo y **no** sustituye el diagnóstico médico profesional.

---

## Dataset

Se utiliza el [Diabetes Dataset oficial de scikit-learn](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset), basado en el estudio de Efron, Hastie, Johnstone y Tibshirani (2004).

| Característica | Descripción |
|---|---|
| **Muestras** | 442 pacientes |
| **Variables** | 10 variables clínicas |
| **Target original** | Progresión continua a 1 año (rango 25–346) |
| **Target binarizado** | Controlada (≤ mediana 140.5) / Descontrolada (> 140.5) |
| **Balance de clases** | 50% / 50% — perfectamente balanceado |
| **Valores nulos** | Ninguno |

### Variables clínicas

| Nombre | Variable | Unidad |
|---|---|---|
| `age` | Edad | años |
| `sex` | Sexo biológico | 0 = Mujer, 1 = Hombre |
| `bmi` | Índice de Masa Corporal | kg/m² |
| `bp` | Presión arterial media | mmHg |
| `tc` | Colesterol total | mg/dL |
| `ldl` | Colesterol LDL | mg/dL |
| `hdl` | Colesterol HDL | mg/dL |
| `tch` | Ratio Colesterol Total / HDL | ratio |
| `ltg` | Log de triglicéridos séricos | log(mg/dL) |
| `glu` | Glucosa en ayunas | mg/dL |

---

## Modelo

- **Algoritmo:** Regresión Logística con regularización L2
- **Implementación:** scikit-learn (`LogisticRegression`, solver `lbfgs`, `C=1.0`)
- **Preprocesamiento:** `StandardScaler` (ajustado sobre train, aplicado sobre test)
- **Split:** 80% entrenamiento (353) / 20% test (89), estratificado
- **Reproducibilidad:** `random_state=42` en todo el pipeline

### Métricas (conjunto de test)

| Métrica | Valor |
|---|---|
| Accuracy | **74.2%** |
| Precision | 71.4% |
| Recall | 79.5% |
| F1-Score | 75.3% |
| AUC-ROC | 0.83 |

### Variables más influyentes

| Variable | Importancia | Dirección |
|---|---|---|
| Log Triglicéridos (`ltg`) | 20.8% | Factor de riesgo |
| IMC (`bmi`) | 17.8% | Factor de riesgo |
| Presión Arterial (`bp`) | 17.3% | Factor de riesgo |
| Colesterol Total (`tc`) | 13.2% | Factor protector |
| Sexo (`sex`) | 11.7% | Factor protector |

---

## Estructura del proyecto

```
Diabetes Progression/
├── index.html                    # Aplicación web completa (autocontenida)
├── train_model.py                # Pipeline de entrenamiento (Python)
├── model_weights.json            # Pesos exportados del modelo
├── tools.ipynb                   # Análisis exploratorio (EDA)
├── estadisticas_diabetes.png     # Estadísticas descriptivas del dataset
├── reporte_rendimiento.png       # Métricas y matriz de confusión
├── curva_roc_auc.png             # Curva ROC real del modelo
├── split_estratificado.png       # Distribución del split train/test
├── comparativa_estandarizacion.png  # Efecto del StandardScaler
├── tabla_diabetes.png            # Primeras filas del dataset
└── split_quesitos.png            # Distribución de clases (pie chart)
```

---

## Uso

### Aplicación web

Disponible en línea en [diabetes-progression.vercel.app](https://diabetes-progression.vercel.app/) o abriendo `index.html` directamente en el navegador. No requiere servidor ni instalación.

La app permite:
- Ajustar las 10 variables clínicas mediante sliders
- Obtener la predicción y probabilidad en tiempo real
- Ver los factores clave que influyen en cada predicción
- Recibir recomendaciones personalizadas basadas en los parámetros del modelo
- Exportar el resultado en **CSV** o **PDF**

### Reentrenar el modelo

```bash
pip install scikit-learn numpy
python train_model.py
```

El script genera un nuevo `model_weights.json`. Para que la app web use los nuevos pesos, el JSON ya está incrustado en `index.html` — si se modifica el pipeline, actualizar la constante `MODEL` en el script con los nuevos valores exportados.

### Análisis exploratorio

Abrir `tools.ipynb` con Jupyter y ejecutar las celdas en orden. Requiere:

```bash
pip install scikit-learn numpy pandas matplotlib seaborn jupyter
```

---

## Tecnologías

| Componente | Tecnología |
|---|---|
| Entrenamiento | Python 3, scikit-learn, NumPy |
| Análisis | Pandas, Matplotlib, Seaborn, Jupyter |
| Frontend | HTML5, CSS3, JavaScript (sin dependencias externas) |
| Inferencia en web | Sigmoid sobre combinación lineal estandarizada (JS puro) |

---

## Autor

**Sergio Pombo** — Grado en Inteligencia Artificial, Universidade da Coruña

---

## Licencia

MIT License — proyecto de uso educativo.

---

## Referencia del dataset

Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). *Least Angle Regression*. The Annals of Statistics, 32(2), 407–499.
