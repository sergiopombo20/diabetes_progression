# Guía de Defensa: Predictor de Progresión de Diabetes

Este documento es tu **guión técnico** para la defensa oral de la asignatura de Bioinformática y Medicina. Contiene la explicación paso a paso de cada decisión que se tomó en el proyecto para que puedas demostrar que entiendes el porqué de cada línea de código.

---

## 1. El Problema Biológico y el Dataset

> [!NOTE]  
> **El contexto:** La diabetes mellitus es una enfermedad crónica que afecta a la forma en que el cuerpo convierte los alimentos en energía. Nuestro objetivo es poder predecir cómo de **grave** será la evolución de la enfermedad en un paciente tras 1 año, basándonos únicamente en sus variables fisiológicas tomadas hoy.

**El Dataset:**
Hemos utilizado el *Diabetes Dataset* oficial incluido en la librería `scikit-learn` (originario del estudio de Efron et al., 2004). 
- **Tamaño:** 442 pacientes.
- **Variables de entrada (Features):** 10 parámetros clínicos:
  - Edad (`age`), Sexo (`sex`), Índice de Masa Corporal (`bmi`), Presión Arterial Media (`bp`).
  - 6 mediciones de suero sanguíneo: `tc` (colesterol total), `ldl`, `hdl` (colesterol bueno), `tch`, `ltg` (triglicéridos) y `glu` (glucosa en ayunas).
- **Target (Variable a predecir):** Originalmente es un valor numérico cuantitativo (entre 25 y 346) que indica la progresión de la enfermedad un año después de las pruebas base.

---

## 2. Decisiones de Diseño en Python (`train_model.py`)

Para poder entrenar un clasificador, tuvimos que transformar el problema matemático original.

### A. Binarización del Target
El dataset original está preparado para una *Regresión Lineal* (predecir el número exacto), pero la rúbrica exigía un clasificador. 
**¿Qué hicimos?** Calculamos la mediana del target (140.5). A los pacientes con una progresión por encima de la mediana los clasificamos como **Clase 1 (Diabetes Descontrolada)** y a los que están por debajo como **Clase 0 (Diabetes Controlada)**. Así el dataset queda perfectamente balanceado (50% / 50%).

### B. Estandarización (`StandardScaler`)
Antes de entrenar, dividimos en Entrenamiento (80%) y Test (20%). Luego aplicamos `StandardScaler`.
**¿Por qué?** Los triglicéridos tienen valores muy bajos (ej: 4.6) y el colesterol total muy altos (ej: 200). Si no estandarizamos (poner todas las variables con media=0 y varianza=1), la Regresión Logística daría erróneamente más importancia al colesterol solo porque sus números son más grandes.

### C. El Modelo: Regresión Logística con regularización L2
Elegimos la Regresión Logística porque es un algoritmo transparente (es decir, nos permite entender qué peso tiene cada variable, a diferencia de una caja negra como una Red Neuronal profunda).
La regularización **L2** penaliza los pesos muy grandes, evitando el sobreajuste (*overfitting*) en un dataset tan pequeño (apenas 442 muestras).

> [!TIP]  
> **Posible pregunta del profesor:** *¿Por qué un accuracy del ~74% y no del 95%?*
> **Tu respuesta:** Porque los datos biológicos y médicos son altamente ruidosos y multifactoriales. Predecir a un año vista solo con 10 variables basales tiene un límite biológico. Forzar un modelo para que de un 95% en training con este dataset implicaría haber hecho overfitting extremo, y fallaría estrepitosamente en el mundo real.

---

## 3. Interpretación Biomédica del Modelo

Uno de los requisitos de la rúbrica es explicar la importancia de las características. La Regresión Logística asigna un "peso" o coeficiente a cada variable biológica.

* **Coeficientes Positivos (Factores de Riesgo):** Aumentan la probabilidad de que la diabetes se descontrole. Nuestro modelo detectó que el **LTG (Log Triglicéridos)**, el **IMC (Índice de Masa Corporal)** y la **Presión Arterial (BP)** son los principales causantes de una mala progresión. Clínicamente esto tiene todo el sentido, ya que la obesidad y los triglicéridos altos generan resistencia a la insulina.
* **Coeficientes Negativos (Factores Protectores):** Reducen la probabilidad. Por ejemplo, el colesterol total y el HDL actúan reduciendo este riesgo relativo en nuestro modelo.

---

## 4. La Arquitectura Web (`index.html`)

Aquí es donde el proyecto destaca técnicamente. Hemos creado una aplicación "Serverless" (sin servidor).

> [!IMPORTANT]  
> **¿Cómo funciona la predicción sin tener Python en el servidor web?**
> En lugar de enviar un API Request a un servidor en la nube cada vez que el usuario mueve un deslizador, **hemos exportado las matemáticas del modelo entrenado a un JSON**. 

Cuando ejecutas `train_model.py`, se extraen:
1. El **bias** (sesgo o intercept).
2. Los 10 **pesos** (coeficientes).
3. La media y desviación estándar de cada variable (necesarias para escalar los datos introducidos por el usuario).

El archivo `index.html` lee este JSON. Cuando haces clic en "Predecir", el código JavaScript hace exactamente la misma matemática que `scikit-learn`:
1. **Estandariza** la entrada del usuario: `(valor - media) / desviacion_estandar`.
2. **Suma ponderada (Logit):** Multiplica cada variable estandarizada por su peso y le suma el bias. Obteniendo un valor `z`.
3. **Función Sigmoide:** Pasa el valor `z` por la función matemática $f(z) = \frac{1}{1 + e^{-z}}$ para convertir ese número en una probabilidad pura entre 0% y 100%.

> [!NOTE]  
> **Ventaja de esta arquitectura para la defensa:** Al hacerlo todo en el cliente (navegador), garantizamos privacidad médica (los datos nunca viajan por internet), coste cero de alojamiento (se puede subir a GitHub Pages estáticamente) y respuestas en tiempo real al usuario (latencia cero al mover los sliders).
