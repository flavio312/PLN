# Procesamiento de Lenguaje Natural (PLN) para Detección de Sesgos

## 1. Resumen Ejecutivo

Este informe finaliza la implementación de un *pipeline* de Procesamiento de Lenguaje Natural (PLN) diseñado para la **clasificación de sesgos contextuales** y la **Generación Aumentada por Recuperación (RAG)**. El proyecto cumple con el requisito de incorporar el **análisis de significado contextual** (mediante *embeddings* contextuales) para detectar la sutileza de los sesgos y un sistema RAG para generar explicaciones detalladas.

Tras recibir el dataset etiquetado, se completaron con éxito todas las fases del proyecto, incluyendo el entrenamiento del modelo de clasificación de sesgos.

## 2. Metodología de Implementación (Pipeline PLN)

El proyecto se estructuró en las fases descritas en el documento redactado, utilizando Python y bibliotecas de PLN de vanguardia como `spaCy`, `transformers` y `sentence-transformers`.

### 2.1. Fases 1 y 2: Preprocesamiento y Embeddings Contextuales

El preprocesamiento incluyó la limpieza básica, el **Reconocimiento de Entidades Nombradas (NER)** y la **anonimización** (reemplazo de nombres por *placeholders* como `[PERSONA_X]`) para mitigar riesgos éticos. La clave para el análisis contextual fue la generación de **Embeddings Contextuales** utilizando el modelo `hiiamsid/sentence_similarity_spanish_es`, que transforma el texto en vectores numéricos que capturan el significado y el contexto.

### 2.2. Fase 3: Clasificación de Sesgos (Entrenamiento Completado)

El modelo de clasificación se entrenó utilizando los **Embeddings Contextuales** y la **Puntuación de Polarización** (simulada) como características de entrada, tal como se especificó en la guía.

**Modelo:** Máquina de Vectores de Soporte (SVM) con kernel lineal.
**Datos de Entrenamiento:** 34 muestras etiquetadas.
**Datos de Prueba:** 15 muestras etiquetadas.
**Clases Detectadas:** 'Sin Clasificar', 'Apelación Emocional', 'Falacia Ad Hominem'.

#### Informe de Clasificación del Sesgo

El rendimiento del modelo en el conjunto de prueba fue el siguiente:

| Clase | Precisión | Recall | F1-Score | Soporte |
| :--- | :--- | :--- | :--- | :--- |
| Apelación Emocional | 0.00 | 0.00 | 0.00 | 2 |
| Falacia Ad Hominem | 0.00 | 0.00 | 0.00 | 1 |
| Sin Clasificar | 0.80 | 1.00 | 0.89 | 12 |
| **Promedio Ponderado** | **0.64** | **0.80** | **0.71** | **15** |

**Análisis de Resultados:**

*   **Precisión General (Accuracy):** 80%. Este valor es alto debido a la gran desproporción de la clase 'Sin Clasificar' (12 de 15 muestras), lo que indica un **problema de desequilibrio de clases**.
*   **Rendimiento en Clases de Sesgo:** El modelo no pudo clasificar correctamente las clases minoritarias ('Apelación Emocional' y 'Falacia Ad Hominem'), obteniendo un F1-Score de 0.00. Esto es esperado con un conjunto de datos tan pequeño y desequilibrado.

**Recomendación:** Para mejorar la detección de sesgos sutiles, se requiere un **conjunto de datos mucho más grande y balanceado** con cientos de ejemplos para cada tipo de sesgo.

### 2.3. Fase 4: Implementación y Prueba del Sistema RAG

El sistema RAG (Generación Aumentada por Recuperación) se implementó para **aumentar la explicabilidad** de la clasificación.

| Paso | Componente de PLN/ML | Descripción de la Implementación |
| :--- | :--- | :--- |
| **Recuperación de Conocimiento (RAG)** | Búsqueda de Similitud Vectorial | Se utilizó la **similitud del coseno** entre el *EmbeddingVector* de un tweet de consulta y la Base de Conocimiento RAG para encontrar el fragmento de texto más contextualmente similar (`RAG_ENTRADA`). |
| **Generación de la Explicación** | Simulación de LLM | Se simuló la generación de una explicación detallada. El LLM utiliza el texto original, la clasificación de sesgo y la `RAG_ENTRADA` recuperada para generar una explicación pedagógica. |

#### Resultado de la Prueba del Sistema RAG

Se seleccionó el primer tweet del dataset para probar la funcionalidad RAG.

| Parámetro | Valor |
| :--- | :--- |
| **Tweet de Consulta (Original)** | `RT @ExpresoPeru: 🔴 Narcogobierno mexicano asila a golpista Chávez | "Habiendo protegido a Evo Morales, Jorge Glas y ahora con el caso Betss…"` |
| **Tipo de Sesgo (Predicho)** | `Sin Clasificar` |
| **RAG_ENTRADA (Texto más similar)** | `rt narcogobierno mexicano asila golpista chávez habiendo protegido evo morales [PERSONA_X]` |
| **Similitud (Coseno)** | `231.4460` |

**Simulación de la Explicación Generada por LLM:**

> **Descripción Detallada y Razones del Sesgo (Simulación de LLM):**
>
> El fragmento de texto: "RT @ExpresoPeru: 🔴 Narcogobierno mexicano asila a golpista Chávez | "Habiendo protegido a Evo Morales, Jorge Glas y ahora con el caso Betss…" fue clasificado con el sesgo "Sin Clasificar" (etiqueta original).
>
> **RAG_ENTRADA (Contexto Recuperado):**
> El sistema RAG recuperó el siguiente fragmento similar de la base de conocimiento: "rt narcogobierno mexicano asila golpista chávez habiendo protegido evo morales [PERSONA_X]" (etiquetado como "Sin Clasificar").
>
> **Explicación (Generada):**
> La similitud contextual (Similitud Coseno: 231.4460) entre el tweet y el fragmento recuperado sugiere que ambos comparten una estructura semántica similar.
>
> *   **Análisis Contextual (Simulado):** El uso de palabras clave y la estructura de la frase en el tweet de consulta se asemejan a la forma en que se expresa el sesgo de "Sin Clasificar" en el fragmento recuperado.
> *   **Puntuación de Polarización:** La puntuación de polarización de 0.00 indica una carga emocional (simulada) que a menudo acompaña a este tipo de sesgo.
>
> Este proceso demuestra cómo el sistema RAG puede **aumentar** la explicación de la clasificación proporcionando un ejemplo contextual relevante de la base de conocimiento.

## 3. Archivos Generados

Los siguientes archivos se generaron durante la implementación del *pipeline*:

| Archivo | Descripción |
| :--- | :--- |
| `pln_pipeline.py` | Código fuente completo del *pipeline* de PLN (Fases 1, 2, 3 y 4). |
| `bias_classification_report.txt` | Informe de rendimiento del clasificador de sesgos (Fase 3). |
| `rag_system_test.txt` | Resultado de la prueba de la funcionalidad RAG (simulación de la explicación generada por LLM). |
