import pandas as pd
import numpy as np
import re
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

RAW_DATA_PATH = "tweets_politica_raw_20251104_125155.csv"
ML_DATA_PATH = "dataset.csv"
PROCESSED_DATA_PATH = "processed_data.csv"
RAG_KNOWLEDGE_BASE_PATH = "rag_knowledge_base.csv"
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
EMBEDDING_MODEL_NAME = "hiiamsid/sentence_similarity_spanish_es"

try:
    nlp = spacy.load("es_core_news_sm")
except:
    print("Error al cargar el modelo de spaCy. Asegúrate de que 'es_core_news_sm' esté instalado.")
    exit()

# --- Fase 1: Preparación, Limpieza y Mitigación de Riesgos ---

def load_and_merge_data():
    """Carga y fusiona los dos datasets."""
    print("Cargando datasets...")
    try:
        # Forzar la lectura con coma como separador, ya que la inspección lo sugiere.
        df_raw = pd.read_csv(RAW_DATA_PATH, encoding='utf-8', sep=',')
        df_ml = pd.read_csv(ML_DATA_PATH, encoding='utf-8', sep=',')
    except Exception as e:
        print(f"Error al cargar CSV: {e}. Intentando con delimitador ';'")
        try:
            df_raw = pd.read_csv(RAW_DATA_PATH, encoding='utf-8', sep=';')
            df_ml = pd.read_csv(ML_DATA_PATH, encoding='utf-8', sep=';')
        except Exception as e:
            print(f"Error al cargar CSV con delimitador ';': {e}. Fallo en la carga.")
            return None

    # Si la carga fue exitosa, continuar con la lógica de fusión
    
    # Inspección inicial de columnas
    print(f"Columnas en df_raw: {df_raw.columns.tolist()}")
    print(f"Columnas en df_ml: {df_ml.columns.tolist()}")

    # Asumiendo que df_ml es el dataset principal con las etiquetas.
    df = df_ml.copy()
    
    # Renombrar las columnas según la inspección previa
    if 'texto' in df.columns:
        df.rename(columns={'texto': 'FragmentoTexto'}, inplace=True)
    # Renombrar las columnas según la inspección previa
    if 'texto' in df.columns:
        df.rename(columns={'texto': 'FragmentoTexto'}, inplace=True)
        
    # El nuevo CSV parece tener una columna de etiqueta sin nombre al final.
    # Asumiremos que la última columna es la etiqueta de sesgo.
    if df.shape[1] > 40: # El dataset original tenía 40 columnas
        df.rename(columns={df.columns[-1]: 'TipoSesgo'}, inplace=True)
    elif 'sesgo_confirmacion' in df.columns:
        df.rename(columns={'sesgo_confirmacion': 'TipoSesgo'}, inplace=True)
    elif 'sesgo_politico' in df.columns:
        df.rename(columns={'sesgo_politico': 'TipoSesgo'}, inplace=True)
    else:
        print("ADVERTENCIA: No se encontró una columna de etiqueta de sesgo ('sesgo_politico', 'sesgo_confirmacion' o la última columna).")
        df['TipoSesgo'] = 'NO_ETIQUETADO' # Crear una columna de etiqueta por defecto

    # Asegurar que las columnas clave existan
    if 'FragmentoTexto' not in df.columns or 'TipoSesgo' not in df.columns:
        print("ERROR: Las columnas 'FragmentoTexto' o 'TipoSesgo' no se encontraron después de la carga/fusión.")
        print(f"Columnas finales: {df.columns.tolist()}")
        return None

    # Rellenar valores nulos en TipoSesgo con 'NO_ETIQUETADO' para evitar la eliminación de filas
    df['TipoSesgo'].fillna('NO_ETIQUETADO', inplace=True)

    # Eliminar duplicados y nulos en el texto
    df.dropna(subset=['FragmentoTexto'], inplace=True)
    df.drop_duplicates(subset=['FragmentoTexto'], inplace=True)
    
    print(f"Datos cargados y limpios. Total de registros: {len(df)}")
    return df

    # y el dataset raw contiene el texto original (FragmentoTexto).
    # Necesitamos encontrar una clave común para fusionar.
    # Por ahora, asumiremos que el dataset ML es el principal y contiene la columna de texto.
    # Si no hay una columna de texto en df_ml, usaremos df_raw y asumiremos que el orden es el mismo.
    
    # Inspección inicial de columnas
    print(f"Columnas en df_raw: {df_raw.columns.tolist()}")
    print(f"Columnas en df_ml: {df_ml.columns.tolist()}")

    # Asumiendo que df_ml es el dataset principal con las etiquetas.
    df = df_ml.copy()
    
    # Renombrar las columnas según la inspección previa
    if 'texto' in df.columns:
        df.rename(columns={'texto': 'FragmentoTexto'}, inplace=True)
    if 'sesgo_politico' in df.columns:
        df.rename(columns={'sesgo_politico': 'TipoSesgo'}, inplace=True)
        
    # Asegurar que las columnas clave existan
    if 'FragmentoTexto' not in df.columns or 'TipoSesgo' not in df.columns:
        print("ERROR: Las columnas 'FragmentoTexto' o 'TipoSesgo' no se encontraron después de la carga/fusión.")
        print(f"Columnas finales: {df.columns.tolist()}")
        return None

    # Eliminar duplicados y nulos en el texto
    df.dropna(subset=['FragmentoTexto', 'TipoSesgo'], inplace=True)
    df.drop_duplicates(subset=['FragmentoTexto'], inplace=True)
    
    print(f"Datos cargados y limpios. Total de registros: {len(df)}")
    return df

def basic_cleaning(text):
    """Paso 2: Limpieza Básica (URLs, HTML, Stop Words)."""
    if not isinstance(text, str):
        return ""
    
    # Eliminar URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Eliminar menciones de Twitter (@usuario)
    text = re.sub(r'@\w+', '', text)
    # Eliminar hashtags (#)
    text = re.sub(r'#', '', text)
    # Eliminar caracteres especiales y números (manteniendo letras y espacios)
    text = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]', '', text)
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar espacios extra
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Eliminación de Stop Words (usando spaCy)
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop]
    
    return " ".join(tokens)

def ner_and_anonymization(text):
    """Paso 3: Reconocimiento de Entidades Nombradas (NER) y Anonimización."""
    doc = nlp(text)
    
    # Reemplazar entidades con placeholders
    anon_text = text
    for ent in doc.ents:
        if ent.label_ in ["PER", "LOC", "ORG"]: # Personas, Lugares, Organizaciones
            placeholder = f"[{ent.label_}_{ent.text.upper()}]" # Usamos el texto original para el placeholder
            anon_text = anon_text.replace(ent.text, placeholder)
            
    # La especificación pide un placeholder genérico como [PERSONA_X], pero
    # para fines de trazabilidad y para no perder información relevante para el sesgo,
    # usaremos un placeholder más descriptivo. Si el usuario insiste en el genérico, se ajustará.
    
    # Para el requisito de [PERSONA_X], haremos una sustitución simple:
    anon_text = re.sub(r'\[PER_.*?\]', '[PERSONA_X]', anon_text)
    anon_text = re.sub(r'\[LOC_.*?\]', '[LUGAR_Y]', anon_text)
    anon_text = re.sub(r'\[ORG_.*?\]', '[ORGANIZACION_Z]', anon_text)
    
    return anon_text

def sentiment_analysis(text):
    """Paso 4: Análisis de Polarización (simulado con un modelo simple por ahora)."""
    # En un proyecto real, se usaría un modelo pre-entrenado (ej. BERT para Sentimiento en español).
    # Aquí, simularemos la polarización con un enfoque basado en palabras clave.
    
    positive_words = ["excelente", "bueno", "positivo", "apoyo", "mejor", "victoria", "éxito"]
    negative_words = ["malo", "pésimo", "corrupción", "fraude", "crisis", "peor", "derrota"]
    
    score = 0
    words = text.lower().split()
    
    for word in words:
        if word in positive_words:
            score += 0.2
        elif word in negative_words:
            score -= 0.2
            
    # Normalizar a un rango de -1.0 a 1.0
    return np.clip(score, -1.0, 1.0)

# --- Fase 2: Ingeniería de Características Semánticas ---

def generate_embeddings(df):
    """Paso 5: Generación de Embeddings Contextuales."""
    print("Cargando modelo de embeddings...")
    try:
        # Usaremos SentenceTransformer para generar embeddings de frases
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Error al cargar el modelo de embeddings: {e}")
        print("Intentando con un modelo más común...")
        try:
            model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        except Exception as e:
            print(f"Error al cargar el modelo de respaldo: {e}")
            return None

    print("Generando embeddings para FragmentoTexto_Anonimizado...")
    
    # Convertir la columna a lista para el modelo
    texts = df['FragmentoTexto_Anonimizado'].tolist()
    
    # Generar embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Almacenar los embeddings como una nueva columna (lista de numpy arrays)
    df['EmbeddingVector'] = list(embeddings)
    
    return df

def create_rag_knowledge_base(df):
    """Paso 6: Almacenamiento Vectorial (Simulado) y Base de Conocimiento RAG."""
    # Para simular la Base de Conocimiento RAG, usaremos el mismo dataset.
    # En un escenario real, la Base de Conocimiento RAG sería un conjunto de documentos
    # que contienen las definiciones y ejemplos de los sesgos.
    
    # Aquí, crearemos una base de conocimiento simple a partir de los datos etiquetados.
    # Cada entrada RAG_ENTRADA será el texto anonimizado y su TipoSesgo.
    
    rag_df = df[['FragmentoTexto_Anonimizado', 'TipoSesgo', 'EmbeddingVector']].copy()
    rag_df.rename(columns={'FragmentoTexto_Anonimizado': 'RAG_ENTRADA', 'TipoSesgo': 'Sesgo_Etiqueta'}, inplace=True)
    
    # Guardar la base de conocimiento RAG (simulando la base vectorial)
    rag_df.to_pickle(RAG_KNOWLEDGE_BASE_PATH)
    print(f"Base de Conocimiento RAG (simulada) guardada en: {RAG_KNOWLEDGE_BASE_PATH}")
    
    return rag_df

# --- Función Principal del Pipeline ---

def run_pln_pipeline():
    # 1. Cargar y fusionar datos
    df = load_and_merge_data()
    if df is None:
        return

    # 2. Aplicar Limpieza Básica
    print("Aplicando limpieza básica...")
    df['FragmentoTexto_Limpio'] = df['FragmentoTexto'].apply(basic_cleaning)

    # 3. Aplicar NER y Anonimización
    print("Aplicando NER y anonimización...")
    df['FragmentoTexto_Anonimizado'] = df['FragmentoTexto_Limpio'].apply(ner_and_anonymization)

    # 4. Análisis de Polarización
    print("Calculando PuntuacionPolarizacion...")
    df['PuntuacionPolarizacion'] = df['FragmentoTexto_Anonimizado'].apply(sentiment_analysis)

    # 5. Generación de Embeddings
    df = generate_embeddings(df)
    if df is None:
        return

    # 6. Creación de la Base de Conocimiento RAG
    create_rag_knowledge_base(df)

    # Guardar el dataset preprocesado para la Fase 3
    df.to_pickle(PROCESSED_DATA_PATH)
    print(f"Dataset preprocesado guardado en: {PROCESSED_DATA_PATH}")
    
    print("Fases 1 y 2 completadas. Listo para la Fase 3 (Clasificación y RAG).")

# --- Fase 3: Detección, Clasificación e Integración (ML) ---

def train_bias_classifier():
    """Paso 7: Clasificación del Sesgo."""
    print("\n--- Fase 3: Clasificación del Sesgo ---")
    try:
        df = pd.read_pickle(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de datos procesados en {PROCESSED_DATA_PATH}")
        return

    # 1. Preparación de Características (X) y Etiquetas (y)
    # X debe ser la concatenación del EmbeddingVector y la PuntuacionPolarizacion
    
    # Convertir la lista de embeddings y la puntuación de polarización en una matriz de características
    X_embeddings = np.array(df['EmbeddingVector'].tolist())
    X_polarization = df['PuntuacionPolarizacion'].values.reshape(-1, 1)
    
    # Concatenar las características
    X = np.hstack((X_embeddings, X_polarization))
    
    # Etiquetas (y)
    y = df['TipoSesgo']
    
    # 2. Manejo de Clases (si hay 'NO_ETIQUETADO', lo ignoramos para el entrenamiento)
    df_train = df[df['TipoSesgo'] != 'NO_ETIQUETADO'].copy()
    
    if len(df_train) == 0:
        print("ADVERTENCIA: No hay datos etiquetados para el entrenamiento (todos son 'NO_ETIQUETADO').")
        print("El modelo de clasificación no puede ser entrenado. Se saltará la Fase 3.")
        return

    X_train_emb = np.array(df_train['EmbeddingVector'].tolist())
    X_train_pol = df_train['PuntuacionPolarizacion'].values.reshape(-1, 1)
    X_train = np.hstack((X_train_emb, X_train_pol))
    y_train = df_train['TipoSesgo']
    
    # 3. Dividir en conjuntos de entrenamiento y prueba (si hay suficientes datos)
    if len(df_train) > 10:
        X_train_split, X_test, y_train_split, y_test = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
        )
    else:
        # Usar todos los datos etiquetados como entrenamiento si son muy pocos
        X_train_split, y_train_split = X_train, y_train
        X_test, y_test = X_train, y_train # Simulación de prueba en el mismo conjunto

    print(f"Datos de entrenamiento: {len(X_train_split)}")
    print(f"Datos de prueba: {len(X_test)}")
    print(f"Clases a clasificar: {y_train_split.unique().tolist()}")

    # 4. Entrenamiento del Modelo (SVM, como se sugiere en el documento)
    print("Entrenando clasificador SVM...")
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train_split, y_train_split)
    
    # 5. Evaluación del Modelo
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    print("\n--- Informe de Clasificación del Sesgo ---")
    print(report)
    
    # 6. Almacenar el modelo y el informe
    # En un entorno real, se serializaría el modelo. Aquí, guardaremos el informe.
    # CORRECCIÓN: añadido encoding='utf-8'
    with open("bias_classification_report.txt", "w", encoding='utf-8') as f:
        f.write(report)
        
    print("Informe de clasificación guardado en bias_classification_report.txt")
    
    # 7. Aplicar el modelo a todo el dataset (incluyendo los no etiquetados)
    df['TipoSesgo_Predicho'] = model.predict(X)
    df['Score_Confianza'] = [max(p) for p in model.predict_proba(X)]
    
    # Guardar el dataset actualizado
    df.to_pickle(PROCESSED_DATA_PATH)
    print(f"Dataset actualizado con predicciones guardado en: {PROCESSED_DATA_PATH}")
    
    return model

# --- Fase 4: Implementación y Prueba del Sistema RAG ---

def run_rag_system():
    """Paso 8 y 9: Recuperación de Conocimiento (RAG) y Generación de la Explicación."""
    print("\n--- Fase 4: Implementación y Prueba del Sistema RAG ---")
    try:
        df = pd.read_pickle(PROCESSED_DATA_PATH)
        rag_kb = pd.read_pickle(RAG_KNOWLEDGE_BASE_PATH)
    except FileNotFoundError:
        print("Error: No se encontraron los archivos de datos procesados o la base de conocimiento RAG.")
        return

    # Usaremos el modelo de embeddings para calcular la similitud
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception:
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    # Seleccionar un tweet de ejemplo para la prueba
    example_tweet = df.iloc[0]
    query_text = example_tweet['FragmentoTexto_Anonimizado']
    query_embedding = example_tweet['EmbeddingVector']
    
    print(f"Tweet de Consulta (FragmentoTexto): {example_tweet['FragmentoTexto']}")
    print(f"TipoSesgo Predicho (Simulado): {example_tweet['TipoSesgo']}") # Usamos el original ya que el predicho no se entrenó

    # 1. Recuperación de Conocimiento (RAG) - Búsqueda de Similitud Vectorial
    print("\n1. Recuperando conocimiento (Búsqueda de Similitud Vectorial)...")
    
    # Calcular la similitud del coseno entre el embedding de la consulta y todos los embeddings de la KB
    kb_embeddings = np.array(rag_kb['EmbeddingVector'].tolist())
    
    # Similitud del coseno (dot product para embeddings normalizados)
    similarities = np.dot(kb_embeddings, query_embedding)
    
    # Obtener el índice del elemento más similar
    most_similar_index = np.argmax(similarities)
    rag_entrada = rag_kb.iloc[most_similar_index]
    
    print(f"RAG_ENTRADA (Texto más similar): {rag_entrada['RAG_ENTRADA']}")
    print(f"Sesgo de la RAG_ENTRADA: {rag_entrada['Sesgo_Etiqueta']}")
    print(f"Similitud (Coseno): {similarities[most_similar_index]:.4f}")

    # 2. Generación de la Explicación (Simulada con LLM)
    print("\n2. Generando la Explicación (Simulación de LLM)...")
    
    # En un entorno real, aquí se usaría un LLM con un prompt como:
    # "Analiza el texto: '{query_text}'. Clasifícalo como '{TipoSesgo}'.
    # Usa la siguiente información de contexto: '{rag_entrada['RAG_ENTRADA']}' para explicar
    # por qué el texto es sesgado."
    
    # Simulación de la explicación generada:
    descripcion_generada = f"""
    **Descripción Detallada y Razones del Sesgo (Simulación de LLM):**
    
    El fragmento de texto: "{example_tweet['FragmentoTexto']}" fue clasificado con el sesgo
    "{example_tweet['TipoSesgo']}" (etiqueta original).
    
    **RAG_ENTRADA (Contexto Recuperado):**
    El sistema RAG recuperó el siguiente fragmento similar de la base de conocimiento:
    "{rag_entrada['RAG_ENTRADA']}" (etiquetado como "{rag_entrada['Sesgo_Etiqueta']}").
    
    **Explicación (Generada):**
    La similitud contextual (Similitud Coseno: {similarities[most_similar_index]:.4f}) entre el tweet
    y el fragmento recuperado sugiere que ambos comparten una estructura semántica similar.
    
    *   **Análisis Contextual (Simulado):** El uso de palabras clave y la estructura de la frase
        en el tweet de consulta se asemejan a la forma en que se expresa el sesgo de
        "{rag_entrada['Sesgo_Etiqueta']}" en el fragmento recuperado.
    *   **Puntuación de Polarización:** La puntuación de polarización de {example_tweet['PuntuacionPolarizacion']:.2f}
        indica una carga emocional (simulada) que a menudo acompaña a este tipo de sesgo.
    
    Este proceso demuestra cómo el sistema RAG puede **aumentar** la explicación de la clasificación
    proporcionando un ejemplo contextual relevante de la base de conocimiento.
    """
    
    print(descripcion_generada)
    
    # CORRECCIÓN: añadido encoding='utf-8'
    with open("rag_system_test.txt", "w", encoding='utf-8') as f:
        f.write(descripcion_generada)
        
    print("Resultado de la prueba RAG guardado en rag_system_test.txt")

if __name__ == "__main__":
    run_pln_pipeline() 
    train_bias_classifier()
    run_rag_system()