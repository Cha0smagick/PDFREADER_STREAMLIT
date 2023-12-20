import streamlit as st
import os
import nltk
import fitz  # PyMuPDF
from io import BytesIO
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import subprocess

# Función para descargar el modelo de spaCy si aún no está instalado
def download_spacy_model(model_name="en_core_web_sm"):
    try:
        # Intenta cargar el modelo para ver si ya está instalado
        spacy.load(model_name)
        print(f"Modelo de spaCy '{model_name}' ya está instalado.")
    except OSError:
        # Si no está instalado, lo descarga e instala
        print(f"Descargando e instalando el modelo '{model_name}'...")
        subprocess.run(["python", "-m", "spacy", "download", model_name])

# Llama a la función para asegurarte de que el modelo está instalado
download_spacy_model()

# Cargar el modelo de spaCy
nlp = spacy.load("en_core_web_sm")

# Configuración de Google Gemini
GOOGLE_API_KEY = 'google_api_key'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Función para enriquecer el texto con análisis sintáctico y semántico
def enrich_text(text):
    doc = nlp(text)
    enriched_sentences = []
    for sent in doc.sents:
        entities = " ".join([ent.text for ent in sent.ents])
        deps = " ".join([token.dep_ for token in sent])
        enriched_sentence = f"{sent.text} {entities} {deps}"
        enriched_sentences.append(enriched_sentence)
    return " ".join(enriched_sentences)

# Función mejorada para encontrar oraciones relevantes
def find_relevant_sentences(question, text, num_sentences=5):
    nltk.download('punkt')
    enriched_text = enrich_text(text)
    sentences = nltk.sent_tokenize(enriched_text)
    vectorizer = TfidfVectorizer().fit([question] + sentences)
    question_vec = vectorizer.transform([question])
    sentences_vec = vectorizer.transform(sentences)

    similarities = cosine_similarity(question_vec, sentences_vec).flatten()

    # Mejora del sistema de ponderación
    sorted_indices = similarities.argsort()[::-1]
    relevant_indices = []
    for idx in sorted_indices:
        if len(relevant_indices) < num_sentences:
            relevant_indices.append(idx)
        else:
            break

    relevant_sentences = [sentences[i] for i in relevant_indices]
    return " ".join(relevant_sentences)

# Función para generar una respuesta con Gemini
def generate_response(query, context):
    prompt = f"Contexto: {context} Pregunta: {query}"
    response = model.generate_content(prompt)
    return response.text

# Interfaz de usuario de Streamlit
st.title("PDF Contextual Question Answering")
pdf_file = st.file_uploader("Sube un PDF", type=["pdf"])
query = st.text_input("Introduce tu pregunta")

if st.button("Enviar") and pdf_file is not None and query:
    pdf_content = BytesIO(pdf_file.read())
    with fitz.open(stream=pdf_content, filetype="pdf") as doc:
        text = " ".join(page.get_text() for page in doc)

    context = find_relevant_sentences(query, text)
    answer = generate_response(query, context)
    st.write(answer)
