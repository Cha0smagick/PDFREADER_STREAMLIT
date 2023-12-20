import streamlit as st
import os
import nltk
import fitz  # PyMuPDF
from io import BytesIO
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.matcher import PhraseMatcher
import subprocess
from langdetect import detect
from deep_translator import GoogleTranslator

# Diccionario de modelos de spaCy por idioma
SPACY_MODELS = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "it": "it_core_news_sm"
}

# Función para descargar el modelo de spaCy si aún no está instalado
def download_spacy_model(language_code):
    model_name = SPACY_MODELS.get(language_code, "en_core_web_sm")
    try:
        spacy.load(model_name)
        print(f"Modelo de spaCy '{model_name}' ya está instalado.")
    except OSError:
        print(f"Descargando e instalando el modelo '{model_name}'...")
        subprocess.run(["python", "-m", "spacy", "download", model_name])
        spacy.load(model_name)  # Cargar el modelo después de instalarlo

# Configuración de Google Gemini
GOOGLE_API_KEY = 'google_api_key'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Función para detectar el idioma de un texto
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"  # Devolver inglés por defecto si la detección falla

# Función para traducir texto
def translate_text(text, target_lang):
    translator = GoogleTranslator(source='auto', target=target_lang)
    return translator.translate(text)

# Función para enriquecer el texto con análisis sintáctico, semántico y estructural
def enrich_text(text, nlp):
    doc = nlp(text)
    enriched_sentences = []
    for sent in doc.sents:
        entities = " ".join([ent.text for ent in sent.ents])
        deps = " ".join([token.dep_ for token in sent])

        is_title = sent.text.isupper() or any(token.is_title for token in sent)

        enriched_sentence = f"{sent.text} {entities} {deps}"
        if is_title:
            enriched_sentence += " TITLE_OR_SUBTITLE"

        enriched_sentences.append(enriched_sentence)
    return " ".join(enriched_sentences)

# Función mejorada para encontrar información relevante
def find_relevant_information(question, text, nlp):
    # Descargar punkt para nltk, si aún no está instalado
    nltk.download('punkt', quiet=True)

    # Procesar la pregunta para extraer entidades
    question_doc = nlp(question)
    question_entities = [ent.text for ent in question_doc.ents]

    # Crear un PhraseMatcher en spaCy para encontrar coincidencias de entidades en el texto
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(entity) for entity in question_entities]
    matcher.add("Entities", patterns)

    # Procesar el texto completo y encontrar oraciones que contengan entidades similares
    doc = nlp(text)
    relevant_sentences = set()
    for sent in doc.sents:
        matches = matcher(sent.as_doc())
        if matches:
            relevant_sentences.add(sent.text.strip())

    # Si no se encuentran oraciones relevantes, usar el enfoque de TF-IDF como respaldo
    if not relevant_sentences:
        enriched_text = enrich_text(text, nlp)
        sentences = nltk.sent_tokenize(enriched_text)

        vectorizer = TfidfVectorizer().fit([question] + sentences)
        question_vec = vectorizer.transform([question])
        sentences_vec = vectorizer.transform(sentences)

        similarities = cosine_similarity(question_vec, sentences_vec).flatten()
        sorted_indices = similarities.argsort()[::-1]
        relevant_indices = sorted_indices[:5]

        relevant_sentences = [sentences[i] for i in relevant_indices]

    return " ".join(relevant_sentences)

# Función para generar una respuesta con Gemini
def generate_response(query, context):
    try:
        prompt = f"Contexto: {context} Pregunta: {query}"
        response = model.generate_content(prompt)
        return response.text
    except ValueError as e:
        if "none were returned" in str(e):
            return "Lo siento, no puedo procesar esa pregunta debido a las restricciones de políticas de Google Gemini."
        else:
            raise e

# Interfaz de usuario de Streamlit
st.title("PDF Contextual Question Answering")
pdf_file = st.file_uploader("Sube un PDF", type=["pdf"])
query = st.text_input("Introduce tu pregunta")

if st.button("Enviar") and pdf_file is not None and query:
    pdf_content = BytesIO(pdf_file.read())
    text = ""
    with fitz.open(stream=pdf_content, filetype="pdf") as doc:
        text = " ".join(page.get_text() for page in doc)

    detected_lang = detect_language(text)
    download_spacy_model(detected_lang)
    nlp = spacy.load(SPACY_MODELS.get(detected_lang, "en_core_web_sm"))

    translated_query = translate_text(query, detected_lang)
    context = find_relevant_information(translated_query, text, nlp)

    answer = generate_response(query, context)
    st.write(answer)
