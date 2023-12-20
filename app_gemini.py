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
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

# Diccionario de modelos de spaCy por idioma
SPACY_MODELS = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "it": "it_core_news_sm"
}

# Descargar recursos de NLTK
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

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
GOOGLE_API_KEY = 'AIzaSyAkbU3CsZ-xmOhRF1XfdlVxasRtt9gdRMk'
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

# Función para extraer palabras clave
def extract_keywords(text, pos_tags=['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']):
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    keywords = [word for word, tag in tagged if tag in pos_tags]
    return keywords

# Función para obtener sinónimos de una palabra
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

# Función mejorada para encontrar información relevante
def find_relevant_information(question, text, nlp):
    # Extraer palabras clave de la pregunta
    question_keywords = extract_keywords(question)

    # Expandir las palabras clave con sinónimos
    expanded_keywords = set(question_keywords)
    for word in question_keywords:
        expanded_keywords.update(get_synonyms(word))

    # Crear un PhraseMatcher en spaCy para encontrar coincidencias en el texto
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(keyword) for keyword in expanded_keywords]
    matcher.add("Keywords", patterns)

    # Procesar el texto y encontrar oraciones relevantes
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

# Función para generar una respuesta con Gemini, con reintento y manejo de chunks
def generate_response(query, context):
    max_retries = 3
    chunk_size = 5000  # Tamaño del chunk, ajustar según las limitaciones de la API
    chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]
    combined_response = ""

    for chunk in chunks:
        for attempt in range(max_retries):
            try:
                prompt = f"Contexto: {chunk} Pregunta: {query}"
                response = model.generate_content(prompt)
                combined_response += response.text + " "
                break
            except ValueError as e:
                if "none were returned" in str(e):
                    combined_response += "Lo siento, no puedo procesar esa pregunta debido a las restricciones de políticas de Google Gemini. "
                    break
                elif attempt < max_retries - 1:
                    print(f"Error detectado, intentando de nuevo (Intento {attempt + 2}/{max_retries})...")
                else:
                    combined_response += "Esta búsqueda no es permitida por las políticas y condiciones de Google Gemini. "
                    break

    # Limpiar y re-procesar la respuesta combinada
    final_context = " ".join(combined_response.split())
    try:
        final_prompt = f"Contexto: {final_context} Pregunta: {query}"
        final_response = model.generate_content(final_prompt)
        return final_response.text
    except ValueError as e:
        return "Error al procesar la respuesta final: " + str(e)

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
