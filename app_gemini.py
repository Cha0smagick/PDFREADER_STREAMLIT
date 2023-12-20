import streamlit as st
import os
import nltk
import fitz  # PyMuPDF
from io import BytesIO
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuración de Google Gemini
GOOGLE_API_KEY = 'google_api_key'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Función para encontrar oraciones relevantes en el texto
def find_relevant_sentences(question, text, num_sentences=5):
    nltk.download('punkt')
    sentences = nltk.sent_tokenize(text)
    vectorizer = TfidfVectorizer().fit([question] + sentences)
    question_vec = vectorizer.transform([question])
    sentences_vec = vectorizer.transform(sentences)

    similarities = cosine_similarity(question_vec, sentences_vec).flatten()
    relevant_indices = similarities.argsort()[-num_sentences:][::-1]
    
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
