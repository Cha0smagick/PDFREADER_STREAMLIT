import streamlit as st
import os
from io import BytesIO, StringIO
import tempfile
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import fitz
from PIL import Image

# Global variables
count = 0
n = 0
chat_history = []
chain = ''
pdf_image_container = st.empty()  # Container for displaying PDF image

# Set your OpenAI API key here
os.environ['OPENAI_API_KEY'] = 'your_api_key_openai'

# Streamlit UI setup
st.title("")  # Clear the default title
st.image("https://i.ibb.co/2j2gGW1/logo-inif.png", use_column_width=True)  # Display the logo centered

# Add the app name centered
st.markdown("<h1 style='text-align: center;'>INIFREADER</h1>", unsafe_allow_html=True)

# Function to process the PDF file and create a conversation chain
def process_file(file_content):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(file_content)
        temp_pdf_path = temp_pdf.name

    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load()

    embeddings = OpenAIEmbeddings()

    pdf_search = Chroma.from_documents(documents, embeddings)

    chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.3),
                                                  retriever=pdf_search.as_retriever(search_kwargs={"k": 1}),
                                                  return_source_documents=True)
    return chain

# Function to generate a response based on the chat history and query
def generate_response(query, pdf_content):
    global count, n, chat_history, chain

    if not pdf_content:
        st.write("Upload a PDF")
        return

    if count == 0:
        chain = process_file(pdf_content)
        count += 1

    result = chain({"question": query, 'chat_history': chat_history}, return_only_outputs=True)
    chat_history.append((query, result["answer"]))
    n = list(result['source_documents'][0])[1][1]['page']

    response_text = ''
    for char in result['answer']:
        response_text += char

    st.write(response_text)

    pdf_data = BytesIO(pdf_content)
    doc = fitz.open(stream=pdf_data.read(), filetype="pdf")
    page = doc[n]
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    
    pdf_image_container.image(image)  # Display PDF image

# Streamlit UI elements
query = st.text_input("Ask your PDF?")
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
if st.button("Send"):
    pdf_content = pdf_file.read()
    generate_response(query, pdf_content)
