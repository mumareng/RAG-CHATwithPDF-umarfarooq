import streamlit as st
import os
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai

# Constants
MAX_QUESTIONS = 15

# Sidebar for API Key
st.sidebar.header("API Key")
gemini_api_key = st.sidebar.text_input("Enter your Gemini API Key (required)", type="password")
if gemini_api_key:
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    genai.configure(api_key=gemini_api_key)

# Fixed top navbar
html_temp = """
<div style="position: fixed; text-align: center; top: 0; right: 0; width: 70%; height: auto; background-color: white; padding: 10px; border-bottom: solid 1px #e0e0e0; z-index: 1000;">
   <h1 style="text-align: center; margin-top: 30px; color: black;">
     <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQIag-4EEBF1dYQ31wn5YTLj7mVZHThEJ0jvhwUdvjJTmTDXK79vSDnUdA_tyIW1tW8xbE&usqp=CAU" alt="Chat PDF" width="100" style="vertical-align: middle;"/> RAG BASED CHATPDF
   </h1>
   <p style="text-align: center; font-size: 20px; color: #3498db;">CHAT WITH PDF</p>
   <p style="text-align: center; color: black;">
     <a href="https://www.linkedin.com/in/muhammad-umar-farooq-85b497237/" target="_blank" style="color: #3498db; text-decoration: none;">
       Developed by <strong>Muhammad Umar Farooq</strong>
     </a>
   </p>
</div>
<div style="margin-top: 150px; margin-bottom: 15px;"></div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# Sidebar for PDF Upload
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# Session states
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'question_count' not in st.session_state:
    st.session_state.question_count = 0
if 'query' not in st.session_state:
    st.session_state.query = ""

# If a file is uploaded
if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    pdf_text = ""

    for page in pdf_reader.pages:
        pdf_text += page.extract_text() or ""

    # Split the extracted text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = splitter.split_text(pdf_text)

    # Create embeddings and FAISS vector store
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        vector_store = FAISS.from_texts(text_chunks, embedding_model)
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        st.stop()

    # Input for the query
    query = st.text_input("Ask a question about the document", value=st.session_state.query)
    st.session_state.query = query

    if query:
        if st.session_state.question_count < MAX_QUESTIONS or gemini_api_key:
            try:
                # Retrieve context from FAISS
                docs = vector_store.similarity_search(query, k=3)
                context = "\n".join([doc.page_content for doc in docs])

                # Gemini model call
                model = genai.GenerativeModel(model_name="models/gemini-pro")
                response = model.generate_content(
                    f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion:\n{query}"
                )
                result = response.text

            except Exception as e:
                result = f"Error using Gemini: {e}"

            # Store in session state
            st.session_state.qa_history.append((query, result))
            st.session_state.question_count += 1
        else:
            st.warning("Please enter your Gemini API key to continue.")

    # Display Q/A history
    if st.session_state.qa_history:
        st.write("### Question and Answer History:")
        for idx, (question, answer) in reversed(list(enumerate(st.session_state.qa_history))):
            col1, col2 = st.columns([1, 9])
            with col1:
                try:
                    st.image("user.jpeg", width=50)
                except:
                    st.write("🧑‍💻")
            with col2:
                st.write(f"**You:** {question}")

            col1, col2 = st.columns([1, 9])
            with col1:
                st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQIag-4EEBF1dYQ31wn5YTLj7mVZHThEJ0jvhwUdvjJTmTDXK79vSDnUdA_tyIW1tW8xbE&usqp=CAU", width=50)
            with col2:
                st.write(f"**ChatPDF:** {answer}")

