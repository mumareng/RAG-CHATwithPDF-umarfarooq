import streamlit as st

st.title("RAG ")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)


import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
try:
    from langchain.embeddings import HuggingFaceEmbeddings
    print("HuggingFaceEmbeddings imported successfully!")
except ModuleNotFoundError:
    print("HuggingFaceEmbeddings not found.")
try:
    from langchain.vectorstores import FAISS
    print("FAISS imported successfully!")
except ModuleNotFoundError:
    print("FAISS not found.")

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

# Maximum number of free questions
MAX_QUESTIONS = 0

# Add a section in the sidebar for the user to input the OpenAI API key
st.sidebar.header("API Key")
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key (required)", type="password")

# If the OpenAI API key is provided, set it in the environment variable
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

html_temp = f"""
 <div style="position: fixed;text-align: center; top: 0; right: 0; width: 70%; height: auto;background-color:white;  padding: 10px; border-bottom: solid 1px #e0e0e0; z-index: 1000;">
   <h1 style="text-align: center; margin-top: 30px; color: black;">
     <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQIag-4EEBF1dYQ31wn5YTLj7mVZHThEJ0jvhwUdvjJTmTDXK79vSDnUdA_tyIW1tW8xbE&usqp=CAU" alt="Chat PDF" width="100" style="vertical-align: middle;"/> RAG BASED CHATPDF
   </h1>
   <p style="text-align: center; font-size: 20px; color: #3498db;">CHAT WITH PDF</p>
   <p style="text-align: center; color: black;"><a href="https://www.linkedin.com/in/muhammad-umar-farooq-85b497237/" target="_blank" style="color:  #3498db; text-decoration: none;">
    Developed by <strong>Muhammad Umar Farooq </strong>
  </a></p>
 </div>
 <div style="margin-top: 150px; margin-bottom: 15px;"> 
 </br>
 </br></br>
"""

# Add the fixed navbar to the Streamlit app
st.markdown(html_temp, unsafe_allow_html=True)

# Sidebar for file upload
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# Initialize session state for question-answer pairs and question count
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'question_count' not in st.session_state:
    st.session_state.question_count = 0

# Check if a file was uploaded
if uploaded_file is not None:
    # Read the PDF file using PyPDF2
    pdf_reader = PdfReader(uploaded_file)
    pdf_text = ""

    # Extract text from each page
    for page_num, page in enumerate(pdf_reader.pages):
        pdf_text += page.extract_text();

    # Split the extracted text into chunks
    def split_text(text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_text(text)

    text_chunks = split_text(pdf_text)

    # Generate embeddings using HuggingFace
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    embeddings = embedding_model.embed_documents(text_chunks)

    # Store embeddings in FAISS vector store
    vector_store = FAISS.from_texts(text_chunks, embedding_model)

    # Setup the OpenAI-based QA chain if API key is available
    if 'OPENAI_API_KEY' in os.environ:
        llm = OpenAI(temperature=0)
        retriever = vector_store.as_retriever(search_type="mmr")  # Use MMR for retrieval
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Display the extracted text in the right sidebar
    st.sidebar.write("### PDF Content:")
    st.sidebar.write(pdf_text)

    # Initialize session state for query if not exists
    if 'query' not in st.session_state:
        st.session_state.query = ""

    # Create a fixed input field for the question
    query = st.text_input("Ask a question about the document", value=st.session_state.query)

    if query:
        # Check if the number of questions is less than or equal to MAX_QUESTIONS
        if st.session_state.question_count < MAX_QUESTIONS:
            # Handle the query within the free question limit
            if query.lower() in pdf_text.lower():
                result = f"The document contains: {query}"
            else:
                st.write("### Retrieving relevant information...")
                result = "OpenAI API required for retrieval "  # Handle without OpenAI

            # Store the question and answer in session state
            st.session_state.qa_history.append((query, result))

            # Increase the question count
            st.session_state.question_count += 1

        # If the user reaches the limit and no API key is provided, show a warning
        elif st.session_state.question_count >= MAX_QUESTIONS and not openai_api_key:
            st.warning("Please enter your OpenAI API key to continue.")

        # Handle additional questions if API key is provided
        if openai_api_key and st.session_state.question_count >= MAX_QUESTIONS:
            if query.lower() in pdf_text.lower():
                result = f"The document contains: {query}"
            else:
                st.write("### Retrieving relevant information...")
                result = qa_chain.run(query)

            # Store the question and answer in session state
            st.session_state.qa_history.append((query, result))

    # Display the history of questions and answers in descending order
    if st.session_state.qa_history:
        st.write("### Question and Answer History:")

        # Iterate over the history in reverse order for descending display
        for idx, (question, answer) in reversed(list(enumerate(st.session_state.qa_history))):
            # Display user image and question
            col1, col2 = st.columns([1, 9])  # Create two columns
            with col1:
                st.image("user.jpeg", width=50)  # Display user's image
            with col2:
                st.write(f"**You:** {question}")  # Display the question

            # Display chatbot image and answer
            col1, col2 = st.columns([1, 9])  # Create two columns
            with col1:
                st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQIag-4EEBF1dYQ31wn5YTLj7mVZHThEJ0jvhwUdvjJTmTDXK79vSDnUdA_tyIW1tW8xbE&usqp=CAU", width=50)  # Display chatbot's image
            with col2:
                st.write(f"**ChatPDF:** {answer}")  # Display the answer

            st.write("")  # Add a blank line for spacing


st.markdown("</div>", unsafe_allow_html=True)
