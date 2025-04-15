import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("🧠 AI Research Assistant")
st.markdown("Upload PDFs of research papers and ask questions about them.")

# User name input field
user_name = st.text_input("Enter your name to start or continue a session")

# File uploader for PDFs
uploaded_files = st.file_uploader("Upload one or more research papers (PDFs)", accept_multiple_files=True, type="pdf")

# Process PDFs when uploaded
if uploaded_files:
    raw_text = ""
    for pdf in uploaded_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content

    # Split the text into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    texts = text_splitter.split_text(raw_text)

    # Create HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Use FAISS for storing vector embeddings
    vector_db = FAISS.from_texts(texts, embedding=embeddings)

    # Initialize Groq LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")  # Use an appropriate model

    # Load QA chain
    chain = load_qa_chain(llm, chain_type="stuff")

    # Ask user for a question
    query = st.text_input("Ask a question about the uploaded documents:")
    if query:
        docs = vector_db.similarity_search(query)
        response = chain.run(input_documents=docs, question=query)
        st.write("**Answer:**")
        st.write(response)
