import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI
st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("📄 AI Research Assistant")
st.caption("Upload PDFs of research papers and ask questions about them.")

# File uploader
uploaded_files = st.file_uploader("Upload one or more research papers (PDFs)", accept_multiple_files=True, type="pdf")

# Load documents from PDFs
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

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vector store
    vectordb = Chroma.from_texts(texts, embedding=embeddings)

    # Groq LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")  # Or use gemma-7b

    # Load QA chain
    chain = load_qa_chain(llm, chain_type="stuff")

    # User query
    query = st.text_input("Ask a question about the uploaded documents:")
    if query:
        docs = vectordb.similarity_search(query)
        response = chain.run(input_documents=docs, question=query)
        st.write("**Answer:**")
        st.write(response)
