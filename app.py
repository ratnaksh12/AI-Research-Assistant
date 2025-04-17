import streamlit as st
import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
import os
import uuid
import json
from datetime import datetime

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("AI-Powered Research Assistant")

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Login Simulation
username = st.text_input("Enter your name to start or continue a session")
if username:
    session_file = f"sessions/{username}_session.json"
    os.makedirs("sessions", exist_ok=True)

    if os.path.exists(session_file):
        with open(session_file, "r") as f:
            saved_data = json.load(f)
            chat_history_raw = saved_data.get("chat_history", [])
            st.session_state.chat_history = [tuple(pair) for pair in chat_history_raw]
            st.success(f"Welcome back, {username}!")
    else:
        with open(session_file, "w") as f:
            json.dump({"chat_history": []}, f)
        st.success(f"Session started for {username}")

# Upload PDFs
uploaded_files = st.file_uploader("Upload one or more research papers (PDFs)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

    # Use only model_name (no manual loading)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    all_texts = []
    for uploaded_file in uploaded_files:
        file_id = str(uuid.uuid4())
        file_path = f"temp_{file_id}.pdf"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        all_texts.extend(texts)

    # Create FAISS index
    sample_embedding = embeddings.embed_documents(["This is a sample document."])[0]
    embedding_dimension = len(sample_embedding)
    faiss_index = faiss.IndexFlatL2(embedding_dimension)
    embeddings_matrix = np.array([embeddings.embed_document(text.page_content) for text in all_texts])
    faiss_index.add(embeddings_matrix)

    # Save FAISS index to disk
    faiss.write_index(faiss_index, "faiss_index.index")

    st.session_state.retriever = faiss_index  # Set the retriever to FAISS index
    st.success("PDFs processed and stored successfully!")

# QA Chat
if st.session_state.retriever:
    llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")

    qa_chain = ConversationalRetrievalChain.from_llm(llm, st.session_state.retriever, return_source_documents=True)

    query = st.text_input("Ask a question about your documents")
    if query:
        result = qa_chain({
            "question": query,
            "chat_history": st.session_state.chat_history
        })

        st.session_state.chat_history.append((query, result["answer"]))

        if username:
            with open(session_file, "w") as f:
                json.dump({"chat_history": st.session_state.chat_history}, f)

        st.markdown("### Answer:")
        st.write(result["answer"])

        if result["source_documents"]:
            st.markdown("#### Source(s):")
            for doc in result["source_documents"]:
                st.code(doc.page_content[:500] + "...")

# Chat History
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history[::-1]):
        st.markdown(f"**Q{i+1}:** {q}")
        st.markdown(f"**A{i+1}:** {a}")
