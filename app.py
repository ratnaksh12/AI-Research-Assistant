import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq  # type: ignore
from langchain.prompts import PromptTemplate
import os
import uuid
from datetime import datetime
import json
import faiss
import numpy as np
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("AI-Powered Research Assistant")

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Login Simulation (replace with Firebase/Auth for production)
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
    embeddings_list = []
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

        # Generate embeddings for each text chunk
        embeddings_list.extend([embeddings.embed_document(doc.page_content) for doc in texts])

    # Convert the embeddings to a numpy array for FAISS
    embeddings_array = np.array(embeddings_list).astype('float32')

    # Build FAISS index
    dim = embeddings_array.shape[1]  # Dimension of the embeddings
    faiss_index = faiss.IndexFlatL2(dim)  # Using L2 distance for similarity
    faiss_index.add(embeddings_array)

    # Store the FAISS index in session state
    st.session_state.vectordb = faiss_index
    st.session_state.all_texts = all_texts
    st.success("PDFs processed and stored successfully!")

# Document Summary Generator
if uploaded_files and st.button("Generate Summary"):
    all_text = "\n".join([doc.page_content for doc in documents])
    summary_prompt = PromptTemplate.from_template("Summarize the following document in simple terms:\n{text}")
    llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")

    summary = llm.predict(summary_prompt.format(text=all_text[:3000]))  # Limit text size
    st.markdown("### Document Summary")
    st.write(summary)

# QA Chat
if st.session_state.vectordb:
    llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")

    query = st.text_input("Ask a question about your documents")
    if query:
        # Convert the query to embedding
        query_embedding = embeddings.embed_query(query)
        query_embedding = np.array(query_embedding).astype('float32')

        # Search the FAISS index for the nearest neighbors
        _, indices = st.session_state.vectordb.search(query_embedding, k=3)  # Get top 3 most similar texts

        # Collect the most similar texts
        retrieved_docs = [st.session_state.all_texts[idx] for idx in indices]

        # Create a Conversational Retrieval Chain
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retrieved_docs, return_source_documents=True)

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
