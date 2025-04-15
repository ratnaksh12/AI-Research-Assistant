import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq  # type: ignore
from langchain.prompts import PromptTemplate
import os
import uuid
from datetime import datetime
import json

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
            st.session_state.chat_history = [tuple(pair) for pair in chat_history_raw]  # Convert list -> tuple
            st.success(f"Welcome back, {username}!")
    else:
        with open(session_file, "w") as f:
            json.dump({"chat_history": []}, f)
        st.success(f"Session started for {username}")

# Upload PDFs
uploaded_files = st.file_uploader("Upload one or more research papers (PDFs)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_texts = []
    for uploaded_file in uploaded_files:
        file_id = str(uuid.uuid4())
        file_path = f"temp_{file_id}.pdf"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        all_texts.extend(texts)

    embeddings = HuggingFaceEmbeddings()
    vectordb = Chroma.from_documents(all_texts, embedding=embeddings, persist_directory="db")
    vectordb.persist()
    st.session_state.vectordb = vectordb
    st.session_state.retriever = vectordb.as_retriever()
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

        # Save as list (JSON-compatible, will convert back to tuple on load)
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
