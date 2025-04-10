# 🤖 AI-Powered Research Assistant

An intelligent research assistant built using **Streamlit**, **Langchain**, and **Groq's LLaMA3-70B** model. This tool allows you to upload research papers in PDF format, summarize them, and chat with your documents for instant insights.

---

## 🚀 Features

- 📄 Upload and process **multiple PDFs**
- 🧠 Generate **natural language summaries** of your papers
- 💬 Chat with documents using **ConversationalRetrievalChain**
- ⚡ Powered by **Groq API (LLaMA3-70B)** for lightning-fast inference
- 📚 Uses **Chroma vector store** with **HuggingFace embeddings**
- 💾 Saves chat history in local `sessions` folder by username
- 🔐 Environment variables managed with `.env`

---

## 📂 File Structure

ai-research-assistant/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
└── sessions/
