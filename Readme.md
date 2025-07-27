# 📄 RAG App: Chat with Your Documents (Gemini + Qdrant)

This is a Streamlit-based RAG (Retrieval-Augmented Generation) application that allows users to upload documents (`.pdf`, `.docx`, `.txt`) and interact with them through a chatbot powered by **Google Gemini** and **LangChain**. It leverages **Qdrant** for semantic vector search and **HuggingFace embeddings** to enable intelligent document understanding.

🔗 **Live App**: [Click here to try it out](https://vikram-353-rag-app-rag-kmjgoa.streamlit.app/)

---

## 🚀 Features

- Upload multiple document formats: `.pdf`, `.docx`, `.txt`
- Automatic text extraction and chunking using LangChain
- Embedding generation via `all-MiniLM-L6-v2` from HuggingFace
- In-memory vector store using Qdrant for fast similarity search
- Google Gemini-based LLM answering your document queries
- Editable chat history and re-generation of answers

---

## 🧠 Tech Stack

- [Streamlit](https://streamlit.io/) – UI framework
- [LangChain](https://www.langchain.com/) – RAG pipeline and document loaders
- [Qdrant](https://qdrant.tech/) – Vector database (in-memory mode used here)
- [HuggingFace Transformers](https://huggingface.co/) – Sentence embeddings
- [Google Generative AI (Gemini)](https://ai.google.dev/) – LLM
- Python 3.10+

---

## ⚙️ Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/Vikram-353/RAG-APP/
   cd RAG-APP
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Create .env file**

   ```bash
   API_KEY=your_gemini_api_key
   ```

4. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

## 📁 Folder Structure

```bash

    .
    ├── app.py               # Main Streamlit application
    ├── requirements.txt     # Python dependencies
    └── .env                 # Environment variables (not committed)

    ```
