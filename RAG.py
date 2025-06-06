import streamlit as st
import os
import tempfile
# from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Optional, List, Any
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
# import chromadb


load_dotenv()

api_key = os.getenv("API_KEY")
# Configure Gemini
genai.configure(api_key=api_key)

# Custom Gemini LLM Wrapper
class GeminiLLM(LLM):
    model: Any = genai.GenerativeModel("gemini-1.5-flash")
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "google_gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.model.generate_content(prompt)
        return response.text

# Streamlit UI
st.title("📄 RAG App: Chat with Your Documents (Gemini)")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "edit_index" not in st.session_state:
    st.session_state.edit_index = None

uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "txt", "docx"], accept_multiple_files=True)

# Load documents and build retriever
if uploaded_files:
    raw_text = ""
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(tmp_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        else:
            continue
        documents = loader.load()
        for doc in documents:
            raw_text += doc.page_content + "\n"

    # Split and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = splitter.split_text(raw_text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(texts, embedding=embeddings)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=GeminiLLM(),
        retriever=retriever,
        return_source_documents=True
    )

    # Show chat history with edit buttons
    st.markdown("### 💬 Chat History")
    for i, chat in enumerate(st.session_state.chat_history):
        with st.expander(f"Q{i+1}: {chat['user']}"):
            st.markdown(f"**Gemini:** {chat['answer']}")
            if st.button("✏️ Edit this question", key=f"edit_{i}"):
                st.session_state.edit_index = i

    # Edit mode
    if st.session_state.edit_index is not None:
        edit_i = st.session_state.edit_index
        new_query = st.text_input("Edit your question:", value=st.session_state.chat_history[edit_i]['user'])
        if st.button("Regenerate Answer"):
            with st.spinner("Generating edited response..."):
                result = qa_chain({"query": new_query})
                st.session_state.chat_history[edit_i] = {
                    "user": new_query,
                    "answer": result["result"]
                }
                st.session_state.edit_index = None
                st.rerun()

    else:
        # Normal new question input
        query = st.text_input("Ask a new question about your documents:")

        if query:
            with st.spinner("Generating response..."):
                result = qa_chain({"query": query})
                answer = result["result"]

                # Append to history
                st.session_state.chat_history.append({
                    "user": query,
                    "answer": answer
                })

                # Show result
                st.markdown(f"**RAG App:** {answer}")


