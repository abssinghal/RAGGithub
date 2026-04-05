from __future__ import annotations
from dotenv import load_dotenv
import os

load_dotenv()
#  .\.venv\Scripts\Activate.ps1
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))

from src.data_loader import SUPPORTED_EXTENSIONS, list_supported_files, load_all_documents
from src.search import RAGSearch
from src.vectorstore import FaissVectorStore

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAISS_DIR = BASE_DIR / "faiss_store"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner=False)
def get_rag_engine() -> RAGSearch:
    return RAGSearch(data_dir=str(DATA_DIR), persist_dir=str(FAISS_DIR))


def save_uploaded_files(uploaded_files) -> int:
    saved_count = 0
    for uploaded_file in uploaded_files:
        destination = DATA_DIR / uploaded_file.name
        with open(destination, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_count += 1
    return saved_count


def human_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def render_file_preview(files):
    st.subheader("Stored documents")
    st.write(f"Total files in data folder: **{len(files)}**")
    if not files:
        st.info("No supported files found in the data folder yet.")
        return

    for file_info in files:
        with st.expander(f"{file_info['name']}  ·  {human_size(file_info['size_bytes'])}"):
            st.write(f"**Path:** `{file_info['path']}`")
            st.write(f"**Type:** {file_info['suffix']}")


def render_answer(answer: dict):
    st.subheader("Answer summary")
    st.write(answer["summary"])

    st.subheader("Documents used")
    if answer["documents"]:
        for doc in answer["documents"]:
            st.write(f"- {doc}")
    else:
        st.write("No documents matched.")

    st.subheader("Citations")
    if answer["citations"]:
        for citation in answer["citations"]:
            page_text = f", page {citation['page']}" if citation["page"] else ""
            chunk_text = f", chunk {citation['chunk_id']}" if citation["chunk_id"] else ""
            st.markdown(
                f"**[{citation['id']}] {citation['file_name']}**{page_text}{chunk_text}  \n"
                f"`{citation['source']}`"
            )
            if citation["excerpt"]:
                st.caption(citation["excerpt"])
    else:
        st.write("No citations available.")


st.title("📚 RAG Document Assistant")
st.write(
    "Upload files, keep them in the `data/` folder, preview what is already there, and ask questions with grounded citations."
)

with st.sidebar:
    st.header("Upload documents")
    uploaded_files = st.file_uploader(
        "Add one or more files",
        type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
    )

    if st.button("Save uploaded files", use_container_width=True):
        if not uploaded_files:
            st.warning("Pick at least one file before saving.")
        else:
            saved = save_uploaded_files(uploaded_files)
            try:
                engine = get_rag_engine()
                engine.refresh_index()
                st.success(f"Saved {saved} file(s) and refreshed the vector index.")
            except Exception as exc:
                st.error(f"Files were saved, but indexing failed: {exc}")

    if st.button("Refresh index from data folder", use_container_width=True):
        try:
            engine = get_rag_engine()
            engine.refresh_index()
            st.success("Vector index refreshed from the current data folder.")
        except Exception as exc:
            st.error(f"Failed to refresh index: {exc}")

files = list_supported_files(DATA_DIR)
left, right = st.columns([1, 1.2])

with left:
    render_file_preview(files)

with right:
    st.subheader("Ask a question")
    query = st.text_area(
        "Dynamic query input",
        placeholder="Example: What is LangServe for API deployment?",
        height=120,
    )
    top_k = st.slider("How many chunks to retrieve", min_value=1, max_value=10, value=4)

    if st.button("Get answer", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Enter a query first.")
        else:
            try:
                engine = get_rag_engine()
                answer = engine.answer_query(query=query.strip(), top_k=top_k)
                render_answer(answer)
            except Exception as exc:
                st.error(f"Query failed: {exc}")
