## 📦 Libraries and Their Usage

This project uses a combination of libraries for document processing, embedding generation, vector storage, LLM interaction, and UI development. Below is a component-wise breakdown:

---

### 📄 1. Document Loading & Preprocessing

#### `langchain_community.document_loaders`
- Provides ready-made loaders for multiple file formats
- Used to ingest documents into a standardized format

#### Loaders Used:
- `PyPDFLoader` → Loads and parses PDF files
- `TextLoader` → Handles `.txt` files
- `CSVLoader` → Loads structured CSV data
- `Docx2txtLoader` → Extracts text from Word documents
- `JSONLoader` → Parses JSON files into documents
- `UnstructuredExcelLoader` → Processes `.xlsx` files

📌 **Why used?**  
To support multi-format document ingestion for real-world enterprise data.

---

### ✂️ 2. Text Chunking

#### `langchain_text_splitters`
- Provides utilities to split large documents into smaller chunks

#### `RecursiveCharacterTextSplitter`
- Splits text into overlapping chunks
- Maintains context across chunks

📌 **Why used?**  
LLMs and embedding models have token limits, so large documents must be broken into smaller, meaningful segments.

---

### 🧠 3. Embedding Generation

#### `sentence_transformers`
- Library for generating dense vector embeddings from text

#### `SentenceTransformer`
- Model used: `all-MiniLM-L6-v2`
- Converts text into semantic vectors

#### `numpy`
- Used to store and manipulate embedding vectors

📌 **Why used?**  
To represent text semantically so similar content can be retrieved based on meaning, not keywords.

---

### 🗄️ 4. Vector Database (Storage & Retrieval)

#### `faiss`
- Facebook AI Similarity Search
- High-performance vector search library

#### `pickle`
- Serializes metadata for persistent storage

📌 **Why used?**  
- Stores embeddings efficiently
- Enables fast nearest-neighbor search for retrieval
- Supports saving and loading index from disk

---

### 🔍 5. Retrieval + LLM Integration

#### `langchain_groq`
- Integration layer for Groq LLM

#### `ChatGroq`
- Used to interact with Groq-hosted models (e.g., LLaMA)

#### `python-dotenv`
- Loads environment variables from `.env` file

📌 **Why used?**  
- Retrieves relevant context
- Sends structured prompts to LLM
- Ensures secure API key management

---

### 🖥️ 6. Frontend (User Interface)

#### `streamlit`
- Framework for building interactive web apps in Python

📌 **Why used?**
- Enables:
  - File upload
  - Query input
  - Dynamic responses
  - Visualization of results and citations

---

### ⚙️ 7. Core Python Libraries

#### `os`
- File handling and environment variable access

#### `sys`
- Used for path management

#### `pathlib`
- Cleaner file path operations

#### `typing`
- Type annotations for better code clarity

📌 **Why used?**  
To improve code readability, maintainability, and structure.

---

## 🧠 Summary

| Component            | Library Used                    | Purpose |
|---------------------|--------------------------------|--------|
| Document Loading     | LangChain Loaders              | Multi-format ingestion |
| Chunking             | RecursiveCharacterTextSplitter | Text segmentation |
| Embeddings           | Sentence Transformers          | Semantic representation |
| Vector Storage       | FAISS                          | Fast similarity search |
| LLM Integration      | Groq + LangChain               | Answer generation |
| UI                   | Streamlit                      | User interaction |
| Utilities            | NumPy, OS, Pathlib             | Data handling |

---

## 🚀 Why This Stack?

This stack was chosen to:
- Support **real-world document formats**
- Enable **fast semantic search**
- Ensure **low-latency LLM responses (Groq)**
- Maintain **modular and scalable architecture**
- Provide **interactive UI for usability**
