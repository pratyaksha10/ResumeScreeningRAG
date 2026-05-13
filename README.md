# 🚀 Anything-RAG: Advanced Resume Screening Assistant

Welcome to **Anything-RAG**, a powerful, production-ready AI assistant designed to revolutionize the resume screening process for hiring managers and recruiters. 

Built with modern AI tools, this platform moves beyond simplistic keyword searches and leverages the power of **Retrieval-Augmented Generation (RAG)** and **Knowledge Graphs** to semantically understand, retrieve, and compare candidate profiles at a human-like level of comprehension.

## ✨ Key Features & Enhancements

This application has been significantly upgraded from traditional RAG designs to include robust visual analytics and flexible model integration:

- 🧠 **Model Agnostic (OpenRouter Integration)**: Connect to top-tier LLMs like Gemini 2.0 Flash, Claude 3, or GPT-4 seamlessly by dropping in your OpenRouter API key.
- 📁 **Universal File Support**: Upload resumes in multiple formats! The system accepts **PDFs, Excel (.xlsx, .xls), and CSV** files, extracting text and indexing them on the fly.
- 🌐 **Interactive Vector Space Visualizer**: Ever wonder *why* a resume was retrieved? The built-in PCA visualizer plots your query and the entire resume dataset in a 2D space, highlighting exactly how close top candidates are to your ideal job description.
- 📊 **Dynamic Knowledge Graphs**: With the click of a button, generate an interactive physics-based Knowledge Graph (powered by `pyvis` and `networkx`) that maps out retrieved candidates and their shared technical skills, revealing hidden patterns in your applicant pool.
- 🔍 **Hybrid Retrieval Pipeline**: Utilizes both standard similarity-based retrieval and **RAG Fusion** to handle complex, multifaceted job descriptions effectively.
- 💬 **Context-Aware Chat**: The assistant maintains conversation history, allowing you to ask follow-up questions, request candidate summaries, or perform cross-comparisons naturally.

## 🛠️ Technology Stack

- **RAG Framework**: `langchain`, `langchain-community`, `langchain-core`
- **Vector Database**: `FAISS` for high-speed similarity search
- **Embeddings**: `HuggingFaceEmbeddings` (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
- **Frontend & UI**: `streamlit` for the tabbed interactive web interface
- **Visualizations**: `plotly` (Vector Space) and `pyvis` + `networkx` (Knowledge Graphs)
- **File Parsing**: `PyPDF2`, `openpyxl`, `pandas`

## 🚀 Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/pratyaksha10/ResumeScreeningRAG.git
cd ResumeScreeningRAG
```

### 2. Environment Configuration
Create an `.env` file in the root directory (you can use `.env.example` as a template):
```env
DATA_PATH = "./data/supplementary-data/pdf-resumes.csv"
FAISS_PATH = "./vectorstore-pdf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENROUTER_API_KEY = "your_openrouter_api_key_here"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Ensure you also have `pyvis`, `networkx`, `PyPDF2`, and `openpyxl` installed as per the latest updates).*

### 4. Run the Application
Launch the Streamlit web interface locally:
```bash
python -m streamlit run demo/interface.py
```

## 👨‍💻 About the Developer

This advanced pipeline is maintained and developed by **Pratyaksha**. 

I am passionate about building intelligent systems and pushing the boundaries of what AI can do in real-world applications. Feel free to connect with me or check out my other work!

- **Portfolio**: [pratyaksha-11.vercel.app](https://pratyaksha-11.vercel.app/)
- **GitHub**: [pratyaksha10](https://github.com/pratyaksha10/)
- **LinkedIn**: [Pratyaksha](https://www.linkedin.com/in/pratyaksha-pratyaksha-079226190/)

---
*If you find this project helpful or interesting, please don't hesitate to give the repository a ⭐!*
