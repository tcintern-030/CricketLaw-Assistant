# 🏏 CricketLaw Assistant

**CricketLaw Assistant** is a **Retrieval-Augmented Generation (RAG)** based application that answers questions about the **Laws of Cricket** using the provided MCC Laws of Cricket document.

The project demonstrates how **LangChain, ChromaDB, local embeddings, Gemini, FastAPI, and a web frontend** can be combined to build a domain-specific AI assistant.

> **Note:** This project currently focuses on the **MCC Laws of Cricket** and is intended as a learning project for understanding and implementing a complete RAG workflow.

---

## 📌 Project Overview

Instead of relying only on the knowledge of an LLM, CricketLaw Assistant retrieves relevant sections from the MCC Laws of Cricket and provides them as context to the LLM.

### RAG Workflow

```text
                 MCC Laws of Cricket PDF
                           │
                           ▼
                    Document Loader
                           │
                           ▼
                    Text Splitting
                           │
                           ▼
                  Local Embeddings
                (all-MiniLM-L6-v2)
                           │
                           ▼
                       ChromaDB
                           │
                           │
                    ─── RAG ───
                           │
                           ▼
                    User Question
                           │
                           ▼
                      Retriever
                           │
                           ▼
                 Relevant Law Sections
                           │
                           ▼
                     Gemini LLM
                           │
                           ▼
                  Grounded Response
                           │
                           ▼
                    Web Frontend
```

---

# ✨ Features

* 🏏 Ask questions about the Laws of Cricket
* 📄 Uses the MCC Laws of Cricket as the knowledge source
* 🔍 Semantic retrieval using vector similarity
* 🧠 Local HuggingFace embeddings
* 🗄️ ChromaDB vector database
* 🤖 Gemini LLM for answer generation
* ⚡ FastAPI backend
* 🌐 Modern responsive web frontend
* 📚 Displays retrieved source sections
* 🔄 Complete document ingestion and retrieval pipelines
* 🔐 API key stored using environment variables
* ⚠️ Provides an informational disclaimer

---

# 🛠️ Technologies Used

### Backend

* Python
* FastAPI
* Uvicorn

### RAG

* LangChain
* LangChain Community
* LangChain Text Splitters
* ChromaDB
* HuggingFace Embeddings
* Sentence Transformers

### LLM

* Google Gemini API

### Frontend

* HTML
* CSS
* JavaScript

### Document Processing

* PyPDF
* LangChain `PyPDFLoader`

---

# 📂 Project Structure

```text
CricketLaw-Assistant/
│
├── laws/
│   └── Laws Of cricket.pdf
│
├── vector_db/
│   └── ChromaDB data
│
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── Injection.py
├── Retrieval.py
├── main.py
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

# 🔄 RAG Pipeline

## 1. Document Loading

The MCC Laws of Cricket PDF is loaded using LangChain's `PyPDFLoader`.

```text
PDF
 ↓
PyPDFLoader
 ↓
LangChain Documents
```

Each page is converted into a LangChain `Document`.

---

## 2. Text Splitting

The documents are divided into smaller chunks using:

```text
RecursiveCharacterTextSplitter
```

This makes the document easier to search semantically.

The project uses approximately:

```text
Chunk Size: 1000
Chunk Overlap: 0
```

---

## 3. Embeddings

Instead of using Gemini embeddings, this project uses a **local HuggingFace embedding model**:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This has an important advantage:

```text
No Gemini embedding API requests
        ↓
No Gemini embedding quota consumption
        ↓
Embeddings generated locally
```

---

## 4. Vector Database

The generated embeddings are stored in **ChromaDB**.

```text
Document Chunks
       ↓
Embeddings
       ↓
ChromaDB
```

The vector database is stored locally in:

```text
vector_db/
```

---

# 🔎 Retrieval Pipeline

When the user asks a question:

```text
User Question
      ↓
Question Embedding
      ↓
ChromaDB
      ↓
Similarity Search
      ↓
Relevant MCC Sections
```

The retrieved documents are then passed to the Gemini LLM as context.

---

# 🤖 Response Generation

The LLM receives:

```text
Context
+
User Question
```

and generates an answer based on the retrieved MCC Laws.

This helps keep the response grounded in the provided document rather than relying entirely on the model's general knowledge.

---

# ⚡ FastAPI

The backend exposes an API endpoint:

```text
POST /ask
```

Example request:

```json
{
    "question": "What is Law 40?"
}
```

Example response:

```json
{
    "question": "What is Law 40?",
    "answer": "Law 40 deals with Timed Out..."
}
```

The API can also return retrieved source information when configured to do so.

---

# 🌐 Frontend

The project includes a responsive web interface built with:

* HTML
* CSS
* JavaScript

The frontend allows users to:

1. Enter a cricket-law question.
2. Send the question to FastAPI.
3. Display the generated answer.
4. Display retrieved sources.
5. View a disclaimer.
6. Try suggested questions.

The frontend is served through FastAPI.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd CricketLaw-Assistant
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

The Gemini API key is **not included in the repository**.

Create a file named:

```text
.env
```

in the root directory.

Add:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Replace:

```text
your_gemini_api_key
```

with your actual Gemini API key.

### Important

Do **not** commit `.env` to GitHub.

Add it to `.gitignore`:

```text
.env
```

---

# 📥 Build the Vector Database

Before running the application for the first time, run the injection pipeline:

```bash
python Injection.py
```

This performs:

```text
MCC PDF
   ↓
Load Documents
   ↓
Split Documents
   ↓
Generate Local Embeddings
   ↓
Store in ChromaDB
```

After successful execution, the `vector_db` directory will contain the generated vector database.

---

# ▶️ Run the Application

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The application will run locally.

### API Documentation

Open:

```text
http://127.0.0.1:8000/docs
```

This opens the FastAPI Swagger UI.

### Frontend

Open:

```text
http://127.0.0.1:8000/app
```

You can now ask questions through the CricketLaw Assistant interface.

---

# 🧪 Example Questions

You can test the application with questions such as:

```text
What is Law 40?
```

```text
What does Timed Out mean?
```

```text
What is a no-ball?
```

```text
What is a wide ball?
```

```text
What is LBW?
```

```text
What are the ways a batter can be dismissed?
```

```text
What happens when a batter is timed out?
```

```text
What is the role of the umpire?
```

The quality of an answer depends on whether the relevant information exists in the provided MCC document and whether the retriever successfully retrieves the relevant section.

---

# 🧩 Why Local Embeddings?

Initially, Gemini embeddings were considered for the project.

However, the Gemini embedding API has request quotas. During the ingestion of the MCC document, the project encountered a `429 ResourceExhausted` quota error because the document contained many chunks.

Therefore, the project uses:

```text
HuggingFace
    ↓
all-MiniLM-L6-v2
    ↓
Local Embeddings
```

The Gemini API is still used for **LLM response generation**.

This gives the project:

* Local embedding generation
* No embedding API cost
* No embedding API rate-limit dependency
* Suitable performance for a small document collection

---

# 🛡️ Disclaimer

> **Disclaimer:** CricketLaw Assistant provides information based on the provided MCC Laws of Cricket document for educational and informational purposes. It is not a substitute for the official Laws of Cricket, competition regulations, umpiring guidance, or professional advice. Always refer to the applicable official rules and regulations for authoritative decisions.

---

# 🚀 Future Improvements

Possible improvements for future versions include:

* [ ] Add ICC playing conditions
* [ ] Add different editions of MCC Laws
* [ ] Support multiple cricket documents
* [ ] Improve retrieval accuracy
* [ ] Add metadata filtering
* [ ] Display exact law numbers
* [ ] Display PDF page numbers
* [ ] Display retrieved context
* [ ] Add conversation/chat history
* [ ] Add streaming responses
* [ ] Add authentication
* [ ] Deploy the FastAPI backend
* [ ] Deploy the frontend
* [ ] Add evaluation/testing for RAG responses
* [ ] Add LangSmith tracing
* [ ] Improve citation/source handling

---

# 🎯 Learning Objectives

This project was developed to understand and demonstrate a complete **Retrieval-Augmented Generation workflow**:

```text
Document Loading
      ↓
Document Splitting
      ↓
Embeddings
      ↓
Vector Database
      ↓
Retriever
      ↓
Context Retrieval
      ↓
LLM
      ↓
Grounded Answer
      ↓
FastAPI
      ↓
Frontend
```

It demonstrates how RAG can be applied to a **specific domain rather than general-purpose chatbot applications**.

---

# 👨‍💻 Author

**Ahmad Mustafa**

Developed as a learning project to demonstrate the implementation of a domain-specific **RAG-based AI assistant** using Python, LangChain, ChromaDB, Gemini, FastAPI, and local HuggingFace embeddings.
