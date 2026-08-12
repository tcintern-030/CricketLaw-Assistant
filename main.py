import os

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from typer import prompt


load_dotenv()


app = FastAPI(
    title="CricketLaw Assistant API",
    description="RAG-based assistant for the MCC Laws of Cricket",
    version="1.0.0"
)


app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_db = Chroma(
    persist_directory="vector_db",
    embedding_function=embeddings
)


retriever = vector_db.as_retriever(
    search_kwargs={"k": 10}
)


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/app")
def frontend():
    return FileResponse("static/index.html")


@app.post("/ask")
def ask_question(request: QuestionRequest):

    query = request.question

    documents = retriever.invoke(query)

    context = ""

    for document in documents:
        context += document.page_content + "\n\n"

    prompt = f"""
You are CricketLaw Assistant.

Answer the user's question using ONLY the information
provided in the context below.

Do not make up cricket laws.
If the answer cannot be found in the provided context,
say that the information is not available in the provided
MCC Laws of Cricket.

Context:
-------------------------
{context}
-------------------------

Question:
{query}

Answer:
"""
    try:
        response = llm.invoke(prompt)

    except Exception as e:
        raise HTTPException(
            status_code=429,
            detail="Gemini API quota exceeded. Please try again later."
        )

    return {
        "question": query,
        "answer": response.content
    }