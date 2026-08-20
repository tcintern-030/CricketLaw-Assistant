import os

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()


def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma(
        persist_directory="vector_db",
        embedding_function=embeddings
    )

    return vector_db

vector_db = load_vector_db()

def create_retriever(vector_db):
    retriever = vector_db.as_retriever(
        search_kwargs={"k": 7}
    )

    return retriever

retriever = create_retriever(vector_db)

def retrieve_documents(query, retriever):
    documents = retriever.invoke(query)

    return documents

def create_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    return llm

llm = create_llm()

def generate_response(query, documents, llm):

    context = ""

    for document in documents:
        context += document.page_content + "\n\n"

    prompt = f"""
You are CricketLaw AI.

Answer the user's question using ONLY the information
provided in the context.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content


def run_rag(query):
    documents = retrieve_documents(query, retriever)
    response = generate_response(query, documents, llm)

    return {
        "answer": response,
        "documents": documents
    }

if __name__ == "__main__":
    query = input("Enter your query about Laws of cricket: ")

    result = run_rag(query)

    print("\n\nGenerated Response:\n")
    print(result["answer"])