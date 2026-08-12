import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader, UnstructuredFileLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv

load_dotenv()


loader = PyPDFLoader("laws/Laws Of cricket.pdf")
documents = loader.load()

print(documents)

"""
dir_loader = DirectoryLoader(
    "sample_directory", 
    glob="*.txt",
    loader_cls=TextLoader
)
directory = dir_loader.load()
"""

"""
print("\n\nDocuments from directory:")

for i, doc in enumerate(directory):
    print(f"Document {i+1}:")
    print(doc)
    print("\n")
"""

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

chunks = split_documents(documents)

"""
print("\n\nChunks:")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(chunk)
    print("\n")
"""

def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

embeddings = create_embeddings()

"""
print("\nGenerating Embeddings...\n")

for i, chunk in enumerate(chunks):
    text = chunk.page_content
    embedding = embeddings.embed_query(text)

    print(f"Chunk {i+1}")
    print(f"Embedding Dimension: {len(embedding)}")
    print(f"First 5 elements: {embedding[:5]}")
    print("\n")

"""

def create_vector_db(chunks, embeddings):
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vector_db"
    )

    return vector_db

vector_db = create_vector_db(chunks, embeddings)


"""
results = vector_db.get()

print(results.keys())
print(results["documents"])
print(results["metadatas"])
"""
