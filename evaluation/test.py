import os

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")

print("API key loaded:", bool(api_key))
print("API key length:", len(api_key) if api_key else 0)

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0,
    api_key=api_key
)

response = llm.invoke("Say hello in one sentence.")

print(response.content)