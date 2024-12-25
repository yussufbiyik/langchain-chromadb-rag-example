from langchain_core.messages import HumanMessage, SystemMessage
# Ollama related // Ollama ile alakalı
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Argument parsing // Argüman ayrıştırma
import argparse
parser = argparse.ArgumentParser(description="RAG example using langchain & Chromadb")
parser.add_argument("--model", type=str, default="llama3.2", help="Name of the model to use")
parser.add_argument("--ingestion-folder", type=str, default="./ingest", help="Folder to ingest documents from")
parser.add_argument("--ollama-address", type=str, default="http://127.0.0.1:11434", help="Ollama server address")
args = parser.parse_args()

# Load the model // Modeli yükle
try:
    model = ChatOllama(model=args.model, base_url=args.ollama_address)
except Exception as e:
    print(f"Error loading model: {e}\n Make sure you have installed the model and ollama is running")

# Initial message // İlk mesaj
model.invoke([SystemMessage("You are a helpful assistant that can answer questions about the given documents, only answer questions that are related to the documents if there are any documents.")])

def get_response(user_input):
    messages = [HumanMessage(user_input)]
    return model.invoke(messages)

if __name__ == '__main__':
    print("Welcome, type 'help' for help and 'exit' to exit.")
    while True:
        user_input = input(">> ")
        if user_input == "exit":
            print("Goodbye!")
            break
        if user_input == "help":
            print(parser.format_help())
            continue
        response = get_response(user_input)
        print(response.content)