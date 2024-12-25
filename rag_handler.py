from langchain_core.messages import HumanMessage, SystemMessage
# Ollama related // Ollama ile alakalı
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
# Document loaders // Doküman yükleyiciler
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, csv_loader, TextLoader
# For directory watcher // Dizin izleyici için
import watchdog
from watchdog.observers import Observer


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
# Watch the ingestion folder // İçe aktarma klasörünü izle
class DocumentHandler(watchdog.events.FileSystemEventHandler):
    def on_created(self, event):
        print(f"\nLOG:New file detected: {event.src_path}")
    def on_modified(self, event):
        print("\nLOG:File modified")
    def on_deleted(self, event):
        print("\nLOG:File deleted")
observer = Observer()
observer.schedule(DocumentHandler(), path=args.ingestion_folder, recursive=True)
observer.start()

def get_response(user_input):
    return model.invoke([HumanMessage(user_input)])

if __name__ == '__main__':
    print("Welcome, type 'help' for help and 'exit' to exit.")
    try:
        while True:
            user_input = input(">> ")
            if user_input == "exit":
                observer.stop()
                print("\nGoodbye!")
                break
            if user_input == "help":
                print(parser.format_help())
                continue
            response = get_response(user_input)
            print(response.content)
    except KeyboardInterrupt:
        observer.stop()
        print("\nGoodbye!")
    observer.join()