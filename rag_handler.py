from langchain_core.messages import HumanMessage, SystemMessage
# Ollama related // Ollama ile alakalı
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
# For document loading, splitting, storing // Belge yükleme, bölme, saklama
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, csv_loader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# For directory watcher // Dizin izleyici için
import watchdog
from watchdog.observers import Observer


# Argument parsing // Argüman ayrıştırma
import argparse
parser = argparse.ArgumentParser(description="RAG example using langchain & Chromadb")
parser.add_argument("--model", type=str, default="llama3.2", help="Name of the model to use")
parser.add_argument("--ingestion-folder", type=str, default="./ingest", help="Folder to ingest documents from")
parser.add_argument("--database-folder", type=str, default="./database", help="Folder to store the database")
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
    return model.invoke([HumanMessage(user_input)])

# Initialize the Chroma database // Chroma veritabanını başlat
vector_store = Chroma(
        collection_name="information",
        persist_directory=args.database_folder,
        embedding_function=FastEmbedEmbeddings(),
    )

# Text splitter // Metin bölücü
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

# Watch the ingestion folder // İçe aktarma klasörünü izle
class FileSystemWatcher(watchdog.events.FileSystemEventHandler):
    def on_created(self, event):
        # Load the document based on the file extension // Dosya uzantısına göre belgeyi yükle
        docs = None
        if event.src_path.endswith(".pdf"):
            loader = PyPDFLoader(event.src_path)
            docs = loader.load();
        elif event.src_path.endswith(".docx"):
            loader = Docx2txtLoader(event.src_path)
            docs = loader.load();
        elif event.src_path.endswith(".csv"):
            loader = csv_loader.CSVLoader(event.src_path)
            docs = loader.load();
        elif event.src_path.endswith(".txt"):
            loader = TextLoader(event.src_path)
            docs = loader.load();
        else:
            print("Unsupported file type")
            return
        chunks = text_splitter.split_documents(docs)
        vector_store.add_documents(documents=chunks)
    def on_deleted(self, event):
        print("\nLOG:File deleted")
        
observer = Observer()
observer.schedule(FileSystemWatcher(), path=args.ingestion_folder, recursive=True)
observer.start()

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