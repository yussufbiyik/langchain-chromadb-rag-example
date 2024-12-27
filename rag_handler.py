# Argument parsing // Argüman ayrıştırma
import argparse
# For directory watcher // Dizin izleyici için
import watchdog
from watchdog.observers import Observer
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
# Ollama related // Ollama ile alakalı
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
# For document loading, splitting, storing // Belge yükleme, bölme, saklama
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, csv_loader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

parser = argparse.ArgumentParser(description="RAG example using langchain & Chromadb")
def parse_arguments():
    parser.add_argument("--model", type=str, default="llama3.2", help="Name of the model to use")
    parser.add_argument("--ingestion-folder", type=str, default="./ingest", help="Folder to ingest documents from")
    parser.add_argument("--database-folder", type=str, default="./database", help="Folder to store the database")
    parser.add_argument("--system-prompt", type=str, default="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise and use short sentences unless told to do otherwise.", help="System prompt for the ai model to use")
    parser.add_argument("--ollama-address", type=str, default="http://127.0.0.1:11434", help="Ollama server address")
    return parser.parse_args()
args = parse_arguments()

def load_model():
    try:
        return ChatOllama(model=args.model, base_url=args.ollama_address)
    except Exception as e:
        print(f"Error loading model: {e}\n Make sure you have installed the model and ollama is running")
        exit(1)

template = args.system_prompt+"\n\nContext: {context}\n\nQuestion: {user_input}"
def get_response(model, user_input, useRAG=False):
    if useRAG:
        context = vector_store.similarity_search(user_input)
        prompt_template = PromptTemplate(
            input_variables=["context", "user_input"],
            template=template
        )
        prompt = prompt_template.format(context=context, user_input=user_input)
        return model.invoke([HumanMessage(prompt)])
    return model.invoke([HumanMessage(user_input)])


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
def initialize_chroma():
    return Chroma(
        collection_name="information",
        persist_directory=args.database_folder,
        embedding_function=FastEmbedEmbeddings(),
    )
vector_store = initialize_chroma()

# Load the document based on the file extension // Dosya uzantısına göre belgeyi yükle
def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        return loader.load();
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
        return loader.load();
    elif file_path.endswith(".csv"):
        loader = csv_loader.CSVLoader(file_path)
        return loader.load();
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
        return loader.load();
    else:
        print("Unsupported file type")
        return
# Watch the ingestion folder // İçe aktarma klasörünü izle
class FileSystemWatcher(watchdog.events.FileSystemEventHandler):
    def on_created(self, event):
        docs = load_document(event.src_path)
        # Split the text into chunks // Metni parçalara böl 
        chunks = text_splitter.split_documents(docs)
        # Add the documents to the database // Belgeyi veritabanına ekle
        vector_store.add_documents(documents=chunks)
    def on_deleted(self, event):
        print("\nLOG:File deleted")
# Start the folder observer // Klasör gözlemcisini başlat
observer = Observer()
observer.schedule(FileSystemWatcher(), path=args.ingestion_folder, recursive=True)
observer.start()

def main():
    model = load_model()
    if not model:
        return
    # System prompt // Sistem promptu
    model.invoke([SystemMessage(args.system_prompt)])
    
    # Main loop // Ana döngü
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
            # Use RAG if chromadb exists // Chromadb varsa RAG kullan
            # Otherwise, just use the model // Aksi takdirde sadece modeli kullan
            if vector_store._collection.count() > 0:
                response = get_response(model, user_input, True)
            response = get_response(model, user_input)
            print(response.content)
    except KeyboardInterrupt:
        observer.stop()
        print("\nGoodbye!")
    observer.join()

if __name__ == '__main__':
    main()