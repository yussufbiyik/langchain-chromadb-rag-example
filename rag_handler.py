# Argument parsing // Argüman ayrıştırma
import argparse
import json
# For directory watcher // Dizin izleyici için
import os
import watchdog
from watchdog.observers import Observer
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
# Ollama related // Ollama ile alakalı
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
# For document loading, splitting, storing // Belge yükleme, bölme, saklama
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, csv_loader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


with open("config.json", mode="r", encoding="utf-8") as read_file:
    global config
    config = json.load(read_file)

parser = argparse.ArgumentParser(description="RAG example using langchain & Chromadb")
def parse_arguments():
    parser.add_argument("--model", type=str, default="llama3.2", help="Name of the model to use")
    parser.add_argument("--ingestion-folder", type=str, default="./ingest", help="Folder to ingest documents from")
    parser.add_argument("--database-folder", type=str, default="./database", help="Folder to store the database")
    parser.add_argument("--system-prompt", type=str, default=config["llm_options"]["system_prompt"], help="System prompt for the ai model to use")
    parser.add_argument("--ollama-address", type=str, default="http://127.0.0.1:11434", help="Ollama server address")
    return parser.parse_args()
args = parse_arguments()

def load_model():
    try:
        return ChatOllama(
                model=args.model, 
                base_url=args.ollama_address,
                temperature=config["llm_options"]["temperature"],
                num_predict=config["llm_options"]["tokens_to_generate"],
            )
    except Exception as e:
        print(f"Error loading model: {e}\n Make sure you have installed the model and ollama is running")
        exit(1)

def initialize_chroma():
    return Chroma(
        collection_name="information",
        persist_directory=args.database_folder,
        embedding_function=FastEmbedEmbeddings(),
    )
vector_store = initialize_chroma()

# Load the document based on the file extension // Dosya uzantısına göre belgeyi yükle
def load_document(file_path):
    loader = None
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = csv_loader.CSVLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        print("Unsupported file type")
        return
    if config["rag_options"]["delete_file_after_ingestion"] and os.path.exists(file_path):
        os.remove(file_path)
    return loader.load()
# Watch the ingestion folder // İçe aktarma klasörünü izle
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
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

prompt_template = PromptTemplate(
    input_variables=["context", "user_input"],
    template="""Use the following context to answer the question. 
        Context: {context}
        Question: {user_input}
    """
)
def get_response(model, user_input, useRAG=False):
    if useRAG:
        # Get the related documents // İlgili belgeleri al
        related_docs = vector_store.similarity_search_with_relevance_scores(
            query=user_input,
            k=config["rag_options"]["results_to_return"],
            score_threshold= config["rag_options"]["similarity_threshold"],
        )
        # Combine the contents of the related document parts // İlgili belge parçalarının içeriklerini birleştir
        context = ""
        for result in related_docs:
            doc = result[0]
            context += doc.page_content+"\n"
        prompt = prompt_template.format(context=context, user_input=user_input)
        return model.invoke([
            HumanMessage(prompt),
        ])
    return model.invoke([HumanMessage(user_input)])

def main():
    model = load_model()
    if not model:
        return
    if config["rag_options"]["clear_database_on_start"] and vector_store._collection.count() > 0:
        vector_store.reset_collection()
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
            else:
                response = get_response(model, user_input)
            print(response.content)
    except KeyboardInterrupt:
        observer.stop()
        print("\nGoodbye!")
    observer.join()

if __name__ == '__main__':
    main()