import argparse
import json
import logging
# For directory watcher // Dizin izleyici için
import os
import watchdog
from watchdog.observers import Observer

import rag_handler as rh
import model_handler as mh


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

with open("config.json", mode="r", encoding="utf-8") as read_file:
    global config
    config = json.load(read_file)
    read_file.close()

parser = argparse.ArgumentParser(description="RAG example using langchain & Chromadb")
def parse_arguments():
    parser.add_argument("--model", type=str, default="llama3.2", help="Name of the model to use")
    parser.add_argument("--ingestion-folder", type=str, default="./ingest", help="Folder to ingest documents from")
    parser.add_argument("--database-folder", type=str, default="./database", help="Folder to store the database")
    parser.add_argument("--system-prompt", type=str, default=config["llm_options"]["system_prompt"], help="System prompt for the ai model to use")
    parser.add_argument("--ollama-address", type=str, default="http://127.0.0.1:11434", help="Ollama server address")
    return parser.parse_args()
args = parse_arguments()

# Watch the ingestion folder // İçe aktarma klasörünü izle
class FileSystemWatcher(watchdog.events.FileSystemEventHandler):
    def on_created(self, event):
        logging.info(f"Detected and Ingesting: {event.src_path}")
        docs = rag_handler.load_document(event.src_path)
        # Delete the file after ingestion if thew option is true in config // Config'de belirlendiyse içe aktarmadan sonra dosyayı sil
        if config["rag_options"]["delete_file_after_ingestion"] and os.path.exists(event.src_path):
            os.remove(event.src_path)
            logging.info(f"Deleted After Ingestion: {event.src_path}")
        # Add the documents to the database // Belgeyi veritabanına ekle
        rag_handler.add_document_to_chroma(docs)
        logging.info(f"Ingested: {event.src_path}")
    def on_deleted(self, event):
        logging.info(f"Deleted: {event.src_path}")
# Start the folder observer // Klasör gözlemcisini başlat
observer = Observer()
observer.schedule(FileSystemWatcher(), path=args.ingestion_folder, recursive=True)
observer.start()

model_handler = mh.ModelHandler(args, config)
rag_handler = rh.RAGHandler(args, config)

model = model_handler.load_model()
if not model:
    logging.error("Error loading model. Make sure you have installed the model and Ollama is running. Exiting...")
    exit(1)
if config["rag_options"]["clear_database_on_start"] and rag_handler.vector_store._collection.count() > 0:
    rag_handler.vector_store.reset_collection()

def main():
    # Main loop // Ana döngü
    print("Welcome, type 'help' for help and 'exit' to exit.")
    try:
        while True:
            user_input = input(">> ")
            if user_input == "exit":
                observer.stop()
                print("Goodbye!")
                logging.info("Exiting...")
                break
            if user_input == "help":
                print(parser.format_help())
                continue
            # Use RAG if chromadb exists // Chromadb varsa RAG kullan
            # Otherwise, just use the model // Aksi takdirde sadece modeli kullan
            if rag_handler.vector_store._collection.count() > 0:
                related_docs = rag_handler.get_docs_by_similarity(user_input)
                response = model_handler.get_response(user_input, related_docs, True)
            else:
                response = model_handler.get_response(user_input, None, False)
            print(f"Response: {response.content}")
    except KeyboardInterrupt:
        observer.stop()
        logging.info("Exiting...")
    observer.join()

if __name__ == '__main__':
    main()