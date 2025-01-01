# For document loading, splitting, storing // Belge yükleme, bölme, saklama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, csv_loader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

class RAGHandler:
    def __init__(self, cli_args, config):
        self.cli_args = cli_args
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.vector_store = self.initialize_chroma()

    def initialize_chroma(self):
        return Chroma(
            collection_name="information",
            persist_directory=self.cli_args.database_folder,
            embedding_function=FastEmbedEmbeddings(),
        )

    # Load the document based on the file extension // Dosya uzantısına göre belgeyi yükle
    def load_document(self, file_path):
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
            return None
        return loader.load()

    def add_document_to_chroma(self, document):
        if document is None:
            print(f"Failed to load document.")
            return
        chunks = self.text_splitter.split_documents(document)
        self.vector_store.add_documents(chunks)
        print(f"Added document to the database.")

    def get_docs_by_similarity(self, query):
        return self.vector_store.similarity_search_with_relevance_scores(
                query=query,
                k=self.config["rag_options"]["results_to_return"],
                score_threshold= self.config["rag_options"]["similarity_threshold"],
            )