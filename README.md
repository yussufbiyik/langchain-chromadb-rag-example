# langchain-chromadb-rag-example
My attempt at implementing retreival augmented generation on Ollama and other LLM services using chromadb and langchain while also providing an easy to understand, clean code for others since basically nobody else does

## Getting Started

### Requirements
- Keep in mind that this code was tested on an environment running Python 3.12
- Make sure you have Ollama installed with the model of your choice and running beforehand when you start the script.
- Change the model using config.json or cli argument if you will be using a model other than llama3.2

### Installation
#### Download the Repository
```bash
# Clone the repository
git clone https://github.com/yussufbiyik/langchain-chromadb-rag-example.git
# Navigate to the project directory
cd langchain-chromadb-rag-example
# Install dependencies
pip install -r requirements.txt
```
### Usage
#### Running Locally
You can then safely run the code.
```bash
# You can use cli arguments after the app.py if you want to
python app.py
```
#### Running on Docker
```bash
# Build the Docker image
docker build -t <your_image_name> .
# Running the image
# You can use cli arguments after the image name if you want to
docker run <your_image_name> 
```
#### CLI Arguments
You can also type help after running the script

| Argument Name| Description | Default |
| ----------- | ----------- | ----------- |
|--model|Name of the model to use|llama3.2|
|--ingestion-folder|Folder to ingest documents from|./ingest|
|--database-folder|Folder to store the database|./database|
|--system-prompt|System prompt for the ai model to use|(specified in the config.json file)|
|--ollama-address|Server address of Ollama|http://127.0.0.1:11434|


## Features
- Basically running out of the box with:
    - Ollama&ChromaDB Support
    - Support for multiple types of documents (pdf,txt,csv,docx and probably more to come)
    - Persistant Memory
- Adjustability through a config file or a set of cli arguments
- Relatively easy to understand codebase for others to learn from
- Dockerized
- Proper logging

## Future Plans
- [ ] Support for other types of documents.
- [ ] Adding other LLM services (maybe?)

## Todo:
- [X] Initial example of RAG working with Ollama and Langchain
- [X] Continuously listen for input
- [X] Continuously monitor changes in the RAG ingestion folder
- [X] Persistant memory using Chromadb
- [X] Divide everything into related files (rag_handler, chat, chroma etc.)
- [ ] Support for ChromaDB running on another address (seems to be possible)
- [X] Refactor
- [X] Dockerize
- [X] Update readme to cover installation and troubleshooting etc.
