from langchain_core.messages import HumanMessage, SystemMessage
# Ollama related
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Load the model
selected_model = "llama3.2"

try:
    model = ChatOllama(model=selected_model)
except Exception as e:
    print(f"Error loading model: {e}\n Make sure you have installed the model, if not, run:\n ollama pull model_name")

messages = [
    SystemMessage("You are a chatbot"),
    HumanMessage("Hello, how are you?"),
]

# model.invoke(messages)

for message_chunk in model.stream(messages):
    print(message_chunk.content, end="")