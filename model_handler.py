# Ollama related // Ollama ile alakalı
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama

class ModelHandler:
    def __init__(self, cli_args, config):
        self.cli_args = cli_args
        self.config = config
        
        self.model = self.load_model()
        self.model.invoke([SystemMessage(cli_args.system_prompt)])
        
        self.prompt_template = PromptTemplate(
            input_variables=["context", "user_input"],
            template="""Use the following context to answer the question. 
                Context: {context}
                Question: {user_input}
            """
        )
    
    def load_model(self):
        try:
            return ChatOllama(
                    model=self.cli_args.model, 
                    base_url=self.cli_args.ollama_address,
                    temperature=self.config["llm_options"]["temperature"],
                    num_predict=self.config["llm_options"]["tokens_to_generate"],
                )
        except Exception as e:
            print(f"Error loading model: {e}\n Make sure you have installed the model and ollama is running")
            exit(1)

    def combine_context(self, related_docs):
        context = ""
        for result in related_docs:
            doc = result[0]
            context += doc.page_content+"\n"
        return context

    def get_response(self, user_input, related_docs, useRAG=False):
        if useRAG:
            # Combine the contents of the related document parts // İlgili belge parçalarının içeriklerini birleştir
            context = self.combine_context(related_docs)
            prompt = self.prompt_template.format(context=context, user_input=user_input)
            return self.model.invoke([HumanMessage(prompt)])
        return self.model.invoke([HumanMessage(user_input)])