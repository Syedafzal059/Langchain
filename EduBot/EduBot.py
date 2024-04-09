from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain_community.llms.ctransformers import CTransformers
from langchain.chains.retrieval_qa.base import RetrievalQA
from config import *


class EduBotCreator:
    def __init__(self):
        self.prompt_temp = PROMPT_TEMPLATE
        self.input_variables = INP_VARS
        #self.input_variables = ['context', 'question']
        self.embedder = EMBEDDER
        self.vector_db_path = VECTOR_DB_PATH
        self.model_ckpt = MODEL_CKPT
        self.max_new_tokens = MAX_NEW_TOKENS
        self.temperature = TEMPERATURE
        self.model_type = MODEL_TYPE
        self.search_kwargs = SEARCH_KWARGS
        self.chain_type = CHAIN_TYPE
    #     self.prompt_temp ='''
    # With the information provided try to answer the question.
    # If you cant answer the question based on the information either say you cant find the answer or unable to find.
    # This is related to the cricket domain. so try to understand in depth about the context and answer only based on the data provided
    # Context:{context}
    # Question:{question}
    # Do provide only correct answers

    # Correct answer:
    # '''




    def create_custom_prompt(self):
        coustom_prompt_temp = PromptTemplate(template=self.prompt_temp,
                                        input_variables=['context', 'question'])
        return coustom_prompt_temp
    
    def load_llm(self):
        llm = CTransformers(
            model=self.model_ckpt,
            model_type = self.model_type,
            max_new_tokens = self.max_new_tokens,
            temperatture = self.temperature
        )
        return llm


    def load_vectordb(self):
        hfembeddings = HuggingFaceHubEmbeddings(
            model=self.embedder,
            model_kwargs={'device': 'cpu'}
        )
        vector_db= FAISS.load_local(self.vector_db_path, hfembeddings, allow_dangerous_deserialization=True)
        return vector_db
    

    def create_bot(self, custom_prompt, vectordb, llm):
        print("__________________I will make the bot_______________________")
        retrieval_qa_chain = RetrievalQA.from_chain_type(
            llm= llm,
            chain_type= self.chain_type,
            retriever = vectordb.as_retriever(search_kwargs = self.search_kwargs),
            return_source_documents =True,
            chain_type_kwargs={"prompt":custom_prompt}
        )
        return retrieval_qa_chain


    def create_edubot(self):
        self.custom_prompt = self.create_custom_prompt()
        self.vector_db = self.load_vectordb()
        self.llm = self.load_llm()
        self.bot = self.create_bot(self.custom_prompt, self.vector_db, self.llm)
        return self.bot


    







     