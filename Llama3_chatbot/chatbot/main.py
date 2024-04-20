from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import ollama
import streamlit as st
import os
from dotenv import load_dotenv


load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]= "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. PLease response to use quaries"),
        ("user", "Question: {question}")
    ]
)



#streamlit

st.title("Langchain Demo with LLAMA_3")
input_text = st.text_input("Search the topic u want")

#Ollama LLM
llm = ollama.Ollama(model="llama3")
output_parser  = StrOutputParser()
chain = prompt|llm|output_parser


if input_text:
    st.write(chain.invoke({'question':input_text}))

