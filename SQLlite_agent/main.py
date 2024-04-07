from langchain import chat_models

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
)

from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from dotenv import load_dotenv

from tools.sql import run_query_tool, list_tables, describe_tables_tool




chat = chat_models
tables = list_tables()
prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content=(
                      "You are an AI that has access to a SQLite database.\n"
                      f"The database has tables of : {tables}\n"
                      "Do not make any assumption about what tables exist"
                      "or what column exist. Insted, use the 'describe_tables' function"
        )),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
        
    ]
)


tools = [run_query_tool, describe_tables_tool]
agent = OpenAIFunctionsAgent(
    llm=chat,
    prompt=prompt,
    tools=tools
)


agent_exucator = AgentExecutor(
    agent=agent,
    verbose=True,
    tools= tools
)


agent_exucator("How many users are in the database?")