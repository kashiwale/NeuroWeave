from langchain import hub
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_openai import ChatOpenAI


#llm = ChatOpenAI(temperature=0)
llm = ChatOpenAI(
    model='deepseek-chat',  # Replace with the actual model name for DeepSeek
    openai_api_base='https://api.deepseek.com',  # DeepSeek's API endpoint
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),  # Use the environment variable
    temperature=0  # Set temperature for deterministic responses
)

prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()
