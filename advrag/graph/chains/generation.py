from langchain import hub
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_openai import ChatOpenAI


#llm = ChatOpenAI(temperature=0)
# Configure the LLM with DeepSeek for deterministic response generation
llm = ChatOpenAI(
    model='deepseek-chat',
    openai_api_base='https://api.deepseek.com',
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0
)

# Load a RAG (Retrieval-Augmented Generation) prompt from LangChain Hub
prompt = hub.pull("rlm/rag-prompt")

# Construct a generation chain: Prompt → LLM → String Parser
generation_chain = prompt | llm | StrOutputParser()