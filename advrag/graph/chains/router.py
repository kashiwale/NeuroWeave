from typing import Literal
import os
from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


# Define the output model for the routing decision
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question, choose to route it to 'websearch' or 'vectorstore'."
    )


#llm = ChatOpenAI(temperature=0)
# Initialize the LLM using DeepSeek configuration
llm = ChatOpenAI(
    model='deepseek-chat',
    openai_api_base='https://api.deepseek.com',
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0
)

# Wrap LLM with structured output for routing decision
structured_llm_router = llm.with_structured_output(RouteQuery)

# Define the system prompt guiding the routing decision
system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web search."""

# Compose the prompt using LangChain's ChatPromptTemplate
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Create the routing chain
question_router = route_prompt | structured_llm_router
