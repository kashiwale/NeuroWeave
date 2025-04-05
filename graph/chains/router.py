from typing import Literal
import os
from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


#llm = ChatOpenAI(temperature=0)
llm = ChatOpenAI(
    model='deepseek-chat',  # Replace with the actual model name for DeepSeek
    openai_api_base='https://api.deepseek.com',  # DeepSeek's API endpoint
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),  # Use the environment variable
    temperature=0  # Set temperature for deterministic responses
)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
