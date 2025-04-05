from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import os
from serpapi.utils import api_key_from_environment

#llm = ChatOpenAI(temperature = 0)
llm = ChatOpenAI(
    model='deepseek-chat',  # Replace with the actual model name for DeepSeek
    openai_api_base='https://api.deepseek.com',  # DeepSeek's API endpoint
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),  # Use the environment variable
    temperature=0  # Set temperature for deterministic responses
)

class GradeDocuments(BaseModel):
    """
    Binary score for relevance check on retrieved documents
    """

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes', 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader


