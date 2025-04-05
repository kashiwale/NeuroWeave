from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import os
from serpapi.utils import api_key_from_environment

# Initialize the LLM with DeepSeek API configuration
#llm = ChatOpenAI(temperature = 0)
llm = ChatOpenAI(
    model='deepseek-chat',  # Replace with the actual model name for DeepSeek
    openai_api_base='https://api.deepseek.com',  # DeepSeek's API endpoint
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),  # Use the environment variable
    temperature=0  # Set temperature for deterministic responses
)


# Define a Pydantic model to structure the output from the LLM
class GradeDocuments(BaseModel):
    """
    Binary score for relevance check on retrieved documents.
    'yes' indicates relevance, 'no' indicates irrelevance.
    """
    binary_score: str = Field(
        description="Documents are relevant to the question: 'yes' or 'no'"
    )

# Wrap the LLM to return structured output matching GradeDocuments
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Define the system prompt for relevance grading
system = """You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

# Compose the prompt using LangChain's ChatPromptTemplate
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# Create the grading chain by piping prompt into the structured LLM
retrieval_grader = grade_prompt | structured_llm_grader


