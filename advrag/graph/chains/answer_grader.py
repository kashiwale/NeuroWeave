from langchain_core.prompts import ChatPromptTemplate
from pydantic  import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
import os

class GradeAnswer(BaseModel):

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


#llm = ChatOpenAI(temperature=0)
llm = ChatOpenAI(
    model='deepseek-chat',  # Replace with the actual model name for DeepSeek
    openai_api_base='https://api.deepseek.com',  # DeepSeek's API endpoint
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),  # Use the environment variable
    temperature=0  # Set temperature for deterministic responses
)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader