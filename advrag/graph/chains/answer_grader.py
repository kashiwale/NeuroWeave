from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
import os

# Define a schema for grading an answer with a binary (yes/no) score
class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

#llm = ChatOpenAI(temperature=0)
# Configure the LLM with DeepSeek model and deterministic behavior
llm = ChatOpenAI(
    model='deepseek-chat',
    openai_api_base='https://api.deepseek.com',
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0
)

# Bind the LLM with the structured output schema
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Define system message that guides the LLM's grading logic
system = """You are a grader assessing whether an answer addresses / resolves a question. 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""

# Construct a prompt using both the system message and dynamic user inputs
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

# Define a runnable pipeline: Prompt → LLM → Structured Output Parser
answer_grader: RunnableSequence = answer_prompt | structured_llm_grader