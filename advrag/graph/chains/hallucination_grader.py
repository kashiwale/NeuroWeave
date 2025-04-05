from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
import os
# llm = ChatOpenAI(temperature=0)
# Configure the LLM with DeepSeek API and deterministic response (temperature=0)
llm = ChatOpenAI(
    model='deepseek-chat',
    openai_api_base='https://api.deepseek.com',
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0
)

# Define expected output format using Pydantic
class GradeHallucinations(BaseModel):
    """
    Represents a binary evaluation of whether the generated answer contains hallucinations.
    """
    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# Wrap the LLM to enforce structured output matching the defined schema
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Define the system prompt for evaluating hallucination grounding
system = (
    "You are a grader assessing whether an LLM generation is grounded in / "
    "supported by a set of retrieved facts.\n"
    "Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded."
)

# Construct a chat prompt using the system and human messages
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

# Compose the hallucination grader chain
hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
