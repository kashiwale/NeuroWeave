from typing import List, TypedDict

class GraphState(TypedDict):
    """
    Defines the shared state structure for LangGraph.

    Attributes:
        question (str): User input question.
        generation (str): LLM response or generation.
        web_search (bool): Whether a web search was triggered.
        documents (List[str]): Retrieved or searched documents for grounding.
    """
    question: str
    generation: str
    web_search: bool
    documents: List[str]
