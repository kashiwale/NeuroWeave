from typing import Any, Dict

from advrag.graph.state import GraphState
from advrag.ingestion import retriever

# This function retrieves relevant documents based on the input question.
# It uses a retriever component from the ingestion module.
def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---Retrieve---")
    question = state["question"]

    # Invoke the retriever with the question to get relevant documents
    documents = retriever.invoke(question)

    # Return both the question and retrieved documents for downstream use
    return {"documents": documents , "question": question}