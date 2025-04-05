# This module defines a node for generating answers based on query and documents
# using a language model chain. It is used as part of the graph workflow.

from typing import Any, Dict

from advrag.graph.chains.generation import generation_chain
from advrag.graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
