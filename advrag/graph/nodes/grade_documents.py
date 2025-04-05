# grade_documents.py - Handles the document grading logic within the workflow.
# This module likely evaluates retrieved documents against the original query
# or expected answers using some form of comparison logic, possibly leveraging
# LLMs or traditional scoring metrics.


from typing import Any, Dict

from advrag.graph.chains.retrieval_grader import retrieval_grader
from advrag.graph.state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determine whether the  retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state

    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue
    return {"documents": filtered_docs, "question": question , "web_search": web_search}
