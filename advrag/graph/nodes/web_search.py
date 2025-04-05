from typing import Any, Dict

from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from advrag.graph.state import GraphState
from dotenv import load_dotenv

# Load environment variables from a .env file for secure API key access
load_dotenv()

# Instantiate the Tavily search tool with a limit on number of results
web_search_tool = TavilySearchResults(max_results=3)


# This function performs a web search based on the input question and appends results to existing documents
def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Perform web search using Tavily tool
    tavily_results = web_search_tool.invoke({"query": question})

    # Concatenate all results into a single document
    joined_tavily_results = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_results)

    # Append the web results to the existing documents or start a new list
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"documents": documents, "question": question}


# Optional: Run standalone web search for testing/debugging
if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})
