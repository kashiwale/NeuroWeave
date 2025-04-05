# Define constants used as node/step labels in the LangGraph workflow

RETRIEVE = "retrieve"              # Vectorstore retrieval node
GRADE_DOCUMENTS = "grade_documents"  # Grade the relevance of retrieved documents
GENERATE = "generate"              # Generate an LLM-based answer
WEBSEARCH = "web_search"           # Web search fallback node