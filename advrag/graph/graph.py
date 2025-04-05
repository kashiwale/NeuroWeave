from dotenv import load_dotenv

from langgraph.graph import END, StateGraph

from advrag.graph.chains.answer_grader import answer_grader
from advrag.graph.chains.hallucination_grader import hallucination_grader
from advrag.graph.chains.router import question_router, RouteQuery
from advrag.graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from advrag.graph.nodes import generate, grade_documents, retrieve, web_search
from advrag.graph.state import GraphState
# Load environment variables from .env

load_dotenv()


def decide_to_generate(state):
    """
    Decision node that determines whether to proceed with generation
    or trigger a web search based on grading results.
    """
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT, INCLUDE WEB SEARCH---")
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    """
    Grades the generated answer for factual grounding and relevance.
    Returns:
        - "useful": if both hallucination and answer grading pass
        - "not useful": if grounded but irrelevant
        - "not supported": if hallucination detected
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def route_question(state: GraphState) -> str:
    """
    Routes the question to either web search or vector store based on the router.
    """
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE

# Create the LangGraph workflow
workflow = StateGraph(GraphState)

# Define the nodes

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

# Define the entry routing logic
workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)

# Define the edges and branching logic
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

# Compile the final app
app = workflow.compile()

# Optionally render the graph as a Mermaid PNG for visualization
app.get_graph().draw_mermaid_png(output_file_path="graph2.png")
