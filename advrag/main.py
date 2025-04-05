"""
main.py

Entry point for the NeuroWeave agent. This script initializes and runs the LangGraph-based application workflow.
"""

from advrag.graph.graph import app  # Import the workflow application

# This script simply runs the application when the module is executed directly.
if __name__ == "__main__":
    app.invoke()  # Invoke the LangGraph application