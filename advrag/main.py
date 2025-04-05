from dotenv import load_dotenv

load_dotenv()

from advrag.graph.graph import app

if __name__ == "__main__":
    print("Hello Advanced RAG")
    print(app.invoke(input={"question": "agent memory?"}))