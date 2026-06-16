"""
Command-line interface to interact with the RAG pipeline.
"""
from .rag_pipeline import RAGPipeline

def main():
    """
    Initializes the RAG pipeline and answers a user's question.
    """
    # Initialize the pipeline
    # This will load all models and data, which can take a moment.
    rag_pipeline = RAGPipeline()

    # Define the question
    question = "What is Software-Defined Networking (SDN) and how does it work?"

    # Get the answer
    answer = rag_pipeline.ask(question)

    # Print the final answer
    print("\n" + "="*70)
    print("Final Answer")
    print("="*70)
    print(answer)

if __name__ == "__main__":
    main()
