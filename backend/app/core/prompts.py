"""All prompt templates in one place."""

# Qwen3-Embedding is trained to receive queries with a task instruction; documents
# are embedded as-is. Omitting the instruction typically costs a few points of recall.
QUERY_INSTRUCTION = "Given a biomedical question, retrieve relevant PubMed abstracts"


def embedding_query(question: str) -> str:
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery: {question}"
