import ollama
import dataset as d
import cosinesim as c
import embedding as e
import heapq


def build_index():
    if e.VECTOR_DB:
        return

    for chunk in d.dataset:
        e.add_chunk_to_database(chunk.strip())


def retrieve(query, top_n=3):
    build_index()

    # Convert the user query into an embedding
    query_embedding = ollama.embed(
        model=e.EMBEDDING_MODEL,
        input=query
    )["embeddings"][0]

    # Min-heap containing only the top N results
    heap = []

    # Calculate similarity for every chunk
    for chunk, embedding in e.VECTOR_DB:

        similarity = c.cosine_similarity(
            query_embedding,
            embedding
        )

        # Add the first top_n results
        if len(heap) < top_n:
            heapq.heappush(
                heap,
                (similarity, chunk)
            )

        # If the new result is better than the
        # smallest result currently in the heap
        elif similarity > heap[0][0]:

            heapq.heapreplace(
                heap,
                (similarity, chunk)
            )

    # Heap contains (similarity, chunk)
    # Convert it back to (chunk, similarity)
    # because the rest of your RAG code expects that format.
    results = [
        (chunk, similarity)
        for similarity, chunk in heap
    ]

    # Highest similarity first
    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results