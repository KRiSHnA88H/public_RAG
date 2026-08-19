import ollama 
import dataset as d
import cosinesim as c
import embedding as e


def build_index():
  if e.VECTOR_DB:
    return
  for chunk in d.dataset:
    e.add_chunk_to_database(chunk.strip())


def retrieve(query, top_n=3):
  build_index()
  query_embedding = ollama.embed(model=e.EMBEDDING_MODEL, input=query)['embeddings'][0]
  # temporary list to store (chunk, similarity) pairs
  similarities = []
  for chunk, embedding in e.VECTOR_DB:
    similarity = c.cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  # sort by similarity in descending order, because higher similarity means more relevant chunks
  similarities.sort(key=lambda x: x[1], reverse=True)
  # finally, return the top N most relevant chunks
  return similarities[:top_n]
