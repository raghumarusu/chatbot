import qdrant_client
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, Match

# Initialize Qdrant client
client = qdrant_client.QdrantClient(host="localhost", port=6333)

# Function to store vectors in Qdrant
def store_vectors(document_chunks, vectors):
    points = [PointStruct(id=i, vector=vector, payload={"chunk": chunk}) for i, (chunk, vector) in enumerate(zip(document_chunks, vectors))]
    client.upsert(collection_name="chatbot", points=points)

# Function to search in Qdrant
def search_vectors(query_vector):
    search_result = client.search(
        collection_name="chatbot",
        query_vector=query_vector,
        limit=5  # Return top 5 results
    )
    return [hit.payload["chunk"] for hit in search_result]
