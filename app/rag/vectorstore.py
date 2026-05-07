from qdrant_client import QdrantClient, models
import os
# load API key from .env file
from dotenv import load_dotenv
load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")


def create_vectorstore():
    # setup the client
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # check if the collection already exists
    if client.collection_exists(collection_name="rag-lite"):
        print("Collection Already Exists!!!")
    else:
        client.create_collection(
            collection_name="rag-lite",
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
        )

def upsert_points(embeddings_with_metadata, collection_name="rag-lite"):
    """
    Upsert embeddings with metadata into Qdrant vector store.
    
    Args:
        embeddings_with_metadata: List of (embedding, text, metadata) tuples from batch_embed_with_metadata
        collection_name (str): Name of the Qdrant collection
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Setup the client
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        # Prepare points for upsert
        points = []
        for i, (embedding, text, metadata) in enumerate(embeddings_with_metadata):
            # Create unique ID
            point_id = i + 1
            
            # Create payload with text and metadata
            payload = {
                "text": text,
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page", 0),
                "total_pages": metadata.get("total_pages", 0),
                **metadata  # Include all original metadata
            }
            
            # Create point
            point = models.PointStruct(
                
                id=point_id,
                vector=embedding,
                payload=payload
            )
            points.append(point)
        
        # Upsert points in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch_points = points[i:i + batch_size]
            
            operation_info = client.upsert(
                collection_name=collection_name,
                points=batch_points
            )
            
            print(f"Upserted batch {i//batch_size + 1}: {len(batch_points)} points")
        
        print(f"Successfully upserted {len(points)} points to collection '{collection_name}'")
        return True
        
    except Exception as e:
        print(f"Error upserting points: {e}")
        return False

def upsert_embeddings_from_documents(documents, batch_size=32, collection_name="rag-lite"):
    """
    Complete pipeline: embed documents and upsert to vector store.
    
    Args:
        documents: List of Document objects with page_content and metadata
        batch_size (int): Batch size for embedding
        collection_name (str): Name of the Qdrant collection
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Import the embedding function
        from rag.embeddings import batch_embed_with_metadata
        
        print(f"Embedding {len(documents)} documents...")
        embeddings_with_metadata = batch_embed_with_metadata(documents, batch_size)
        
        print(f"Upserting {len(embeddings_with_metadata)} embeddings to Qdrant...")
        success = upsert_points(embeddings_with_metadata, collection_name)
        
        return success
        
    except Exception as e:
        print(f"Error in complete pipeline: {e}")
        return False

def search_similar_documents(query, collection_name="rag-lite", limit=5, score_threshold=0.7):
    """
    Search for similar documents using a query.
    
    Args:
        query (str): Search query
        collection_name (str): Name of the Qdrant collection
        limit (int): Maximum number of results to return
        score_threshold (float): Minimum similarity score
    
    Returns:
        List[dict]: List of search results with text, metadata, and scores
    """
    try:
        # Import the query embedding function
        from embeddings import embed_single_query
        
        # Setup the client
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        # Embed the query
        query_embedding = embed_single_query(query)
        
        # Search for similar documents
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold
        )
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                "text": result.payload.get("text", ""),
                "source": result.payload.get("source", "unknown"),
                "page": result.payload.get("page", 0),
                "score": result.score,
                "metadata": result.payload
            })
        
        return results
        
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []

def get_collection_info(collection_name="rag-lite"):
    """
    Get information about the collection.
    
    Args:
        collection_name (str): Name of the Qdrant collection
    
    Returns:
        dict: Collection information
    """
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        # Get collection info
        collection_info = client.get_collection(collection_name)
        
        # Get points count
        points_count = client.count(collection_name)
        
        return {
            "collection_name": collection_name,
            "vector_size": collection_info.config.params.vectors.size,
            "distance": collection_info.config.params.vectors.distance,
            "points_count": points_count.count,
            "status": collection_info.status
        }
        
    except Exception as e:
        print(f"Error getting collection info: {e}")
        return {}