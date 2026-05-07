from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def load_embedding_model(model_name=None, model_kwargs=None, encode_kwargs= None):
    if model_name is None:
        model_name = "BAAI/bge-large-en-v1.5"
    if model_kwargs is None:
        model_kwargs = {'device': 'cpu'}
    if encode_kwargs is None:
        encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction = "search_query:",
        embed_instruction = "search_document:"
    )
    return model

def batch_embed_documents(texts, batch_size=32):
    """
    Embed multiple documents in batches for better performance.
    
    Args:
        texts (List[str]): List of text documents to embed
        batch_size (int): Number of documents to process in each batch
    
    Returns:
        List[List[float]]: List of embedding vectors
    """
    model = load_embedding_model()
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.embed_documents(batch)
        embeddings.extend(batch_embeddings)
    
    return embeddings

def batch_embed_with_metadata(documents, batch_size=32):
    """
    Embed documents with their texts and metadata preserved.
    
    Args:
        documents (List[Document]): List of Document objects with page_content and metadata
        batch_size (int): Number of documents to process in each batch
    
    Returns:
        List[tuple]: List of (embedding, texts, metadata) tuples
    """
    model = load_embedding_model()
    results = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        
        # Get embeddings for this batch
        batch_embeddings = model.embed_documents(texts)
        
        # Combine embeddings with metadata
        for embedding, text, metadata in zip(batch_embeddings, texts, metadatas):
            results.append((embedding, text, metadata))
    
    return results

def embed_single_query(query):
    """
    Embed a single query for search purposes.
    
    Args:
        query (str): The search query to embed
    
    Returns:
        List[float]: Query embedding vector
    """
    model = load_embedding_model()
    return model.embed_query(query)
