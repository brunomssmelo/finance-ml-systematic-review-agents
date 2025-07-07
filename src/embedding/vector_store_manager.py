from sentence_transformers import SentenceTransformer # Used for embeddings
import chromadb # For the VectorStore
import os

class VectorStoreManager:
    """
    Manages the VectorStore (ChromaDB) for storing and retrieving text chunks.
    """
    def __init__(self, db_path: str = "vector_store/chroma_data", collection_name: str = "articles_collection"):
        """
        Initializes the ChromaDB client and loads the embedding model.

        Args:
            db_path (str): The path to persist the ChromaDB data.
            collection_name (str): The name of the collection within ChromaDB.
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        # Ensure the directory exists
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            # For all-MiniLM-L6-v2, ChromaDB will automatically handle embedding
            # if we pass the model name here. Or we can manually embed.
            # Using default embedding function for simplicity, will rely on `SentenceTransformer` directly.
            # You might need to install 'sentence-transformers' via pip.
            # For production, consider using an explicit embedding function or pre-embedding
        )
        
        # Load the SentenceTransformer model
        # 'all-MiniLM-L6-v2' is a good balance of performance and efficiency
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"Loaded embedding model: {self.embedding_model.model_name}")

    def add_chunks(self, chunks: list[dict]):
        """
        Generates embeddings for a list of text chunks and adds them to ChromaDB.

        Args:
            chunks (list[dict]): A list of chunk dictionaries, each with 'content' and 'metadata'.
                                 Example: [{'content': '...', 'metadata': {'article_id': '...', 'page_number': ...}}]
        """
        if not chunks:
            print("No chunks to add.")
            return

        documents = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        # Generate unique IDs for each chunk. A simple approach is to use a hash or UUID.
        # For demo, let's use a combination of article_id, page_number, and an index.
        ids = [f"{m['article_id']}-{m.get('page_number', 'na')}-{i}" for i, m in enumerate(metadatas)]

        try:
            # Generate embeddings
            embeddings = self.embedding_model.encode(documents).tolist()

            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )
            print(f"Successfully added {len(chunks)} chunks to ChromaDB collection '{self.collection_name}'.")
        except Exception as e:
            print(f"Error adding chunks to ChromaDB: {e}")

    def query_vector_store(self, query_text: str, n_results: int = 5, filter_metadata: dict = None) -> list[dict]:
        """
        Performs a semantic search on the VectorStore.

        Args:
            query_text (str): The text query to search for.
            n_results (int): The number of top relevant results to return.
            filter_metadata (dict, optional): A dictionary of metadata to filter results (e.g., {'article_id': 'XYZ'}).

        Returns:
            list[dict]: A list of dictionaries containing 'document' (chunk text), 'metadata', and 'distance'.
        """
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata, # Apply metadata filtering
            include=['documents', 'metadatas', 'distances']
        )
        
        # Reformat results for easier consumption
        formatted_results = []
        if results and results['documents']:
            for i in range(len(results['documents'])):
                formatted_results.append({
                    'document': results['documents'][i],
                    'metadata': results['metadatas'][i],
                    'distance': results['distances'][i]
                })
        return formatted_results

    def reset_collection(self):
        """Deletes and recreates the collection, effectively clearing it."""
        print(f"Resetting collection '{self.collection_name}'...")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        print(f"Collection '{self.collection_name}' reset successfully.")

# Example Usage (for testing/demonstration)
if __name__ == "__main__":
    # Create a dummy chunk list
    dummy_chunks = [
        {'content': 'This article discusses a novel deep learning model for financial forecasting using LSTM networks.', 
         'metadata': {'article_id': 'CI9TCB38', 'page_number': 1, 'type': 'text_chunk'}},
        {'content': 'Figure 3: Architecture of the proposed CNN-LSTM model showing convolutional layers, pooling, and then LSTM layers.', 
         'metadata': {'article_id': 'CI9TCB38', 'page_number': 3, 'type': 'figure_caption', 'source_element': 'Figure 3'}},
        {'content': 'We evaluate the model using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) on gold price data.', 
         'metadata': {'article_id': 'CI9TCB38', 'page_number': 7, 'type': 'text_chunk'}},
        {'content': 'Future research will explore integrating social media sentiment analysis into the forecasting pipeline.', 
         'metadata': {'article_id': 'CI9TCB38', 'page_number': 10, 'type': 'text_chunk'}}
    ]

    manager = VectorStoreManager()
    
    # Optional: Reset the collection before adding to ensure a clean state for testing
    # manager.reset_collection()

    manager.add_chunks(dummy_chunks)

    # Example Query
    query = "What kind of models are used for financial prediction and how are they evaluated?"
    print(f"\nQuerying for: '{query}'")
    results = manager.query_vector_store(query, n_results=2)
    
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} (Distance: {res['distance']:.4f}) ---")
        print(f"Document: {res['document'][:100]}...")
        print(f"Metadata: {res['metadata']}")

    # Example Query with Metadata Filtering
    query_filtered = "What is the model architecture?"
    print(f"\nQuerying with filter for article CI9TCB38, type 'figure_caption': '{query_filtered}'")
    filtered_results = manager.query_vector_store(query_filtered, n_results=1, filter_metadata={'article_id': 'CI9TCB38', 'type': 'figure_caption'})
    
    for i, res in enumerate(filtered_results):
        print(f"\n--- Filtered Result {i+1} (Distance: {res['distance']:.4f}) ---")
        print(f"Document: {res['document'][:100]}...")
        print(f"Metadata: {res['metadata']}")