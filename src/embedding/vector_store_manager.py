from sentence_transformers import SentenceTransformer # Used for embeddings
import chromadb # For the VectorStore
import os
import logging
from typing import List, Dict

class VectorStoreManager:
    def __init__(self, db_path: str = "vector_store/chroma_data", collection_name: str = "articles_collection", embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.logger.info(f"Initialized ChromaDB collection at '{db_path}' using model '{embedding_model_name}'")

    def add_chunks(self, chunks: List[Dict]):
        if not chunks:
            self.logger.warning("No chunks to add.")
            return

        documents = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [f"{m.get('article_id', 'na')}-{m.get('page_number', 'na')}-{i}" for i, m in enumerate(metadatas)]

        try:
            embeddings = self.embedding_model.encode(documents, batch_size=32, show_progress_bar=True).tolist()
            self.collection.add(documents=documents, metadatas=metadatas, embeddings=embeddings, ids=ids)
            self.logger.info(f"Successfully added {len(chunks)} chunks to collection '{self.collection_name}'")
        except Exception as e:
            self.logger.error(f"Error adding chunks to ChromaDB: {e}")

    def query_vector_store(self, query_text: str, n_results: int = 5, filter_metadata: Dict = None) -> List[Dict]:
        query_embedding = self.embedding_model.encode(query_text).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        return [
            {
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            }
            for i in range(len(results["documents"][0]))
        ]

    def reset_collection(self):
        self.logger.info(f"Resetting collection '{self.collection_name}'...")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.logger.info(f"Collection '{self.collection_name}' reset successfully.")