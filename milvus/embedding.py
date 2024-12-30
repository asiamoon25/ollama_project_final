from langchain.embeddings import OllamaEmbeddings
from pymilvus import Collection, utility
import numpy as np

class EmbeddingManager:
    def __init__(self, model_name="tazarov/all-minilm-l6-v2-f32:latest", milvus_collection_name="LangChainCollection"):
        """
        Initialize the embedding manager with a specified Ollama model and Milvus collection.
        """
        self.model_name = model_name
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.collection_name = milvus_collection_name

        # Ensure Milvus collection exists
        if not utility.has_collection(self.collection_name):
            raise ValueError(f"Collection '{self.collection_name}' does not exist in Milvus.")
        self.collection = Collection(name=self.collection_name)

    def generate_embedding(self, text):
        """
        Generate embedding for a single text input.

        Args:
             text (str): Input text.

        Returns:
            np.ndarray: Generated embedding.
        """
        try:
            return np.array(self.embeddings.embed_query(text))
        except Exception as e:
            print(f"Failed to generate embedding for text '{text}': {e}")
            raise

    def insert_embedding(self,texts, ids=None):
        """
        Insert embeddings for a list of texts into Milvus.

        Args:
             texts (list of str): List of input texts.
             ids (list of int, optional): List of IDs for the embeddings.

        Returns:
            list: Inserted primary keys in Milvus.
        """
        try:
            embeddings = [self.generate_embedding(text) for text in texts]
            if ids is None:
                ids = list(range(len(texts)))
            insert_result = self.collection.insert([ids, embeddings])
            print(f"Inserted {len(texts)} records into collection '{self.collection_name}'")
            return insert_result.primary_keys
        except Exception as e:
            print(f"FAiled to insert embeddings: {e}")
            raise

    def search_embedding(self, query_text, top_k=10):
        """
        Search for similar embeddings in Milvus for a query text.

        Args:
            query_text (str): Query text.
            top_k (int): Number of top results to retrieve.

        Return:
             list: Search results from Milvus.
        """
        try:
            query_embedding = self.generate_embedding(query_text)
            search_params = {"metri_type": "L2", "params": {"nprobe":10}}
            results = self.collection.search(
                data = [query_embedding],
                anns_field="vector",
                param=search_params,
                limit=top_k,
            )
            print(f"Found {len(results[0])} similar results for query '{query_text}'.")
            return results[0]
        except Exception as e:
            print(f"Search failed for query '{query_text}': {e}")
            raise
