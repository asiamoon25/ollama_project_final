from pymilvus import connections, Collection
import os

class MilvusClient:
    def __init__(self, host='localhost', port='19530', alias='default'):
        """
        initialize the milvus client with connection details
        """

        self.host = host
        self.port = port
        self.alias = alias
        self._connect()

    def _connect(self):
        """
        Connect to Milvus server using provided host and port.
        """
        try:
            connections.connect(alias=self.alias, host=self.host, port=self.port)
            print(f"Connected to Milvus server at {self.host}:{self.port} (alias: {self.alias})")
        except Exception as e:
            print(f"Failed to connect to Milvus server: {e}")
            raise

    def create_collection(self, collection_name, schema):
        """
        Create a collection in Milvus.
        """
        try:
            collection = Collection(name=collection_name, schema=schema)
            print(f"Collection '{collection_name}' created successfully.")
            return collection
        except Exception as e:
            print(f"Failed to create collection: {e}")
            raise

    def disconnect(self):
        """
        Disconnect the current Milvus connection.
        """
        try:
            connections.disconnect(alias=self.alias)
            print(f"Disconnected from Milvus server (alias: {self.alias})")
        except Exception as e:
            print(f"Failed to disconnect: {e}")
            raise
