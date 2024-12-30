from pymilvus import Collection, utility
from pymilvus.exceptions import MilvusException


def create_collection(collection_name, schema, **kwargs):
    """
    Create a new collection with the specified schema.
    """
    try:
        collection = Collection(name=collection_name, schema=schema, **kwargs)
        print(f"Collection '{collection_name}' created successfully.")
        return collection
    except MilvusException as e:
        print(f"Failed to create collection: {e}")
        raise


def delete_collection(collection_name):
    """
    Delete a collection by its name.
    """
    try:
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"Collection '{collection_name}' deleted successfully.")
        else:
            print(f"Collection '{collection_name}' does not exist.")
    except MilvusException as e:
        print(f"Failed to delete collection: {e}")
        raise


def search_data(collection_name, vectors, search_params, limit=10):
    """
    Search for the closest vectors in a collection.
    """
    try:
        collection = Collection(name=collection_name)
        results = collection.search(
            data=vectors,
            anns_field="vector",
            param=search_params,
            limit=limit
        )
        print(f"Search completed. Found {len(results)} results.")
        return results
    except MilvusException as e:
        print(f"Search failed: {e}")
        raise


def list_collections():
    """
    List all collections in Milvus.
    """
    try:
        collections = utility.list_collections()
        print(f"Available collections: {collections}")
        return collections
    except MilvusException as e:
        print(f"Failed to list collections: {e}")
        raise


def get_collection_info(collection_name):
    """
    Check if the Milvus server is alive.
    """
    try:
        is_alive = utility.get_server_status()
        print(f"Milvus server status: {is_alive}")
        return is_alive
    except MilvusException as e:
        print(f"Failed to check server status: {e}")
        raise
