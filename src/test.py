# test_qdrant_api.py
from qdrant_client import QdrantClient

client = QdrantClient(":memory:")

# Check what methods are available
print("Available methods:")
print([method for method in dir(client) if not method.startswith('_')])

