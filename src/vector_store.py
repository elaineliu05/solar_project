from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any
import pandas as pd
from embeddings import EmbeddingGenerator

class SolarVectorStore:
    """Manage solar data in Qdrant vector database."""
    
    def __init__(
        self, 
        collection_name: str = "solar_locations",
        use_memory: bool = True,  # NEW: Default to in-memory
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        embedding_generator: EmbeddingGenerator = None
    ):
        """
        Initialize Qdrant vector store.
        
        Args:
            collection_name: Name of the Qdrant collection
            use_memory: If True, use in-memory mode (no Docker needed)
            qdrant_host: Qdrant host (for local Docker)
            qdrant_port: Qdrant port (for local Docker)
            qdrant_url: Qdrant Cloud URL (if using cloud)
            qdrant_api_key: Qdrant Cloud API key (if using cloud)
            embedding_generator: EmbeddingGenerator instance
        """
        self.collection_name = collection_name
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        
        # Connect to Qdrant
        if use_memory:
            print("Using Qdrant in-memory mode (no Docker required)")
            self.client = QdrantClient(":memory:")
        elif qdrant_url:
            print(f"Connecting to Qdrant Cloud: {qdrant_url}")
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            print(f"Connecting to local Qdrant Docker: {qdrant_host}:{qdrant_port}")
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        print("✅ Connected to Qdrant")
    
    def create_collection(self, overwrite: bool = False):
        """Create a new collection in Qdrant."""
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if exists:
            if overwrite:
                print(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                print(f"Collection '{self.collection_name}' already exists")
                return
        
        # Create collection
        print(f"Creating collection: {self.collection_name}")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_generator.dimension,
                distance=Distance.COSINE
            )
        )
        print("✅ Collection created")
    
    def index_documents(self, df: pd.DataFrame):
        """
        Index solar location documents into Qdrant.
        
        Args:
            df: DataFrame with columns: location_name, summary_text, latitude, 
                longitude, avg_annual_solar_radiation, total_annual_ac_production_kwh
        """
        print(f"Indexing {len(df)} documents...")
        
        # Generate embeddings for all summary texts
        texts = df['summary_text'].tolist()
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Create points for Qdrant
        points = []
        for idx, (_, row) in enumerate(df.iterrows()):
            point = PointStruct(
                id=idx,
                vector=embeddings[idx],
                payload={
                    "location_name": row['location_name'],
                    "latitude": float(row['latitude']),
                    "longitude": float(row['longitude']),
                    "summary_text": row['summary_text'],
                    "avg_annual_solar_radiation": float(row['avg_annual_solar_radiation']),
                    "total_annual_ac_production_kwh": float(row['total_annual_ac_production_kwh'])
                }
            )
            points.append(point)
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"✅ Indexed {len(points)} documents")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        # Generate embedding for query
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Search in Qdrant v1.19.1 uses query_points
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,  # Pass the embedding directly as 'query'
            limit=top_k
        )
        
        # Extract points from result
        results = search_result.points
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "score": result.score,
                "location_name": result.payload['location_name'],
                "summary_text": result.payload['summary_text'],
                "metadata": {
                    k: v for k, v in result.payload.items() 
                    if k not in ['summary_text']
                }
            })
        
        return formatted_results


# Test function
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize with in-memory mode
    embedding_gen = EmbeddingGenerator()
    vector_store = SolarVectorStore(
        embedding_generator=embedding_gen,
        use_memory=True  # This is the key change!
    )
    
    # Create collection
    vector_store.create_collection(overwrite=True)
    
    # Load data
    df = pd.read_csv("data/solar_summaries.csv")
    
    # Index documents
    vector_store.index_documents(df)
    
    # Test search
    query = "Where can I generate the most solar energy in summer?"
    results = vector_store.search(query, top_k=3)
    
    print(f"\n=== Search Results for: '{query}' ===")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['location_name']} (Score: {result['score']:.3f})")
        print(f"   {result['summary_text'][:200]}...")