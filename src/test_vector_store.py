# test to verify embeddings and vector store work
import pandas as pd
from embeddings import EmbeddingGenerator
from vector_store import SolarVectorStore
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    print("testing Solar RAG vector store")
    # initialize embedding generator
    print("1. initializing embedding generator")
    embedding_gen = EmbeddingGenerator(model_name = "all-MiniLM-L6-v2", provider = 'sentence-transformers')
    #initialize vector store
    print("2. initializating vector store")
    vector_store = SolarVectorStore(
        embedding_generator = embedding_gen, 
        qdrant_host = os.getenv("QDRANT_HOST", "localhost"), 
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    )

    print("3. creating collection")
    vector_store.create_collection(overwrite=True)
    print("4. loading solar data")
    df = pd.read_csv("data/solar_summaries.csv")
    print(f"loaded {len(df)} documents")

    print("5. indexing documents")
    vector_store.index_documents(df)

    print("6. testing searches")
    test_queries = [
        "Where can I generate the most solar energy?", 
        "Whats the best location for solar panels in the winter?", 
        "How much electricity can I produce in Charlottesville?"
        "Which city has the highest solar potential?"
    ]
    for query in test_queries:
        print(f"\n query: {query}")
        results = vector_store.search(query, top_k=3)

        for i, result in enumerate(results, 1):
            print(f"{i}. {result['location_name']} (Score: {result['score']:.3f})")
            print(f" Annual Production: {result['metadata']['total_annual_ac_production_kwh']:.0f} kWh")
    print('all tests passed!')

if __name__ == "__main__":
    main()