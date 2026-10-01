# embedding generation using sentence transformers
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np

class EmbeddingGenerator: 
	def __init__(self, model_name: str = "all-MiniLM-L6-v2", provider: str = "sentence-transformers"):
		self.provider = provider
		self.model_name = model_name
		print(f"loading sentence transformer model: {model_name}")
		self.model = SentenceTransformer(model_name)
		self.dimension = self.model.get_sentence_embedding_dimension()
		print(f"model loaded. embedding dimension: {self.dimension}")

	def generate_embedding(self, text: str) -> List[float]:
		# generate embedding for a single text
		embedding = self.model.encode(text, convert_to_numpy = True)
		return embedding.tolist()

	def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
		# generate embeddings for multiple texts (batch)
		embeddings = self.model.encode(texts, convert_to_numpy=True)
		return embeddings.tolist()

if __name__ == "__main__":
	generator = EmbeddingGenerator(model_name = "all-MiniLM-L6-v2", provider = "sentence-transformers")
	test_texts = ["Solar panels generate electricity from sunlight", 
			"Photovoltaic systems convert solar energy", 
			"I like pizza"
]
	embeddings = generator.generate_embeddings(test_texts)
	print(f"generated {len(embeddings)} embeddings")
	print(f"embedding dimension: {len(embeddings[0])}")

	# calculate similarity scores
	from numpy import dot
	from numpy.linalg import norm
	
	def cosine_similarity(a, b):
		sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
		sim_2_3 = cosine_similarity(embeddings[1], embeddings[2])
		print(f"similarity between 1 and 2: {sim_1_2:.3f}")
		print(f"similarity between 2 and 3: {sim_2_3:.3f}")

