import numpy as np
import itertools

# Load all embeddings and flatten
embeddings = [
    np.load(f'./test_images/{i}.npy').flatten()
    for i in range(1, 8)
]

# Normalize each embedding
embeddings = [e / np.linalg.norm(e) for e in embeddings]

# Compute cosine similarity between all pairs
n = len(embeddings)
print("Cosine Similarity Matrix:\n")

for i in range(n):
    for j in range(n):
        cosine_sim = np.dot(embeddings[i], embeddings[j])
        print(f"Cosine({i+1}.npy, {j+1}.npy): {cosine_sim:.4f}")
    print()
