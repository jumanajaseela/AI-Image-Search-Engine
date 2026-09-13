import numpy as np
import time

np.random.seed(42)

a = np.random.rand(512).astype(np.float32)
b = np.random.rand(512).astype(np.float32)

start = time.perf_counter()

for _ in range(10000):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot / (norm_a * norm_b)

end = time.perf_counter()

print("Python/NumPy cosine similarity:", similarity)
print("Python/NumPy time:", (end - start) * 1000, "ms")
