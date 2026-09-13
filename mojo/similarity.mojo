from std.math import sqrt


def cosine_similarity(
    a: SIMD[DType.float32, 4],
    b: SIMD[DType.float32, 4]
) -> Float32:
    var dot: Float32 = 0.0
    var norm_a: Float32 = 0.0
    var norm_b: Float32 = 0.0

    for i in range(4):
        dot += a[i] * b[i]
        norm_a += a[i] * a[i]
        norm_b += b[i] * b[i]

    return dot / (sqrt(norm_a) * sqrt(norm_b))


def main():
    # Example values representing part of an image embedding
    var a = SIMD[DType.float32, 4](0.12, -0.03, 0.45, 0.21)
    var b = SIMD[DType.float32, 4](0.11, -0.02, 0.44, 0.20)

    var similarity = cosine_similarity(a, b)

    print("Embedding similarity:", similarity)