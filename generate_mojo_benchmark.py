import numpy as np

np.random.seed(42)

a = np.random.rand(512).astype(np.float32)
b = np.random.rand(512).astype(np.float32)

with open("benchmark_mojo.mojo", "w") as f:
    f.write("from std.math import sqrt\n")
    f.write("from time import now\n\n")
    f.write("def main():\n")

    for i in range(128):
        x = a[i * 4:i * 4 + 4]
        y = b[i * 4:i * 4 + 4]

        f.write(
            f"    var a{i} = SIMD[DType.float32, 4]"
            f"({x[0]}, {x[1]}, {x[2]}, {x[3]})\n"
        )

        f.write(
            f"    var b{i} = SIMD[DType.float32, 4]"
            f"({y[0]}, {y[1]}, {y[2]}, {y[3]})\n"
        )

    f.write("    var similarity: Float32 = 0.0\n")
    f.write("    var start = now()\n\n")

    f.write("    for _ in range(10000):\n")
    f.write("        var dot: Float32 = 0.0\n")
    f.write("        var norm_a: Float32 = 0.0\n")
    f.write("        var norm_b: Float32 = 0.0\n")

    for i in range(128):
        f.write(
            f"        dot += "
            f"a{i}[0] * b{i}[0] + "
            f"a{i}[1] * b{i}[1] + "
            f"a{i}[2] * b{i}[2] + "
            f"a{i}[3] * b{i}[3]\n"
        )

        f.write(
            f"        norm_a += "
            f"a{i}[0] * a{i}[0] + "
            f"a{i}[1] * a{i}[1] + "
            f"a{i}[2] * a{i}[2] + "
            f"a{i}[3] * a{i}[3]\n"
        )

        f.write(
            f"        norm_b += "
            f"b{i}[0] * b{i}[0] + "
            f"b{i}[1] * b{i}[1] + "
            f"b{i}[2] * b{i}[2] + "
            f"b{i}[3] * b{i}[3]\n"
        )

    f.write(
        "        similarity = dot / "
        "(sqrt(norm_a) * sqrt(norm_b))\n\n"
    )

    f.write("    var elapsed = now() - start\n")
    f.write('    print("Mojo cosine similarity:", similarity)\n')
    f.write('    print("Mojo elapsed:", elapsed)\n')
