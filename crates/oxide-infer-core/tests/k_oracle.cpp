// Byte-exact oracle for ggml K-type quantization, using llama.cpp's own code.
// Usage: k_oracle <type> <in_f32.bin> <packed.bin> <dequant_f32.bin> [--only-packed]
//   type: q1_0|q2_0|q4_0|q4_1|q5_0|q5_1|q8_0|q8_1|q2_k|q3_k|q4_k|q5_k|q6_k|q8_k
// If --only-packed given, dequant output is skipped; else both are written.
#include "ggml.h"
#include "ggml-quants.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

namespace {
void fake_dequant_q8_1(const block_q8_1*, float*, int64_t) { std::abort(); }

template <typename B>
void run(const char* /*type*/, const std::vector<float>& x, void(*quant)(const float*, B*, int64_t), void(*dequant)(const B*, float*, int64_t),
         const char* packed_path, const char* dequant_path, bool only_packed, int64_t block_elems) {
    int64_t n = (int64_t)x.size();
    int64_t nb = n / block_elems;
    std::vector<B> blocks(nb);
    quant(x.data(), blocks.data(), n);
    FILE* fp = fopen(packed_path, "wb");
    fwrite(blocks.data(), sizeof(B), nb, fp);
    fclose(fp);
    if (!only_packed) {
        std::vector<float> y(n);
        dequant(blocks.data(), y.data(), n);
        FILE* fd = fopen(dequant_path, "wb");
        fwrite(y.data(), sizeof(float), n, fd);
        fclose(fd);
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <type> <in_f32.bin> <packed.bin> <dequant.bin> [--only-packed]\n", argv[0]); return 2; }
    const char* type = argv[1];
    const bool only_packed = argc > 5 && !strcmp(argv[5], "--only-packed");
    FILE* fi = fopen(argv[2], "rb");
    if (!fi) { fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
    fseek(fi, 0, SEEK_END); long n = ftell(fi) / 4; fseek(fi, 0, SEEK_SET);
    std::vector<float> x(n);
    if ((long)fread(x.data(), 4, n, fi) != n) { fprintf(stderr, "read failed\n"); return 2; }
    fclose(fi);

    if (!strcmp(type, "q1_0"))      { run<block_q1_0>(type, x, quantize_row_q1_0_ref, dequantize_row_q1_0, argv[3], argv[4], only_packed, 128); return 0; }
    else if (!strcmp(type, "q2_0")) { run<block_q2_0>(type, x, quantize_row_q2_0_ref, dequantize_row_q2_0, argv[3], argv[4], only_packed, 64); return 0; }
    else if (!strcmp(type, "q4_0")) { run<block_q4_0>(type, x, quantize_row_q4_0_ref, dequantize_row_q4_0, argv[3], argv[4], only_packed, 32); return 0; }
    else if (!strcmp(type, "q4_1")) { run<block_q4_1>(type, x, quantize_row_q4_1_ref, dequantize_row_q4_1, argv[3], argv[4], only_packed, 32); return 0; }
    else if (!strcmp(type, "q5_0")) { run<block_q5_0>(type, x, quantize_row_q5_0_ref, dequantize_row_q5_0, argv[3], argv[4], only_packed, 32); return 0; }
    else if (!strcmp(type, "q5_1")) { run<block_q5_1>(type, x, quantize_row_q5_1_ref, dequantize_row_q5_1, argv[3], argv[4], only_packed, 32); return 0; }
    else if (!strcmp(type, "q8_0")) { run<block_q8_0>(type, x, quantize_row_q8_0_ref, dequantize_row_q8_0, argv[3], argv[4], only_packed, 32); return 0; }
    else if (!strcmp(type, "q8_1")) { run<block_q8_1>(type, x, quantize_row_q8_1_ref, fake_dequant_q8_1, argv[3], argv[4], only_packed, 32); return 0; }
    else if (!strcmp(type, "q2_k")) { run<block_q2_K>(type, x, quantize_row_q2_K_ref, dequantize_row_q2_K, argv[3], argv[4], only_packed, 256); return 0; }
    else if (!strcmp(type, "q3_k")) { run<block_q3_K>(type, x, quantize_row_q3_K_ref, dequantize_row_q3_K, argv[3], argv[4], only_packed, 256); return 0; }
    else if (!strcmp(type, "q4_k")) { run<block_q4_K>(type, x, quantize_row_q4_K_ref, dequantize_row_q4_K, argv[3], argv[4], only_packed, 256); return 0; }
    else if (!strcmp(type, "q5_k")) { run<block_q5_K>(type, x, quantize_row_q5_K_ref, dequantize_row_q5_K, argv[3], argv[4], only_packed, 256); return 0; }
    else if (!strcmp(type, "q6_k")) { run<block_q6_K>(type, x, quantize_row_q6_K_ref, dequantize_row_q6_K, argv[3], argv[4], only_packed, 256); return 0; }
    else if (!strcmp(type, "q8_k")) { run<block_q8_K>(type, x, quantize_row_q8_K_ref, dequantize_row_q8_K, argv[3], argv[4], only_packed, 256); return 0; }
    fprintf(stderr, "unknown type %s\n", type);
    return 2;
}
//
// Differential oracle for this crate's K-type quantization vs llama.cpp.
//
// Build (against a local llama.cpp checkout, using its built libraries):
//   g++ -O2 -std=c++17 \
//     -I<llama.cpp>/ggml/include -I<llama.cpp>/ggml/src \
//     crates/oxide-infer-core/tests/k_oracle.cpp \
//     -L<llama.cpp>/build/bin -lggml-base -Wl,-rpath,<llama.cpp>/build/bin \
//     -o /tmp/k_oracle
//
// Then generate the artifacts the test reads (any multiple of 256 f32s):
//   for t in q1_0 q2_0 q4_0 q4_1 q5_0 q5_1 q8_0 q8_1 q2_k q3_k q4_k q5_k q6_k q8_k; do
//     /tmp/k_oracle $t $DIR/x.bin $DIR/${t}_packed.bin $DIR/${t}_dequant.bin
//   done
//   /tmp/k_oracle q8_1 $DIR/x.bin $DIR/q8_1_packed.bin $DIR/x.bin --only-packed
//
// Run the test with:
//   OXIDE_K_ORACLE_DIR=$DIR cargo test --release -p oxide-infer-core --test k_oracle
