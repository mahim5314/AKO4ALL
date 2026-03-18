/*
 * CUDA Kernel Template for FlashInfer Competition.
 *
 * Implement your kernel logic here. The Python bindings in binding.py
 * will call functions from this file via TVM FFI.
 *
 * Getting started:
 *   1. Read the `reference` field in docs/definition.json to understand the computation semantics
 *   2. Read docs/definition.json for input/output shapes and dtypes
 *   3. Implement the kernel logic below
 *   4. Update binding.py to call your implementation
 *   5. Run: bash scripts/bench.sh
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void kernel() {
    /*
     * Your CUDA kernel implementation.
     *
     * TODO: Implement your kernel according to docs/definition.json.
     * Study the `reference` field in docs/definition.json for the PyTorch reference implementation.
     */
}
