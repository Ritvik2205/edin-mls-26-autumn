// Task 3: the linear layer as a tiled GEMM. Read docs/task3-gemm.md first.
//
//     y[M, N] = act( x[M, K] * w[N, K]^T + bias[N] )
//
// This is the tiled GEMM from cuda-tutorial/5-tiled-gemm, with three changes:
//   1. w is stored the way PyTorch stores Linear weights: one row per OUTPUT
//      feature, [N, K]. So column n of the result uses row n of w.
//   2. bias is added, and GELU can be applied, before the result is stored
//      (the "epilogue"), so no extra kernel has to read y again.
//   3. every Linear layer of the ASR model and both convolutions run through it.
//
// Block (TILE x TILE threads) computes a TILE x TILE piece of y. Thread (ty, tx)
// computes y[row][col] with row = blockIdx.y*TILE + ty, col = blockIdx.x*TILE + tx.
// The block walks along K one tile at a time, staging a tile of x and a tile of w
// in shared memory so every loaded value is used TILE times.
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "asr_kernels.h"
#include "common.cuh"
#include "task2_reduce.cuh"

constexpr int TILE = 16;  // try 8 or 32 (32 x 32 = 1024 threads, the maximum per block)

__global__ void linear_tiled_kernel(const float* x, const float* w, const float* bias,
                                    float* y, int M, int N, int K, int act) {
  __shared__ float xs[TILE][TILE];      // xs[m][k]: rows of x for this block
  __shared__ float ws[TILE][TILE + 1];  // ws[n][k]: rows of w for this block (+1: see Try this)

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;  // the output row this thread computes
  const int col = blockIdx.x * TILE + tx;  // the output column this thread computes

  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    // ======================== BLANK 3.1 =========================
    // Load this thread's element of the x tile into xs. Use 0 outside x.
    {
      const int k = k0 + tx;
      xs[ty][tx] = (row < M && k < K) ? x[static_cast<size_t>(row) * K + k] : 0.0f;
    }
    // ============================================================

    // ======================== BLANK 3.2 =========================
    // Load this thread's element of the w tile, so that ws[n][k] holds
    // w[blockIdx.x * TILE + n][k0 + k]. Use 0 outside w. Threads with consecutive
    // tx must read consecutive addresses of w.
    {
      // Thread (ty, tx) loads w[n = blockIdx.x*TILE + ty][k = k0 + tx]: consecutive tx
      // read consecutive k of one row of w, so the load is coalesced.
      const int n = blockIdx.x * TILE + ty;
      const int k = k0 + tx;
      ws[ty][tx] = (n < N && k < K) ? w[static_cast<size_t>(n) * K + k] : 0.0f;
    }
    __syncthreads();  // both tiles fully loaded before anyone reads them
    // ============================================================

    // ======================== BLANK 3.3 =========================
    // Add this K step's contribution to acc.
#pragma unroll
    for (int kk = 0; kk < TILE; ++kk) acc += xs[ty][kk] * ws[tx][kk];
    __syncthreads();  // everyone done reading before the next k0 overwrites the tiles
    // ============================================================
  }

  // ======================== BLANK 3.4 =========================
  // Epilogue: for outputs inside y, add bias[col] (bias may be nullptr), apply
  // gelu() when act == ACT_GELU, and store.
  if (row < M && col < N) {
    if (bias != nullptr) acc += bias[col];
    if (act == ACT_GELU) acc = gelu(acc);
    y[static_cast<size_t>(row) * N + col] = acc;
  }
  // ============================================================
}

// ---------------------------------------------------------------------------
// Extension (not one of the HW1 blanks): M = 1, a matrix-vector product.
//
// With a KV cache every decoder linear layer has M = 1. The tiled kernel then
// keeps only one of its TILE thread rows busy and still pays two __syncthreads
// per K tile. The two kernels below drop the shared-memory tiles. They are
// selected by the environment variable ASR_GEMV, read at every launch:
//   unset / "off"  the tiled kernel above for every M (the default, and what
//                  make test and ./run.sh measure)
//   "seq"          one thread per output column, summing over k in ascending
//                  order with the expression of BLANK 3.3, so the result is
//                  bit-identical to the tiled kernel
//   "warp"         one warp per output column, coalesced loads, 32 partial sums
//                  joined by a warp reduction: a different summation order, so
//                  equal only to rounding
// launch_linear keeps its name and inputs.
constexpr int GEMV_THREADS = 256;  // "seq": output columns per block
constexpr int GEMV_WARPS = 8;      // "warp": output columns per block

__device__ __forceinline__ void gemv_store(float acc, const float* bias, float* y, int col, int act) {
  if (bias != nullptr) acc += bias[col];
  if (act == ACT_GELU) acc = gelu(acc);
  y[col] = acc;
}

// VEC: x and w are 16-byte aligned and K is a multiple of 4, so both can be read as float4.
template <bool VEC>
__global__ void linear_gemv_seq_kernel(const float* x, const float* w, const float* bias,
                                       float* y, int N, int K, int act) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= N) return;
  const float* wr = w + static_cast<size_t>(col) * K;  // this thread streams one row of w
  float acc = 0.0f;
  if (VEC) {
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const float4* w4 = reinterpret_cast<const float4*>(wr);
    for (int i = 0; i < K / 4; ++i) {
      const float4 xv = x4[i];  // the same address in every thread: one cached transaction
      const float4 wv = w4[i];
      acc += xv.x * wv.x;
      acc += xv.y * wv.y;
      acc += xv.z * wv.z;
      acc += xv.w * wv.w;
    }
  } else {
    for (int k = 0; k < K; ++k) acc += x[k] * wr[k];
  }
  gemv_store(acc, bias, y, col, act);
}

template <bool VEC>
__global__ void linear_gemv_warp_kernel(const float* x, const float* w, const float* bias,
                                        float* y, int N, int K, int act) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int col = blockIdx.x * GEMV_WARPS + threadIdx.x / WARP_SIZE;
  if (col >= N) return;  // the whole warp leaves together
  const float* wr = w + static_cast<size_t>(col) * K;
  float acc = 0.0f;
  if (VEC) {
    // Lane l takes float4 number l, l + 32, ...: one warp load is 512 consecutive bytes.
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const float4* w4 = reinterpret_cast<const float4*>(wr);
    for (int i = lane; i < K / 4; i += WARP_SIZE) {
      const float4 xv = x4[i];
      const float4 wv = w4[i];
      acc += xv.x * wv.x;
      acc += xv.y * wv.y;
      acc += xv.z * wv.z;
      acc += xv.w * wv.w;
    }
  } else {
    for (int k = lane; k < K; k += WARP_SIZE) acc += x[k] * wr[k];
  }
  acc = warp_reduce_sum(acc);
  if (lane == 0) gemv_store(acc, bias, y, col, act);
}

static int gemv_mode() {
  const char* s = std::getenv("ASR_GEMV");
  if (s == nullptr) return 0;
  if (std::strcmp(s, "seq") == 0) return 1;
  if (std::strcmp(s, "warp") == 0) return 2;
  return 0;
}

ASR_API int launch_linear(const float* x, const float* w, const float* bias, float* y,
                          int M, int N, int K, int act, cudaStream_t stream) {
  const int mode = (M == 1) ? gemv_mode() : 0;
  if (mode != 0) {
    const bool vec = K % 4 == 0 &&
                     ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w)) & 15u) == 0;
    if (mode == 1) {
      const dim3 grid(ceil_div(N, GEMV_THREADS));
      if (vec) linear_gemv_seq_kernel<true><<<grid, GEMV_THREADS, 0, stream>>>(x, w, bias, y, N, K, act);
      else linear_gemv_seq_kernel<false><<<grid, GEMV_THREADS, 0, stream>>>(x, w, bias, y, N, K, act);
    } else {
      const dim3 grid(ceil_div(N, GEMV_WARPS));
      const int threads = GEMV_WARPS * WARP_SIZE;
      if (vec) linear_gemv_warp_kernel<true><<<grid, threads, 0, stream>>>(x, w, bias, y, N, K, act);
      else linear_gemv_warp_kernel<false><<<grid, threads, 0, stream>>>(x, w, bias, y, N, K, act);
    }
    RETURN_LAUNCH_ERROR();
  }
  dim3 block(TILE, TILE);
  dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
  linear_tiled_kernel<<<grid, block, 0, stream>>>(x, w, bias, y, M, N, K, act);
  RETURN_LAUNCH_ERROR();
}
