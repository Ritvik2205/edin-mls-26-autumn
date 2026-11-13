// Task 1: rotary position embedding (RoPE). Read docs/task1-rope.md first.
//
// Before attention, every query and key vector is rotated by an angle that depends
// on the token's position, so attention can tell tokens apart by where they are.
// One vector has head_dim numbers. RoPE pairs element i with element i + half,
// for i = 0 .. half-1 (half = rotary_dim / 2), and rotates each pair:
//
//     angle  = pos * inv_freq[i]
//     new_a  = a * cos(angle) - b * sin(angle)       a = x[i]
//     new_b  = b * cos(angle) + a * sin(angle)       b = x[i + half]
//
// Elements rotary_dim .. head_dim-1 are left unchanged (the audio encoder only
// rotates the first half of each head; the text decoder rotates all of it).
//
// Memory layout, straight out of the q/k projections (no transposes):
//     q: [batch, seq_len, num_q_heads, head_dim]
//     k: [batch, seq_len, num_k_heads, head_dim]
// Seen as a 2D array, each row is one head of one token.
//
// One launch rotates q AND k: one thread per pair, q's pairs first, then k's.
#include "asr_kernels.h"
#include "common.cuh"

constexpr int ROPE_THREADS = 256;

__global__ void rope_kernel(float* q, float* k, const float* inv_freq, int seq_len,
                            int num_q_heads, int num_k_heads, int q_rows, int k_rows,
                            int head_dim, int rotary_dim) {
  const int half = rotary_dim / 2;
  const int q_pairs = q_rows * half;
  const int k_pairs = k_rows * half;

  int idx = 0;  // this thread's pair number, counting q's pairs first, then k's
  // ======================== BLANK 1.1 =========================
  // Set idx to this thread's global index. Threads without a pair
  // (q_pairs + k_pairs in total) do nothing.
  idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= q_pairs + k_pairs) return;
  // ============================================================

  float* x = q;               // the tensor this thread works on
  int num_heads = num_q_heads;
  // ======================== BLANK 1.2 =========================
  // Threads past q's pairs work on k: make x, num_heads and idx describe the
  // tensor this thread works on, with idx counting from 0 inside it.
  if (idx >= q_pairs) {
    x = k;
    num_heads = num_k_heads;
    idx -= q_pairs;
  }
  // ============================================================

  int row = 0;  // which row (one head of one token) of x
  int i = 0;    // which pair inside that row, 0 .. half-1
  int pos = 0;  // position of the token this row belongs to
  // ======================== BLANK 1.3 =========================
  // Set row, i (the pair inside the row) and pos (the token position of the row).
  row = idx / half;                 // rows are [batch, seq, head] flattened
  i = idx % half;
  pos = (row / num_heads) % seq_len;  // drop the head index, then the batch index
  // ============================================================

  float* v = x + static_cast<size_t>(row) * head_dim;  // start of this row
  // ======================== BLANK 1.4 =========================
  // Rotate the pair (v[i], v[i + half]) in place.
  const float angle = static_cast<float>(pos) * inv_freq[i];
  const float c = cosf(angle);
  const float s = sinf(angle);
  const float a = v[i];
  const float b = v[i + half];
  v[i] = a * c - b * s;
  v[i + half] = b * c + a * s;
  // ============================================================
}

ASR_API int launch_rope(float* q, float* k, const float* inv_freq, int batch, int seq_len,
                        int num_q_heads, int num_k_heads, int head_dim, int rotary_dim,
                        cudaStream_t stream) {
  int q_rows = batch * seq_len * num_q_heads;
  int k_rows = batch * seq_len * num_k_heads;
  int total_pairs = (q_rows + k_rows) * (rotary_dim / 2);
  int blocks = ceil_div(total_pairs, ROPE_THREADS);
  rope_kernel<<<blocks, ROPE_THREADS, 0, stream>>>(q, k, inv_freq, seq_len, num_q_heads,
                                                    num_k_heads, q_rows, k_rows, head_dim,
                                                    rotary_dim);
  RETURN_LAUNCH_ERROR();
}

// ---------------------------------------------------------------------------
// Extension (not part of the HW1 blanks): RoPE at positions pos_offset .. pos_offset+S-1.
// KV-cache decoding rotates one new token at position P, but launch_rope always starts
// at position 0. This kernel is rope_kernel with the token position shifted by pos_offset.
// launch_rope and rope_kernel above are unchanged (forum #33: keep names and inputs).
__global__ void rope_at_kernel(float* q, float* k, const float* inv_freq, int seq_len,
                               int num_q_heads, int num_k_heads, int q_rows, int k_rows,
                               int head_dim, int rotary_dim, int pos_offset) {
  const int half = rotary_dim / 2;
  const int q_pairs = q_rows * half;
  const int k_pairs = k_rows * half;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= q_pairs + k_pairs) return;
  float* x = q;
  int num_heads = num_q_heads;
  if (idx >= q_pairs) {
    x = k;
    num_heads = num_k_heads;
    idx -= q_pairs;
  }
  const int row = idx / half;
  const int i = idx % half;
  const int pos = pos_offset + (row / num_heads) % seq_len;
  float* v = x + static_cast<size_t>(row) * head_dim;
  const float angle = static_cast<float>(pos) * inv_freq[i];  // same float ops as rope_kernel
  const float c = cosf(angle);
  const float s = sinf(angle);
  const float a = v[i];
  const float b = v[i + half];
  v[i] = a * c - b * s;
  v[i + half] = b * c + a * s;
}

ASR_API int launch_rope_at(float* q, float* k, const float* inv_freq, int batch, int seq_len,
                           int num_q_heads, int num_k_heads, int head_dim, int rotary_dim,
                           int pos_offset, cudaStream_t stream) {
  if (pos_offset < 0) return static_cast<int>(cudaErrorInvalidValue);
  int q_rows = batch * seq_len * num_q_heads;
  int k_rows = batch * seq_len * num_k_heads;
  int total_pairs = (q_rows + k_rows) * (rotary_dim / 2);
  if (total_pairs == 0) return 0;
  int blocks = ceil_div(total_pairs, ROPE_THREADS);
  rope_at_kernel<<<blocks, ROPE_THREADS, 0, stream>>>(q, k, inv_freq, seq_len, num_q_heads,
                                                       num_k_heads, q_rows, k_rows, head_dim,
                                                       rotary_dim, pos_offset);
  RETURN_LAUNCH_ERROR();
}
