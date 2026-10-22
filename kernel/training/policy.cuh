// SPDX-License-Identifier: Apache-2.0
// The policy network (see core/config.h) and its A2C gradient for one sample,
// summed across the warp.
//
// Performance rule for this file: every index into a per-thread array must be
// a compile-time constant after unrolling. A runtime index moves the array to
// local memory, whose L2 traffic slows the LLM sharing the GPU (see
// docs/architecture.md).
#pragma once

#include "../core/device_util.cuh"
#include "../core/state.cuh"

namespace cotenant {

static_assert(kHidden == 2 * kWarpSize,
              "warp_add_row64 gives each lane two hidden units");

// Copies the newest weights into the block's shared memory. Called by every
// thread of the block; the caller synchronizes.
__device__ __forceinline__ void load_weights(float *w) {
  for (int p = threadIdx.x; p < kParamsPadded; p += kThreads) {
    w[p] = __ldcg(g_ct.weights + p);
  }
}

// ---------------------------------------------------------------- forward

// h1 = relu(W1 o + b1)
__device__ __forceinline__ void
layer1(float const *w, float const *o, float (&h1)[kHidden]) {
#pragma unroll
  for (int j = 0; j < kHidden; j++) {
    float z = w[kOffB1 + j];
#pragma unroll
    for (int k = 0; k < kObs; k++) {
      z += w[kOffW1 + j * kObs + k] * o[k];
    }
    h1[j] = fmaxf(z, 0.f);
  }
}

// Pre-activation of layer-2 unit i: W2[i] . h1 + b2[i].
__device__ __forceinline__ float
layer2_pre(float const *w, float const (&h1)[kHidden], int i) {
  float z = w[kOffB2 + i];
#pragma unroll
  for (int j = 0; j < kHidden; j++) {
    z += w[kOffW2 + i * kHidden + j] * h1[j];
  }
  return z;
}

struct PolicyOutput {
  float logit; // of flapping
  float value; // of the state
};

// Layer 2 and the heads, one hidden unit at a time, so that h2 is never held.
__device__ __forceinline__ PolicyOutput heads(float const *w,
                                              float const (&h1)[kHidden]) {
  PolicyOutput out{w[kOffB3], w[kOffB3 + 1]};
#pragma unroll 4
  for (int i = 0; i < kHidden; i++) {
    float h = fmaxf(layer2_pre(w, h1, i), 0.f);
    out.logit += w[kOffW3 + i] * h;
    out.value += w[kOffW3 + kHidden + i] * h;
  }
  return out;
}

__device__ __forceinline__ PolicyOutput forward(float const *w,
                                                float const *o) {
  float h1[kHidden];
  layer1(w, o, h1);
  return heads(w, h1);
}

__device__ __forceinline__ float flap_probability(float logit) {
  return 1.f / (1.f + __expf(-logit));
}

// ---------------------------------------------------------------- reductions

// Adds the warp's sum of `v` to *dst (one atomic per warp).
__device__ __forceinline__ void warp_add(float *dst, float v, int lane) {
  v = warp_sum(v);
  if (lane == 0) {
    atomicAdd(dst, v);
  }
}

// One reduce-scatter step: lanes split r[0..2N) by lane bit N/2 and keep the
// pairwise sum of their half in r[0..N). N is a template parameter so that
// every index into r stays compile-time.
template <int N>
__device__ __forceinline__ void butterfly_stage(float (&r)[kWarpSize],
                                                int lane) {
  bool upper = lane & (N / 2);
#pragma unroll
  for (int k = 0; k < N; k++) {
    float keep = upper ? r[k + N] : r[k];
    float send = upper ? r[k] : r[k + N];
    r[k] = keep + __shfl_xor_sync(kFullWarp, send, N / 2);
  }
}

// Adds the warp's sum of the 64-vector `scale * x` to dst[0], dst[stride], ...
// dst[63 * stride], with a reduce-scatter butterfly (62 shuffles instead of
// 320). Lane l ends up owning entries 2l and 2l+1.
__device__ __forceinline__ void warp_add_row64(
    float *dst, int stride, float scale, float const (&x)[kHidden], int lane) {
  float r[kWarpSize];
  bool upper = lane & 16;
#pragma unroll
  for (int k = 0; k < kWarpSize; k++) {
    float keep = scale * (upper ? x[k + kWarpSize] : x[k]);
    float send = scale * (upper ? x[k] : x[k + kWarpSize]);
    r[k] = keep + __shfl_xor_sync(kFullWarp, send, 16);
  }
  butterfly_stage<16>(r, lane);
  butterfly_stage<8>(r, lane);
  butterfly_stage<4>(r, lane);
  butterfly_stage<2>(r, lane);
  atomicAdd(dst + (2 * lane) * stride, r[0]);
  atomicAdd(dst + (2 * lane + 1) * stride, r[1]);
}

// ---------------------------------------------------------------- backward

// Adds one sample's A2C gradient, summed over the warp, into g. The loss is
//   -log pi(act | o) * A  -  kEntropyCoef * H(pi)  +  kValueCoef * A^2
// with the advantage A = ret - V(o) held constant in the policy term.
__device__ void accumulate_grad(
    float const *w, float *g, float const *o, float act, float ret, int lane) {
  float h1[kHidden];
  layer1(w, o, h1);
  PolicyOutput const out = heads(w, h1);

  float const p = flap_probability(out.logit);
  float const adv = ret - out.value;
  float const dlogit =
      -(act - p) * adv + kEntropyCoef * out.logit * p * (1.f - p);
  float const dvalue = -2.f * kValueCoef * adv;
  warp_add(g + kOffB3, dlogit, lane);
  warp_add(g + kOffB3 + 1, dvalue, lane);

  // Layer 2, recomputing each unit instead of keeping h2. Units that are off
  // for the whole warp contribute nothing and are skipped.
  float dh1[kHidden];
#pragma unroll
  for (int j = 0; j < kHidden; j++) {
    dh1[j] = 0.f;
  }
#pragma unroll 1
  for (int i = 0; i < kHidden; i++) {
    float const z = layer2_pre(w, h1, i);
    float const h = fmaxf(z, 0.f);
    float const dz =
        z > 0.f ? w[kOffW3 + i] * dlogit + w[kOffW3 + kHidden + i] * dvalue
                : 0.f;
    warp_add(g + kOffW3 + i, dlogit * h, lane);
    warp_add(g + kOffW3 + kHidden + i, dvalue * h, lane);
    if (__any_sync(kFullWarp, dz != 0.f)) {
      warp_add(g + kOffB2 + i, dz, lane);
      warp_add_row64(g + kOffW2 + i * kHidden, 1, dz, h1, lane);
#pragma unroll
      for (int j = 0; j < kHidden; j++) {
        dh1[j] += w[kOffW2 + i * kHidden + j] * dz;
      }
    }
  }

  // Layer 1.
#pragma unroll
  for (int j = 0; j < kHidden; j++) {
    dh1[j] = h1[j] > 0.f ? dh1[j] : 0.f;
  }
#pragma unroll
  for (int j = 0; j < kHidden; j++) {
    warp_add(g + kOffB1 + j, dh1[j], lane);
  }
  // dW1 is laid out [j][k]: each input k is a strided 64-vector.
#pragma unroll
  for (int k = 0; k < kObs; k++) {
    warp_add_row64(g + kOffW1 + k, kObs, o[k], dh1, lane);
  }
}

} // namespace cotenant
