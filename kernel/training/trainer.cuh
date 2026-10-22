// SPDX-License-Identifier: Apache-2.0
// The A2C learner blocks. Every update, each learner block
//   1. plays kRolloutLen steps in each of its kEnvsPerLearner environments,
//   2. turns the rewards into discounted returns,
//   3. sums the gradient of all those samples into a partial gradient,
// then all learners meet at a barrier, and each applies Adam to its own slice
// of the parameters, summing that slice over every learner's partial gradient.
#pragma once

#include "../core/device_util.cuh"
#include "../core/state.cuh"
#include "../game/physics.cuh"
#include "policy.cuh"

namespace cotenant {

struct LearnerSmem {
  float w[kParamsPadded]; // this update's weights
  float g[kParamsPadded]; // this block's gradient sum
  EpisodeStats warp_stats[kWarps];
  int flag; // broadcasts the barrier's stop decision
};

// Barrier across the learner blocks only, called by every thread. The last
// block to arrive decides whether to stop, so every learner leaves the loop
// at the same update. Returns that decision.
__device__ bool learner_barrier(int &flag) {
  LearnerBarrier *bar = g_ct.barrier;
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned generation = ld_acquire(&bar->generation);
    __threadfence();
    unsigned arrived = atomicAdd(&bar->count, 1u);
    if (arrived == unsigned(g_ct.num_learners) - 1) {
      atomicExch(&bar->count, 0u);
      int stop = ld_volatile(g_ct.stop) | ld_volatile(&g_ct.host->stop) |
                 (ld_volatile(&g_ct.progress->updates) + 1 >= g_ct.max_updates);
      st_volatile(&bar->stop, stop);
      __threadfence();
      atomicAdd(&bar->generation, 1u);
    } else {
      while (ld_acquire(&bar->generation) == generation) {
        __nanosleep(64);
      }
    }
    flag = ld_volatile(&bar->stop);
  }
  __syncthreads();
  return flag != 0;
}

// Step 1, per thread: plays kRolloutLen steps of `env`, sampling actions from
// the policy, records them in this thread's column of `r`, and returns the
// episodes that finished.
__device__ __forceinline__ EpisodeStats play_rollout(Env &env,
                                                     uint32_t &action_rng,
                                                     float const *w,
                                                     Rollout &r) {
  int const e = threadIdx.x;
  EpisodeStats ep{0.f, 0.f, 0.f};
  for (int t = 0; t < kRolloutLen; t++) {
    RolloutStep &step = r.step[t];
    float o[kObs];
    env_observe(env, env.y, env.vy, o);
    bool const flap =
        uniform01(action_rng) < flap_probability(forward(w, o).logit);
    bool dead;
    float final_score;
    float const reward = env_step(env, flap, dead, final_score);
#pragma unroll
    for (int k = 0; k < kObs; k++) {
      step.obs[k][e] = o[k];
    }
    step.action[e] = flap ? 1.f : 0.f;
    step.reward[e] = reward;
    step.done[e] = dead ? 1.f : 0.f;
    if (dead) {
      ep.score_sum += final_score;
      ep.count += 1.f;
      ep.best_score = fmaxf(ep.best_score, final_score);
    }
  }
  return ep;
}

// Step 2, per thread: replaces the rewards by discounted returns, bootstrapped
// from the value of the state the rollout ended in.
__device__ __forceinline__ void
compute_returns(Env const &env, float const *w, Rollout &r) {
  int const e = threadIdx.x;
  float o[kObs];
  env_observe(env, env.y, env.vy, o);
  float ret = forward(w, o).value;
  for (int t = kRolloutLen - 1; t >= 0; t--) {
    RolloutStep &step = r.step[t];
    ret = step.reward[e] + kDiscount * ret * (1.f - step.done[e]);
    step.reward[e] = ret;
  }
}

// Step 3, per thread: adds the gradient of this thread's samples to g.
__device__ __forceinline__ void
accumulate_rollout_grad(float const *w, float *g, Rollout const &r) {
  int const e = threadIdx.x;
  // From the unsigned threadIdx.x: with a signed lane the compiler cannot
  // simplify the reductions' lane selects, which costs ~2% of training speed.
  int const lane = threadIdx.x % kWarpSize;
  for (int t = 0; t < kRolloutLen; t++) {
    RolloutStep const &step = r.step[t];
    float o[kObs];
#pragma unroll
    for (int k = 0; k < kObs; k++) {
      o[k] = step.obs[k][e];
    }
    accumulate_grad(w, g, o, step.action[e], step.reward[e], lane);
  }
}

// Publishes this block's gradient and episode stats for the other learners.
// Called by every thread with its own stats.
__device__ __forceinline__ void
publish_partials(int learner, LearnerSmem &s, EpisodeStats ep) {
  int const lane = threadIdx.x % kWarpSize, warp = threadIdx.x / kWarpSize;
  ep.score_sum = warp_sum(ep.score_sum);
  ep.count = warp_sum(ep.count);
  ep.best_score = warp_max(ep.best_score);
  if (lane == 0) {
    s.warp_stats[warp] = ep;
  }
  __syncthreads();
  float *out = g_ct.partial_grads + size_t(learner) * kParamsPadded;
  for (int p = threadIdx.x; p < kParamsPadded; p += kThreads) {
    __stcg(out + p, s.g[p]);
  }
  if (threadIdx.x == 0) {
    EpisodeStats block{0.f, 0.f, 0.f};
    for (int i = 0; i < kWarps; i++) {
      block.score_sum += s.warp_stats[i].score_sum;
      block.count += s.warp_stats[i].count;
      block.best_score = fmaxf(block.best_score, s.warp_stats[i].best_score);
    }
    EpisodeStats *dst = g_ct.episode_stats + learner;
    __stcg(&dst->score_sum, block.score_sum);
    __stcg(&dst->count, block.count);
    __stcg(&dst->best_score, block.best_score);
  }
}

// Applies Adam to this learner's slice of the parameters, averaging the
// gradient over every sample of every learner. `step` counts from 0.
__device__ __forceinline__ void adam_update_slice(int learner, long long step) {
  int const nl = g_ct.num_learners;
  int const per_learner = (kParamsPadded + nl - 1) / nl;
  int const lo = learner * per_learner;
  int const hi = min(kParamsPadded, lo + per_learner);
  float const inv_samples = 1.f / float(nl * kEnvsPerLearner * kRolloutLen);
  float const bias1 = 1.f - powf(kAdamBeta1, float(step + 1));
  float const bias2 = 1.f - powf(kAdamBeta2, float(step + 1));
  for (int p = lo + threadIdx.x; p < hi; p += kThreads) {
    if (p >= kParams) {
      continue;
    }
    float grad = 0.f;
    for (int b = 0; b < nl; b++) {
      grad += __ldcg(g_ct.partial_grads + size_t(b) * kParamsPadded + p);
    }
    grad *= inv_samples;
    float const m = kAdamBeta1 * g_ct.adam_m[p] + (1.f - kAdamBeta1) * grad;
    float const v =
        kAdamBeta2 * g_ct.adam_v[p] + (1.f - kAdamBeta2) * grad * grad;
    g_ct.adam_m[p] = m;
    g_ct.adam_v[p] = v;
    __stcg(g_ct.weights + p,
           __ldcg(g_ct.weights + p) -
               kLearningRate * (m / bias1) / (sqrtf(v / bias2) + kAdamEps));
  }
}

// Learner 0, thread 0: totals the update's episodes across learners and
// publishes the progress counters.
__device__ void record_progress(long long step) {
  EpisodeStats total{0.f, 0.f, 0.f};
  for (int b = 0; b < g_ct.num_learners; b++) {
    EpisodeStats const *src = g_ct.episode_stats + b;
    total.score_sum += __ldcg(&src->score_sum);
    total.count += __ldcg(&src->count);
    total.best_score = fmaxf(total.best_score, __ldcg(&src->best_score));
  }
  HostShared volatile *h = g_ct.host;
  // With no finished episodes (an agent that no longer dies) keep the
  // previous mean rather than reporting 0.
  float const mean =
      total.count > 0.f ? total.score_sum / total.count : h->train_mean_score;

  Progress &progress = *g_ct.progress;
  progress.best_score = fmaxf(progress.best_score, total.best_score);
  progress.episodes += (long long)total.count;
  st_volatile(&progress.updates, step + 1);
  h->updates = step + 1;
  h->episodes = progress.episodes;
  h->train_mean_score = mean;
  h->train_best_score = progress.best_score;
}

// Main loop of learner block `learner` (0-based).
__device__ void run_learner(int learner, char *smem_raw) {
  LearnerSmem &s = *reinterpret_cast<LearnerSmem *>(smem_raw);
  Env *const env_slot =
      g_ct.learner_envs + learner * kEnvsPerLearner + threadIdx.x;
  Rollout &rollout = g_ct.rollouts[learner];

  Env env = *env_slot;
  // Actions draw from their own stream, derived from the env's seed (Knuth's
  // multiplicative hash) so it does not repeat the env's pipe draws.
  uint32_t action_rng = env.rng * 2654435761u + 12345u;
  for (;;) {
    long long const step = ld_volatile(&g_ct.progress->updates);
    load_weights(s.w);
    for (int p = threadIdx.x; p < kParamsPadded; p += kThreads) {
      s.g[p] = 0.f;
    }
    __syncthreads();

    EpisodeStats const ep = play_rollout(env, action_rng, s.w, rollout);
    compute_returns(env, s.w, rollout);
    accumulate_rollout_grad(s.w, s.g, rollout);
    publish_partials(learner, s, ep);
    __threadfence();
    bool const stop = learner_barrier(s.flag);

    adam_update_slice(learner, step);
    if (learner == 0 && threadIdx.x == 0) {
      record_progress(step);
    }
    __threadfence();
    learner_barrier(s.flag); // nobody reads the weights mid-update
    if (stop) {
      break;
    }
  }
  *env_slot = env;
}

} // namespace cotenant
