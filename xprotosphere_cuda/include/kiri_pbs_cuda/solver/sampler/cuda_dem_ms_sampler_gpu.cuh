/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-02-23 22:50:51
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-02-23 23:32:04
 * @FilePath:
 * \XProtoSphere\xprotosphere_cuda\include\kiri_pbs_cuda\solver\sampler\cuda_dem_ms_sampler_gpu.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _CUDA_DEM_MS_SAMPLER_GPU_CUH_
#define _CUDA_DEM_MS_SAMPLER_GPU_CUH_

#pragma once
#include <kiri_pbs_cuda/math/cuda_math_colormap.cuh>
#include <kiri_pbs_cuda/solver/cuda_solver_common_gpu.cuh>
namespace KIRI {

template <typename SDFObj>
__global__ void
_IdentifyBoundaryParticles_CUDA(bool *isBoundary, const float3 *pos,
                                const float3 *vel, const float *sdfData,
                                const float dt, const size_t num, SDFObj geo) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  auto ppos = pos[i] + vel[i] * dt;
  auto phi = geo.SDF(ppos, sdfData);

  if (phi <= 0)
    isBoundary[i] = true;
  else
    isBoundary[i] = false;

  return;
}

static __device__ void _ComputeSamplerOverlap(float *overlap, size_t *counter,
                                              const size_t i, const float3 *pos,
                                              const float *radius,
                                              const size_t cellStart,
                                              const size_t cellEnd) {
  size_t j = cellStart;
  while (j < cellEnd) {

    if (i != j) {
      auto rij = radius[i] + radius[j];
      auto dist = length(pos[i] - pos[j]);
      if (dist < rij) {
        *overlap += rij - dist;
        *counter += 1;
      }
    }
    ++j;
  }
  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash>
__global__ void
_ComputeSamplerOverlapRatio_CUDA(float *overlap, const float3 *pos,
                                 const float *radius, const size_t num,
                                 size_t *cellStart, const int3 gridSize,
                                 Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  auto overlap_ratio = 0.f;
  size_t overlap_num = 0;
  int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
  for (int m = 0; m < 27; ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputeSamplerOverlap(&overlap_ratio, &overlap_num, i, pos, radius,
                           cellStart[hash_idx], cellStart[hash_idx + 1]);
  }

  if (overlap_num != 0)
    overlap_ratio /= (radius[i] * overlap_num);

  overlap[i] = overlap_ratio;
  return;
}

static __global__ void _UpdateSamplerVelocities_CUDA(float3 *vel,
                                                     const float3 *acc,
                                                     const float dt,
                                                     const size_t num) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  auto damp_a = acc[i] * (make_float3(1.f) -
                          0.4f * sgn(acc[i] * (vel[i] + 0.5f * dt * acc[i])));

  vel[i] += damp_a * dt;
  // vel[i] += acc[i] * dt;
  return;
}

template <typename SDFObj>
__global__ void _MoveDEMMSSamplerParticlesWithinBoundary_CUDA(
    float3 *pos, float3 *lpos, float3 *vel, float3 *col, const float *overlap,
    const float *sdfData, const float dt, const float boundaryRestitution,
    const float minOverlap, const float maxOverlap, const bool renderOverlap,
    const size_t num, SDFObj geo) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;
  lpos[i] = pos[i];
  auto pvel = vel[i];
  auto ppos = pos[i] + pvel * dt;
  auto phi = geo.SDF(ppos, sdfData);

  if (phi <= 0) {
    auto grad = geo.gradSDF(ppos, sdfData);
    auto mag2_grad = lengthSquared(grad);
    if (mag2_grad > Tiny<float>()) {
      grad /= sqrtf(mag2_grad);
      ppos -= grad * phi;
      vel[i] = reflect(pvel, grad) * boundaryRestitution;
    }
  }

  pos[i] = ppos;

  if (renderOverlap)
    col[i] =
        GetColor<float>((overlap[i] - minOverlap) / (maxOverlap - minOverlap),
                        ColormapType::Jet)
            .Data();

  return;
}

static __device__ float3 _ComputeDemForces(float3 dij, float3 vij, float rij,
                                           float kn, float ks,
                                           float tanFrictionAngle) {
  float3 f = make_float3(0.f);
  float dist = length(dij);
  float penetration_depth = rij - dist;
  if (penetration_depth > 0.f) {
    float3 n = dij / dist;
    float dot_epslion = dot(vij, n);
    float3 vij_tangential = vij - dot_epslion * n;

    float3 normal_force = kn * penetration_depth * n;

    f = -normal_force;
  }
  return f;
}

static __device__ void
_ComputeMRDemForces(float3 *f, const size_t i, const float3 *pos,
                    const float3 *lpos, const float3 *vel, const float *radius,
                    const float *overlap, const float young,
                    const float poisson, const float tanFrictionAngle, size_t j,
                    const size_t cellEnd, curandState *state) {

  while (j < cellEnd) {

    if (i != j) {

      float3 dij = pos[j] - pos[i];
      float dist = length(dij);

      float r2 = lengthSquared(dij);
      float rij = radius[i] + radius[j];
      float3 vij = vel[j] - vel[i];
      float kni = young * radius[i];
      float knj = young * radius[j];
      float ksi = kni * poisson;
      float ksj = knj * poisson;

      float kn = 2.f * kni * knj / (kni + knj);
      float ks = 2.f * ksi * ksj / (ksi + ksj);

      *f += _ComputeDemForces(dij, vij, rij, kn, ks, tanFrictionAngle);
    }

    ++j;
  }
  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash>
__global__ void _ComputeMRDemForces_CUDA(
    const float3 *pos, const float3 *lpos, const float3 *vel, float3 *acc,
    const float *mass, const float *radius, const float *overlap,
    const float young, const float poisson, const float tanFrictionAngle,
    const float sr, const size_t num, const float3 lowestPoint,
    const float3 highestPoint, size_t *cellStart, const int3 gridSize,
    Pos2GridXYZ p2xyz, GridXYZ2GridHash xyz2hash, curandState *state) {
  const size_t i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  auto f = make_float3(0.f);
  int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
  for (int m = 0; m < 27; ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _ComputeMRDemForces(&f, i, pos, lpos, vel, radius, overlap, young, poisson,
                        tanFrictionAngle, cellStart[hash_idx],
                        cellStart[hash_idx + 1], state);
  }

  acc[i] += 2.f * f / mass[i];
  return;
}

} // namespace KIRI

#endif /* _CUDA_DEM_MS_SAMPLER_GPU_CUH_ */