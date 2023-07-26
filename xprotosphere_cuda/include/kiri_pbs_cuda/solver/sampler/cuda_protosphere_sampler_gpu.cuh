/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-02-21 22:18:22
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-02-21 22:28:35
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\sampler\cuda_protosphere_sampler_gpu.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _CUDA_PROTOSPHERE_SAMPLER_GPU_CUH_
#define _CUDA_PROTOSPHERE_SAMPLER_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/cuda_solver_common_gpu.cuh>

namespace KIRI {

template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename SDFObj>
__global__ void _UpdateSDFWithOffset_CUDA(
    float *sdf, float3 *closest, const float3 *pos, const int *convergence,
    const float *sdfData, const float3 *sdfClosestPointData,
    const float3 *insertedPos, const float *insertedRadius, const uint num,
    size_t *cellStart, const int3 gridSize, Pos2GridXYZ p2xyz,
    GridXYZ2GridHash xyz2hash, SDFObj geo, const float maxSDFVal,
    curandState *state) {
  const uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num || convergence[i])
    return;

  auto rnd_offset =
      (_RndFloat_CUDA(state, i % KIRI_RANDOM_SEEDS) + 0.2f) / 5.f * maxSDFVal;
  auto sdf_val = geo.SDFWithOffset(pos[i], sdfData, rnd_offset);

  auto closest_val = geo.closestPoint(pos[i], sdfClosestPointData);

  int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
  for (int m = 0; m < 27; ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _FindMinistDist(&sdf_val, &closest_val, pos[i], insertedPos, insertedRadius,
                    cellStart[hash_idx], cellStart[hash_idx + 1]);
  }

  __syncthreads();
  sdf[i] = sdf_val;
  closest[i] = closest_val;

  return;
}

__device__ void _FindMinistDist(float *sdf, float3 *closest, const float3 posi,
                                const float3 *insertedPos,
                                const float *insertedRadius,
                                const size_t cellStart, const size_t cellEnd) {
  size_t j = cellStart;
  while (j < cellEnd) {

    auto dist = length(insertedPos[j] - posi) - insertedRadius[j];
    if (dist <= 0.f) {
      *sdf = dist;
      break;
    } else if (dist < *sdf) {
      *sdf = dist;
      *closest =
          insertedPos[j] + insertedRadius[j] * normalize(posi - insertedPos[j]);
    }

    ++j;
  }
  return;
}

__global__ void _SphereMoveRandom_CUDA(float3 *pos, float3 *lastPos,
                                       const uint num, const float halfGridSize,
                                       curandState *state) {
  const uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  float3 dir = make_float3(0.f);
  float theta = _RndFloat_CUDA(state, i % KIRI_RANDOM_SEEDS) * 2.f * KIRI_PI;
  float z = _RndFloat_CUDA(state, i % KIRI_RANDOM_SEEDS) * 2.f - 1.f;
  float len = _RndFloat_CUDA(state, i % KIRI_RANDOM_SEEDS) * halfGridSize;

  dir.x = sqrtf(1.f - z * z) * cosf(theta);
  dir.y = sqrtf(1.f - z * z) * sinf(theta);
  dir.z = sqrtf(1.f - z * z) * cosf(theta);

  float3 move = normalize(dir) * len;
  pos[i] += move;
  lastPos[i] += move;

  return;
}

template <typename Pos2GridXYZ, typename GridXYZ2GridHash, typename SDFObj>
__global__ void
_UpdateSDF_CUDA(float *sdf, float3 *closest, const float3 *pos,
                const int *convergence, const float *sdfData,
                const float3 *sdfClosestPointData, const float3 *insertedPos,
                const float *insertedRadius, const uint num, size_t *cellStart,
                const int3 gridSize, Pos2GridXYZ p2xyz,
                GridXYZ2GridHash xyz2hash, SDFObj geo) {
  const uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num || convergence[i])
    return;

  auto sdf_val = geo.SDF(pos[i], sdfData);
  auto closest_val = geo.closestPoint(pos[i], sdfClosestPointData);

  int3 grid_xyz = p2xyz(pos[i]);

#pragma unroll
  for (int m = 0; m < 27; ++m) {
    int3 cur_grid_xyz =
        grid_xyz + make_int3(m / 9 - 1, (m % 9) / 3 - 1, m % 3 - 1);
    const size_t hash_idx =
        xyz2hash(cur_grid_xyz.x, cur_grid_xyz.y, cur_grid_xyz.z);
    if (hash_idx == (gridSize.x * gridSize.y * gridSize.z))
      continue;

    _FindMinistDist(&sdf_val, &closest_val, pos[i], insertedPos, insertedRadius,
                    cellStart[hash_idx], cellStart[hash_idx + 1]);
  }

  __syncthreads();
  sdf[i] = sdf_val;
  closest[i] = closest_val;

  return;
}

__global__ void _AdvectProtoSphereParticles_CUDA(
    float3 *pos, float3 *lastPos, float *radius, float *learningRate,
    int *convergence, uint *iterationNum, const float *targetRadius,
    const float *sdf, const float3 *closest, const uint num,
    const float errorRate, const uint maxIterationNum, const float decay) {
  const uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num || convergence[i])
    return;

  auto current_radius = sdf[i];
  // printf("current_radius=%.3f \n", current_radius);
  //  current position is incorrect
  if (current_radius < 0.f) {
    convergence[i] = 1;
    if (length(pos[i] - lastPos[i]) == 0.f)
      radius[i] = current_radius;
    else
      pos[i] = lastPos[i];
    return;
  }

  lastPos[i] = pos[i];
  radius[i] = current_radius;

  // check sphere radius is converges or not
  auto radius_dist = targetRadius[i] - current_radius;
  if (abs(radius_dist) < targetRadius[i] * errorRate ||
      iterationNum[i] > maxIterationNum) {
    convergence[i] = 1;
    return;
  }

  // need update pos
  auto q_c = closest[i];
  learningRate[i] *= 1.0 / (1.0 + decay * iterationNum[i]);
  auto current_move = radius_dist * learningRate[i] * normalize(pos[i] - q_c);

  if (length(current_move) <= current_radius)
    pos[i] += current_move;
  // printf("pos=%.3f,%.3f,%.3f \n", pos[i].x, pos[i].y, pos[i].z);
  iterationNum[i]++;
  return;
}

} // namespace KIRI

#endif /* _CUDA_PROTOSPHERE_SAMPLER_GPU_CUH_ */