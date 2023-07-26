/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-03-14 00:15:45
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-05-23 16:23:06
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\particle\cuda_sph_bn_particles_gpu.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_SPH_BN_PARTICLES_GPU_CUH_
#define _CUDA_SPH_BN_PARTICLES_GPU_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

#include <kiri_pbs_cuda/math/cuda_math_colormap.cuh>

namespace KIRI {
__global__ void BNAdvect_CUDA(float3 *pos, float3 *vel, float3 *tpos,
                              float3 *tvel, float3 *col, float *density,
                              float3 *normal, bool *isBoundary, const uint num,
                              const float dt, const float minDensity,
                              const float maxDensity, const float epsilon) {
  const uint i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (i >= num)
    return;

  col[i] =
      GetColor<float>((density[i] - minDensity) / (maxDensity - minDensity),
                      ColormapType::Jet)
          .Data();

  // isBoundary == within
  if (isBoundary[i]) {
    vel[i] = tvel[i];
    pos[i] = tpos[i];
  } else {
    vel[i] = tvel[i] - normal[i] * dot(normal[i], tvel[i]);
  }
  return;
}
} // namespace KIRI

#endif