/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-02-23 22:50:51
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-05-05 12:05:26
 * @FilePath:
 * \XProtoSphere\xprotosphere_cuda\include\kiri_pbs_cuda\solver\sampler\cuda_protosphere_sampler.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-12-14 19:16:17
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-02-21 16:48:39
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\sampler\cuda_protosphere_sampler.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _CUDA_PROTOSPHERE_SAMPLER_CUH_
#define _CUDA_PROTOSPHERE_SAMPLER_CUH_

#pragma once

#include <kiri_pbs_cuda/data/cuda_sampler_params.h>
#include <kiri_pbs_cuda/particle/cuda_protosphere_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_sph_bn_particles.cuh>
#include <kiri_pbs_cuda/sdf/cuda_basic_sdf.cuh>
#include <kiri_pbs_cuda/solver/cuda_base_solver.cuh>

namespace KIRI {
class CudaProtoSphereSampler : public CudaBaseSolver {
public:
  virtual void UpdateSolver(CudaProtoSphereParticlesPtr &particles,
                            const CudaSphBNParticlesPtr &inserted,
                            const CudaArray<size_t> &cellStart,
                            const CudaProtoSphereParams &params,
                            const CudaBoundaryParams &bparams,
                            const int insertStep);

  explicit CudaProtoSphereSampler(const size_t num) : CudaBaseSolver(num) {}

  virtual ~CudaProtoSphereSampler() noexcept {}

  bool CheckConvergence(const CudaProtoSphereParticlesPtr &particles);

  void SphereMoveRandom(CudaProtoSphereParticlesPtr &particles,
                        const float halfGridSize);

protected:
  void UpdateSDF(CudaProtoSphereParticlesPtr &particles,
                 const CudaSphBNParticlesPtr &inserted,
                 const CudaArray<size_t> &cellStart, const float3 lowestPoint,
                 const float kernelRadius, const int3 gridSize);

  void UpdateSDFWithOffset(CudaProtoSphereParticlesPtr &particles,
                           const CudaSphBNParticlesPtr &inserted,
                           const CudaArray<size_t> &cellStart,
                           const float3 lowestPoint, const float kernelRadius,
                           const int3 gridSize);

  void AdvectProtoSphereParticles(CudaProtoSphereParticlesPtr &particles,
                                  const float errorRate,
                                  const uint maxIterationNum,
                                  const float decay);
};

typedef SharedPtr<CudaProtoSphereSampler> CudaProtoSphereSamplerPtr;
} // namespace KIRI

#endif /* _CUDA_PROTOSPHERE_SAMPLER_CUH_ */