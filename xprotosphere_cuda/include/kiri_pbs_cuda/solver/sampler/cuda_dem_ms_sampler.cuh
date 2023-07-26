/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-02-20 15:24:04
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-02-23 14:33:13
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\sampler\cuda_dem_ms_sampler.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _CUDA_DEM_MS_SAMPLER_CUH_
#define _CUDA_DEM_MS_SAMPLER_CUH_

#pragma once

#include <kiri_pbs_cuda/solver/cuda_base_solver.cuh>

#include <kiri_pbs_cuda/data/cuda_boundary_params.h>

#include <kiri_pbs_cuda/kernel/cuda_sph_kernel.cuh>
#include <kiri_pbs_cuda/particle/cuda_sph_bn_particles.cuh>
#include <kiri_pbs_cuda/sdf/cuda_basic_sdf.cuh>

namespace KIRI {
class CudaDEMMultiSizedSampler : public CudaBaseSolver {
public:
  virtual void UpdateSolver(CudaSphBNParticlesPtr &fluids,
                            const CudaArray<size_t> &cellStart,
                            float timeIntervalInSeconds,
                            CudaBoundaryParams bparams);

  explicit CudaDEMMultiSizedSampler(const size_t num) : CudaBaseSolver(num) {}

  virtual ~CudaDEMMultiSizedSampler() noexcept {}

protected:
  float ComputeSubTimeStepsByCFL(CudaSphBNParticlesPtr &fluids);

  void Advect(CudaSphBNParticlesPtr &fluids, const float dt,
              const float3 lowestPoint, const float3 highestPoint);

  void identifyBoundaryParticles(CudaSphBNParticlesPtr &fluids, const float dt);

  void computeOverlapRatio(CudaSphBNParticlesPtr &fluids,
                           const CudaArray<size_t> &cellStart,
                           const float3 lowestPoint, const float kernelRadius,
                           const int3 gridSize);

  void moveParticles(CudaSphBNParticlesPtr &fluids, const float dt,
                     const float boundaryRestitution, const bool renderOverlap);

  void updateVelocities(CudaSphBNParticlesPtr &fluids, const float dt);

  void ComputeMRDemLinearMomentum(
      CudaSphBNParticlesPtr &fluids, const float young, const float poisson,
      const float tanFrictionAngle, const CudaArray<size_t> &cellStart,
      const float3 lowestPoint, const float3 highestPoint,
      const float kernelRadius, const int3 gridSize);

private:
  float mDt = 1e-3f;
  int mIter = 0;
};

typedef SharedPtr<CudaDEMMultiSizedSampler> CudaDEMMultiSizedSamplerPtr;
} // namespace KIRI

#endif /* _CUDA_DEM_MS_SAMPLER_CUH_ */