/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-02-23 22:50:51
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-02-25 02:43:18
 * @FilePath:
 * \XProtoSphere\xprotosphere_cuda\src\kiri_pbs_cuda\solver\sampler\cuda_dem_ms_sampler.cu
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#include <kiri_pbs_cuda/solver/sampler/cuda_dem_ms_sampler.cuh>
#include <kiri_pbs_cuda/solver/sampler/cuda_dem_ms_sampler_gpu.cuh>

#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
namespace KIRI {

void CudaDEMMultiSizedSampler::ComputeMRDemLinearMomentum(
    CudaSphBNParticlesPtr &fluids, const float young, const float poisson,
    const float tanFrictionAngle, const CudaArray<size_t> &cellStart,
    const float3 lowestPoint, const float3 highestPoint,
    const float kernelRadius, const int3 gridSize) {

  curandState *devStates;
  dim3 d3Rnd(KIRI_RANDOM_SEEDS, 1, 1);
  KIRI_CUCALL(cudaMalloc(&devStates, KIRI_RANDOM_SEEDS * sizeof(curandState)));
  _SetUpRndGen_CUDA<<<1, d3Rnd>>>(devStates, time(NULL));
  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();

  _ComputeMRDemForces_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      fluids->posPtr(), fluids->lastPosPtr(), fluids->velPtr(),
      fluids->accPtr(), fluids->massPtr(), fluids->radiusPtr(),
      fluids->overlapRatioPtr(), young, poisson, tanFrictionAngle, 5.f,
      fluids->size(), lowestPoint, highestPoint, cellStart.data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize), devStates);
  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaDEMMultiSizedSampler::updateVelocities(CudaSphBNParticlesPtr &fluids,
                                                const float dt) {

  _UpdateSamplerVelocities_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      fluids->velPtr(), fluids->accPtr(), dt, fluids->size());
  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

float CudaDEMMultiSizedSampler::ComputeSubTimeStepsByCFL(
    CudaSphBNParticlesPtr &fluids) {

  auto vel_array_device = thrust::device_pointer_cast(fluids->velPtr());
  auto radius_array_device = thrust::device_pointer_cast(fluids->radiusPtr());
  float3 max_vel =
      *(thrust::max_element(vel_array_device, vel_array_device + fluids->size(),
                            ThrustHelper::CompareLengthCuda<float3>()));
  float min_radius = *(thrust::min_element(
      radius_array_device, radius_array_device + fluids->size()));
  float max_vel2 = lengthSquared(max_vel);

  float cfl_timestep = max_vel2 > Tiny<float>()
                           ? 0.1f * (2.f * min_radius / max_vel2)
                           : Huge<float>();

  return clamp(cfl_timestep, 1e-6f, 1.f / 30.f);
}

void CudaDEMMultiSizedSampler::Advect(CudaSphBNParticlesPtr &fluids,
                                      const float dt, const float3 lowestPoint,
                                      const float3 highestPoint) {
  size_t num = fluids->size();
  fluids->Advect(dt);

  thrust::fill(thrust::device, fluids->accPtr(), fluids->accPtr() + num,
               make_float3(0.f));
  thrust::fill(thrust::device, fluids->boundaryPtr(),
               fluids->boundaryPtr() + num, false);
  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaDEMMultiSizedSampler::identifyBoundaryParticles(
    CudaSphBNParticlesPtr &fluids, const float dt) {

  _IdentifyBoundaryParticles_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      fluids->boundaryPtr(), fluids->posPtr(), fluids->velPtr(),
      fluids->sdfDataPtr(), dt, fluids->size(),
      Mesh3SDF(fluids->levelSetData()));
  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaDEMMultiSizedSampler::moveParticles(CudaSphBNParticlesPtr &fluids,
                                             const float dt,
                                             const float boundaryRestitution,
                                             const bool renderOverlap) {

  auto min_overlap = 0.f;
  auto max_overlap = 1.f;

  _MoveDEMMSSamplerParticlesWithinBoundary_CUDA<<<mCudaGridSize,
                                                  KIRI_CUBLOCKSIZE>>>(
      fluids->posPtr(), fluids->lastPosPtr(), fluids->velPtr(),
      fluids->colorPtr(), fluids->overlapRatioPtr(), fluids->sdfDataPtr(), dt,
      boundaryRestitution, min_overlap, max_overlap, renderOverlap,
      fluids->size(), Mesh3SDF(fluids->levelSetData()));
  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaDEMMultiSizedSampler::computeOverlapRatio(
    CudaSphBNParticlesPtr &fluids, const CudaArray<size_t> &cellStart,
    const float3 lowestPoint, const float kernelRadius, const int3 gridSize) {

  _ComputeSamplerOverlapRatio_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      fluids->overlapRatioPtr(), fluids->posPtr(), fluids->radiusPtr(),
      fluids->size(), cellStart.data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize));

  if ((++mIter) % 100 == 0) {
    auto avg_overlap = thrust::reduce(
        thrust::device_ptr<float>(fluids->overlapRatioPtr()),
        thrust::device_ptr<float>(fluids->overlapRatioPtr() + fluids->size()));
    avg_overlap /= fluids->size();

    KIRI_LOG_INFO("DEM-Relaxation Iteration Num={0}; Average Overlap={1}",
                  mIter, avg_overlap);
  }

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

} // namespace KIRI
