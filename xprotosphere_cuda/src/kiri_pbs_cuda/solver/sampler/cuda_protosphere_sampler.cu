/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-02-20 15:24:04
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-02-21 16:45:27
 * @FilePath:
 * \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\solver\sampler\cuda_protosphere_sampler.cu
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#include <kiri_pbs_cuda/solver/sampler/cuda_protosphere_sampler.cuh>
#include <kiri_pbs_cuda/solver/sampler/cuda_protosphere_sampler_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>

#include <thrust/device_ptr.h>
namespace KIRI {

void CudaProtoSphereSampler::SphereMoveRandom(
    CudaProtoSphereParticlesPtr &particles, const float halfGridSize) {

  curandState *devStates;
  dim3 d3Rnd(KIRI_RANDOM_SEEDS, 1, 1);
  KIRI_CUCALL(cudaMalloc(&devStates, KIRI_RANDOM_SEEDS * sizeof(curandState)));
  _SetUpRndGen_CUDA<<<1, d3Rnd>>>(devStates, time(NULL));
  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();

  _SphereMoveRandom_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      particles->posPtr(), particles->lastPosPtr(), particles->size(),
      halfGridSize, devStates);
  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();

  KIRI_CUCALL(cudaFree(devStates));
}

void CudaProtoSphereSampler::UpdateSDF(CudaProtoSphereParticlesPtr &particles,
                                       const CudaSphBNParticlesPtr &inserted,
                                       const CudaArray<size_t> &cellStart,
                                       const float3 lowestPoint,
                                       const float kernelRadius,
                                       const int3 gridSize) {
  _UpdateSDF_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      particles->GetCurrentSDFPtr(), particles->GetCurrentClosestPointPtr(),
      particles->posPtr(), particles->convergencePtr(), particles->sdfDataPtr(),
      particles->closestPointPtr(), inserted->posPtr(), inserted->radiusPtr(),
      particles->size(), cellStart.data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize),
      Mesh3SDF(particles->levelSetData()));
  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

void CudaProtoSphereSampler::UpdateSDFWithOffset(
    CudaProtoSphereParticlesPtr &particles,
    const CudaSphBNParticlesPtr &inserted, const CudaArray<size_t> &cellStart,
    const float3 lowestPoint, const float kernelRadius, const int3 gridSize) {

  curandState *devStates;
  dim3 d3Rnd(KIRI_RANDOM_SEEDS, 1, 1);
  KIRI_CUCALL(cudaMalloc(&devStates, KIRI_RANDOM_SEEDS * sizeof(curandState)));
  _SetUpRndGen_CUDA<<<1, d3Rnd>>>(devStates, time(NULL));
  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();

  auto sdf_data_device = thrust::device_pointer_cast(particles->sdfDataPtr());
  auto sdf_data_size = particles->levelSetData().GridSize.x *
                       particles->levelSetData().GridSize.y *
                       particles->levelSetData().GridSize.z;
  float max_sdf_val =
      *(thrust::max_element(sdf_data_device, sdf_data_device + sdf_data_size));

  _UpdateSDFWithOffset_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      particles->GetCurrentSDFPtr(), particles->GetCurrentClosestPointPtr(),
      particles->posPtr(), particles->convergencePtr(), particles->sdfDataPtr(),
      particles->closestPointPtr(), inserted->posPtr(), inserted->radiusPtr(),
      particles->size(), cellStart.data(), gridSize,
      ThrustHelper::Pos2GridXYZ<float3>(lowestPoint, kernelRadius, gridSize),
      ThrustHelper::GridXYZ2GridHash(gridSize),
      Mesh3SDF(particles->levelSetData()), max_sdf_val, devStates);
  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();

  KIRI_CUCALL(cudaFree(devStates));
}

void CudaProtoSphereSampler::AdvectProtoSphereParticles(
    CudaProtoSphereParticlesPtr &particles, const float errorRate,
    const uint maxIterationNum, const float decay) {

  _AdvectProtoSphereParticles_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      particles->posPtr(), particles->lastPosPtr(),
      particles->currentRadiusPtr(), particles->learningRatePtr(),
      particles->convergencePtr(), particles->iterationNumPtr(),
      particles->targetRadiusPtr(), particles->GetCurrentSDFPtr(),
      particles->GetCurrentClosestPointPtr(), particles->size(), errorRate,
      maxIterationNum, decay);
  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

bool CudaProtoSphereSampler::CheckConvergence(
    const CudaProtoSphereParticlesPtr &particles) {

  auto convergenced_num = thrust::reduce(
      thrust::device_ptr<int>(particles->convergencePtr()),
      thrust::device_ptr<int>(particles->convergencePtr() + particles->size()));

  // printf("convergence num=%d \n", convergenced_num);
  if (convergenced_num == particles->size())
    return true;
  return false;
}

} // namespace KIRI
