/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-02-23 22:50:51
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-02-23 23:30:16
 * @FilePath:
 * \XProtoSphere\xprotosphere_cuda\src\kiri_pbs_cuda\searcher\cuda_neighbor_searcher.cu
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher.cuh>
#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher_gpu.cuh>
#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>

#include <thrust/sort.h>

namespace KIRI {

CudaGNBaseSearcher::CudaGNBaseSearcher(const float3 lowestPoint,
                                       const float3 highestPoint,
                                       const size_t maxNumOfParticles,
                                       const float cellSize)
    : mLowestPoint(lowestPoint), mHighestPoint(highestPoint),
      mCellSize(cellSize),
      mGridSize(make_int3((highestPoint - lowestPoint) / cellSize)),
      mNumOfGridCells(mGridSize.x * mGridSize.y * mGridSize.z + 1),
      mCellStart(mNumOfGridCells), mMaxNumOfParticles(maxNumOfParticles),
      mGridIdxArray(max(mNumOfGridCells, maxNumOfParticles)),
      mCudaGridSize(CuCeilDiv(maxNumOfParticles, KIRI_CUBLOCKSIZE)) {}

void CudaGNBaseSearcher::BuildGNSearcher(const CudaParticlesPtr &particles) {
  thrust::transform(
      thrust::device, particles->posPtr(),
      particles->posPtr() + particles->size(), particles->particle2CellPtr(),
      ThrustHelper::Pos2GridHash<float3>(mLowestPoint, mCellSize, mGridSize));

  this->sortData(particles);

  thrust::fill(thrust::device, mCellStart.data(),
               mCellStart.data() + mNumOfGridCells, 0);
  _CountingInCell_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
      mCellStart.data(), particles->particle2CellPtr(), particles->size());
  thrust::exclusive_scan(thrust::device, mCellStart.data(),
                         mCellStart.data() + mNumOfGridCells,
                         mCellStart.data());

  cudaDeviceSynchronize();
  KIRI_CUKERNAL();
}

void CudaGNBaseSearcher::updateWorldSize(const float3 lowestPoint,
                                         const float3 highestPoint) {
  mLowestPoint = lowestPoint;
  mHighestPoint = highestPoint;
  mGridSize = make_int3((highestPoint - lowestPoint) / mCellSize);
  mNumOfGridCells = mGridSize.x * mGridSize.y * mGridSize.z + 1;

  mCellStart.resize(mNumOfGridCells);
  mGridIdxArray.resize(max(mNumOfGridCells, mMaxNumOfParticles));
}

CudaGNSearcher::CudaGNSearcher(const float3 lowestPoint,
                               const float3 highestPoint, const size_t num,
                               const float cellSize)
    : CudaGNBaseSearcher(lowestPoint, highestPoint, num, cellSize) {}

void CudaGNSearcher::sortData(const CudaParticlesPtr &particles) {

  auto particle_size = particles->size();

  auto fluids = std::dynamic_pointer_cast<CudaSphBNParticles>(particles);

  KIRI_CUCALL(cudaMemcpy(mGridIdxArray.data(), fluids->particle2CellPtr(),
                         sizeof(size_t) * particle_size,
                         cudaMemcpyDeviceToDevice));
  thrust::sort_by_key(
      thrust::device, mGridIdxArray.data(),
      mGridIdxArray.data() + particle_size,
      thrust::make_zip_iterator(thrust::make_tuple(
          fluids->posPtr(), fluids->velPtr(), fluids->colorPtr(),
          fluids->massPtr(), fluids->radiusPtr(), fluids->targetRadiusPtr(),
          fluids->lastPosPtr())));

  cudaDeviceSynchronize();
  KIRI_CUKERNAL();
}

} // namespace KIRI
