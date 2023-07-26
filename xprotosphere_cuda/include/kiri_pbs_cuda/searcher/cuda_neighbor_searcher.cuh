/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-02-23 22:50:51
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-02-23 23:29:13
 * @FilePath:
 * \XProtoSphere\xprotosphere_cuda\include\kiri_pbs_cuda\searcher\cuda_neighbor_searcher.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _CUDA_NEIGHBOR_SEARCHER_CUH_
#define _CUDA_NEIGHBOR_SEARCHER_CUH_

#pragma once

#include <kiri_pbs_cuda/particle/cuda_sph_bn_particles.cuh>
namespace KIRI {

class CudaGNBaseSearcher {
public:
  explicit CudaGNBaseSearcher(const float3 lowestPoint,
                              const float3 highestPoint,
                              const size_t numOfParticles,
                              const float cellSize);

  CudaGNBaseSearcher(const CudaGNBaseSearcher &) = delete;
  CudaGNBaseSearcher &operator=(const CudaGNBaseSearcher &) = delete;

  virtual ~CudaGNBaseSearcher() noexcept {}

  float3 lowestPoint() const { return mLowestPoint; }
  float3 highestPoint() const { return mHighestPoint; }
  float cellSize() const { return mCellSize; }
  int3 gridSize() const { return mGridSize; }

  size_t *cellStartPtr() const { return mCellStart.data(); }
  const CudaArray<size_t> &cellStart() const { return mCellStart; }

  size_t *gridIdxArrayPtr() const { return mGridIdxArray.data(); }
  const CudaArray<size_t> &gridIdxArray() const { return mGridIdxArray; }

  void BuildGNSearcher(const CudaParticlesPtr &particles);

  void updateWorldSize(const float3 lowestPoint, const float3 highestPoint);

protected:
  size_t mCudaGridSize;
  int3 mGridSize;
  float mCellSize;
  float3 mLowestPoint;
  float3 mHighestPoint;
  size_t mNumOfGridCells;
  size_t mMaxNumOfParticles;

  CudaArray<size_t> mGridIdxArray;
  CudaArray<size_t> mCellStart;

  virtual void sortData(const CudaParticlesPtr &particles) = 0;
};

class CudaGNSearcher final : public CudaGNBaseSearcher {
public:
  explicit CudaGNSearcher(const float3 lowestPoint, const float3 highestPoint,
                          const size_t num, const float cellSize);

  CudaGNSearcher(const CudaGNSearcher &) = delete;
  CudaGNSearcher &operator=(const CudaGNSearcher &) = delete;

  virtual ~CudaGNSearcher() noexcept {}

protected:
  virtual void sortData(const CudaParticlesPtr &particles) override final;
};

typedef SharedPtr<CudaGNBaseSearcher> CudaGNBaseSearcherPtr;
typedef SharedPtr<CudaGNSearcher> CudaGNSearcherPtr;

} // namespace KIRI

#endif /* CudaNeighborSearcher */