/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-08 19:05:14
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-08-22 15:59:20
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\system\cuda_base_system.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_BASE_SYSTEM_CUH_
#define _CUDA_BASE_SYSTEM_CUH_

#pragma once
#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher.cuh>
#include <kiri_pbs_cuda/solver/cuda_base_solver.cuh>

namespace KIRI {
class CudaBaseSystem {
public:
  explicit CudaBaseSystem(CudaBaseSolverPtr &solver,
                          CudaGNSearcherPtr &searcher,
                          const size_t maxNumOfParticles);

  CudaBaseSystem(const CudaBaseSystem &) = delete;
  CudaBaseSystem &operator=(const CudaBaseSystem &) = delete;
  virtual ~CudaBaseSystem() noexcept {}

  float updateSystem(float timeIntervalInSeconds);
  void updateWorldSize(const float3 lowestPoint, const float3 highestPoint);

protected:
  CudaBaseSolverPtr mSolver;
  CudaGNSearcherPtr mSearcher;

  virtual void onUpdateSolver(float timeIntervalInSeconds) = 0;
};

typedef SharedPtr<CudaBaseSystem> CudaBaseSystemPtr;
} // namespace KIRI

#endif