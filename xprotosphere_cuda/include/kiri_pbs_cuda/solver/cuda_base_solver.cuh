/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-20 11:12:44
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-01-24 16:06:31
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\solver\cuda_base_solver.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _CUDA_BASE_SOLVER_CUH_
#define _CUDA_BASE_SOLVER_CUH_

#pragma once

#include <kiri_pbs_cuda/data/cuda_boundary_params.h>
#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI {
class CudaBaseSolver {
public:
  explicit CudaBaseSolver(const size_t num)
      : mCudaGridSize(CuCeilDiv(num, KIRI_CUBLOCKSIZE)), mNumOfSubTimeSteps(1) {
  }

  virtual ~CudaBaseSolver() noexcept {}
  inline size_t GetSubTimeStepsNum() const { return mNumOfSubTimeSteps; }

protected:
  size_t mCudaGridSize;
  size_t mNumOfSubTimeSteps;
};

typedef SharedPtr<CudaBaseSolver> CudaBaseSolverPtr;
} // namespace KIRI

#endif