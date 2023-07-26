/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-01-25 15:05:15
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-01-25 15:18:48
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\particle\cuda_particles.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _CUDA_PARTICLES_CUH_
#define _CUDA_PARTICLES_CUH_

#pragma once

#include <kiri_pbs_cuda/data/cuda_array.cuh>

namespace KIRI {
class CudaParticles {
public:
  explicit CudaParticles(const size_t numOfMaxParticles,
                         const bool renderField = true)
      : mPos(numOfMaxParticles), mParticle2Cell(numOfMaxParticles),
        mNumOfParticles(0), mNumOfMaxParticles(numOfMaxParticles),
        mRenderColorByField(renderField), mScalarFieldMin(0.f),
        mScalarFieldMax(2000.f), mFieldType(0) {}

  explicit CudaParticles(const Vec_Float3 &p, const bool renderField = true)
      : mPos(p.size()), mParticle2Cell(p.size()), mNumOfParticles(p.size()),
        mNumOfMaxParticles(p.size()), mRenderColorByField(renderField),
        mScalarFieldMin(0.f), mScalarFieldMax(2000.f), mFieldType(0) {
    if (!p.empty())
      KIRI_CUCALL(cudaMemcpy(mPos.data(), &p[0], sizeof(float3) * p.size(),
                             cudaMemcpyHostToDevice));
  }

  explicit CudaParticles(const size_t numOfMaxParticles, const Vec_Float3 &p,
                         const bool renderField = true)
      : mPos(numOfMaxParticles), mParticle2Cell(numOfMaxParticles),
        mNumOfParticles(p.size()), mNumOfMaxParticles(numOfMaxParticles),
        mRenderColorByField(renderField) {
    if (!p.empty())
      KIRI_CUCALL(cudaMemcpy(mPos.data(), &p[0], sizeof(float3) * p.size(),
                             cudaMemcpyHostToDevice));
  }

  CudaParticles(const CudaParticles &) = delete;
  CudaParticles &operator=(const CudaParticles &) = delete;

  virtual ~CudaParticles() noexcept {}

  inline size_t size() const { return mNumOfParticles; }
  inline size_t maxSize() const { return mNumOfMaxParticles; }
  inline float3 *posPtr() const { return mPos.data(); }
  inline size_t *particle2CellPtr() const { return mParticle2Cell.data(); }

  void SetSize(const size_t s) { mNumOfParticles = s; }
  void UpdateRenderFieldParams(int fieldType, bool asField, float minVal,
                               float maxVal) {
    mRenderColorByField = asField;
    mScalarFieldMin = minVal;
    mScalarFieldMax = maxVal;
    mFieldType = fieldType;
  }

protected:
  int mFieldType;
  bool mRenderColorByField;
  float mScalarFieldMin, mScalarFieldMax;

  size_t mNumOfParticles;
  size_t mNumOfMaxParticles;
  CudaArray<float3> mPos;
  CudaArray<size_t> mParticle2Cell;
};

typedef SharedPtr<CudaParticles> CudaParticlesPtr;
} // namespace KIRI

#endif