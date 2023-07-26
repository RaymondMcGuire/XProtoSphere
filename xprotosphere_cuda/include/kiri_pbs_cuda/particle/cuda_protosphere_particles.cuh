/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-11-29 19:50:10
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-12-07 15:37:53
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\particle\cuda_protosphere_particles.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _CUDA_PROTOSPHERE_PARTICLES_CUH_
#define _CUDA_PROTOSPHERE_PARTICLES_CUH_

#pragma once

#include <kiri_pbs_cuda/particle/cuda_particles.cuh>
#include <kiri_pbs_cuda/sampler/cuda_sampler_struct.cuh>
namespace KIRI {
class CudaProtoSphereParticles final : public CudaParticles {
public:
  explicit CudaProtoSphereParticles::CudaProtoSphereParticles(
      const uint numOfMaxParticles, const GridInfo info)
      : CudaParticles(numOfMaxParticles),
        mCudaGridSize(CuCeilDiv(numOfMaxParticles, KIRI_CUBLOCKSIZE)),
        mCol(numOfMaxParticles), mRadius(numOfMaxParticles),
        mTargetRadius(numOfMaxParticles), mInfo(info),
        mConvergence(numOfMaxParticles), mLearningRate(numOfMaxParticles),
        mIterationNum(numOfMaxParticles), mLastPos(numOfMaxParticles),
        mInitialPos(numOfMaxParticles), mCurrentRadius(numOfMaxParticles),
        mCurrentSDF(numOfMaxParticles), mCurrentClosestPoint(numOfMaxParticles),
        mSDFData(info.GridSize.x * info.GridSize.y * info.GridSize.z),
        mClosestPoint(info.GridSize.x * info.GridSize.y * info.GridSize.z) {}

  explicit CudaProtoSphereParticles::CudaProtoSphereParticles(
      const Vec_Float3 &p, const Vec_Float3 &col, const Vec_Float &rad,
      const Vec_Float &targetRadius, const GridInfo info,
      const Vec_Float &sdfData, const Vec_Float3 &closestPoint,
      const float maxRadius, const float minRadius)
      : CudaParticles(p), mCudaGridSize(CuCeilDiv(p.size(), KIRI_CUBLOCKSIZE)),
        mCol(col.size()), mRadius(rad.size()),
        mTargetRadius(targetRadius.size()), mInfo(info), mSDFCPUData(sdfData),
        mConvergence(p.size()), mIterationNum(p.size()), mLastPos(p.size()),
        mInitialPos(p.size()), mLearningRate(p.size()),
        mCurrentRadius(p.size()), mCurrentSDF(p.size()),
        mCurrentClosestPoint(p.size()), mSDFData(sdfData.size()),
        mClosestPoint(closestPoint.size()), mMaxRadius(maxRadius),
        mMinRadius(minRadius) {

    KIRI_CUCALL(cudaMemcpy(mLastPos.data(), &p[0], sizeof(float3) * p.size(),
                           cudaMemcpyHostToDevice));

    KIRI_CUCALL(cudaMemcpy(mInitialPos.data(), &p[0], sizeof(float3) * p.size(),
                           cudaMemcpyHostToDevice));

    KIRI_CUCALL(cudaMemcpy(mSDFData.data(), &sdfData[0],
                           sizeof(float) * sdfData.size(),
                           cudaMemcpyHostToDevice));

    KIRI_CUCALL(cudaMemcpy(mClosestPoint.data(), &closestPoint[0],
                           sizeof(float3) * closestPoint.size(),
                           cudaMemcpyHostToDevice));

    KIRI_CUCALL(cudaMemcpy(mRadius.data(), &rad[0], sizeof(float) * rad.size(),
                           cudaMemcpyHostToDevice));
    KIRI_CUCALL(cudaMemcpy(mTargetRadius.data(), &targetRadius[0],
                           sizeof(float) * targetRadius.size(),
                           cudaMemcpyHostToDevice));
    KIRI_CUCALL(cudaMemcpy(mCol.data(), &col[0], sizeof(float3) * col.size(),
                           cudaMemcpyHostToDevice));
  }

  CudaProtoSphereParticles(const CudaProtoSphereParticles &) = delete;
  CudaProtoSphereParticles &
  operator=(const CudaProtoSphereParticles &) = delete;

  inline GridInfo levelSetData() const { return mInfo; }
  inline Vec_Float sdfCPUData() const { return mSDFCPUData; }

  inline float3 *colorPtr() const { return mCol.data(); }
  inline float *radiusPtr() const { return mRadius.data(); }
  inline float *targetRadiusPtr() const { return mTargetRadius.data(); }
  inline float *currentRadiusPtr() const { return mCurrentRadius.data(); }

  inline float *GetCurrentSDFPtr() const { return mCurrentSDF.data(); }
  inline float3 *GetCurrentClosestPointPtr() const {
    return mCurrentClosestPoint.data();
  }

  inline float *sdfDataPtr() const { return mSDFData.data(); }
  inline float3 *closestPointPtr() const { return mClosestPoint.data(); }

  inline float3 *lastPosPtr() const { return mLastPos.data(); }
  inline float3 *GetInitialPosPtr() const { return mInitialPos.data(); }

  inline float *learningRatePtr() const { return mLearningRate.data(); }
  inline int *convergencePtr() const { return mConvergence.data(); }
  inline uint *iterationNumPtr() const { return mIterationNum.data(); }

  inline float maxRadius() const { return mMaxRadius; }
  inline float minRadius() const { return mMinRadius; }

  virtual ~CudaProtoSphereParticles() noexcept {}

protected:
  CudaArray<float3> mCol;
  CudaArray<float3> mLastPos;
  CudaArray<float3> mInitialPos;
  CudaArray<float> mRadius;
  CudaArray<float> mCurrentRadius;
  CudaArray<float> mTargetRadius;

  CudaArray<float> mCurrentSDF;
  CudaArray<float3> mCurrentClosestPoint;

  CudaArray<float> mSDFData;
  CudaArray<float3> mClosestPoint;

  CudaArray<int> mConvergence;
  CudaArray<uint> mIterationNum;
  CudaArray<float> mLearningRate;

private:
  uint mCudaGridSize;
  GridInfo mInfo;
  Vec_Float mSDFCPUData;
  float mMaxRadius = 0.f, mMinRadius = 0.f;
};

typedef SharedPtr<CudaProtoSphereParticles> CudaProtoSphereParticlesPtr;
} // namespace KIRI

#endif