/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-02-25 03:59:39
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-03-04 15:46:41
 * @FilePath:
 * \XProtoSphere\xprotosphere_cuda\include\kiri_pbs_cuda\particle\cuda_sph_bn_particles.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _CUDA_SPH_BN_PARTICLES_CUH_
#define _CUDA_SPH_BN_PARTICLES_CUH_

#pragma once

#include <kiri_pbs_cuda/particle/cuda_sph_particles.cuh>
#include <kiri_pbs_cuda/sampler/cuda_sampler_struct.cuh>

namespace KIRI {
class CudaSphBNParticles final : public CudaSphParticles {
public:
  explicit CudaSphBNParticles::CudaSphBNParticles(const uint numOfMaxParticles,
                                                  const GridInfo info,
                                                  const Vec_Float &sdfData)
      : CudaSphParticles(numOfMaxParticles),
        mCudaGridSize(CuCeilDiv(numOfMaxParticles, KIRI_CUBLOCKSIZE)),
        mIsBoundary(numOfMaxParticles), mPorosity(numOfMaxParticles),
        mTargetRadius(numOfMaxParticles), mOverlapRatio(numOfMaxParticles),
        mInfo(info),
        mSDFData(info.GridSize.x * info.GridSize.y * info.GridSize.z),
        mLastPos(numOfMaxParticles)

  {

    KIRI_CUCALL(cudaMemcpy(mSDFData.data(), &sdfData[0],
                           sizeof(float) * sdfData.size(),
                           cudaMemcpyHostToDevice));
  }

  explicit CudaSphBNParticles::CudaSphBNParticles(
      const Vec_Float3 &pos, const Vec_Float3 &col, const Vec_Float &rad,
      const Vec_Float &mass, const GridInfo info, const Vec_Float &sdfData)
      : CudaSphParticles(pos, mass, rad, col),
        mCudaGridSize(CuCeilDiv(pos.size(), KIRI_CUBLOCKSIZE)),
        mIsBoundary(pos.size()), mPorosity(pos.size()),
        mTargetRadius(pos.size()), mOverlapRatio(pos.size()), mInfo(info),
        mSDFData(info.GridSize.x * info.GridSize.y * info.GridSize.z),
        mLastPos(pos.size()) {

    KIRI_CUCALL(cudaMemcpy(mSDFData.data(), &sdfData[0],
                           sizeof(float) * sdfData.size(),
                           cudaMemcpyHostToDevice));

    KIRI_CUCALL(cudaMemcpy(mLastPos.data(), &pos[0],
                           sizeof(float3) * pos.size(),
                           cudaMemcpyHostToDevice));
  }

  CudaSphBNParticles(const CudaSphBNParticles &) = delete;
  CudaSphBNParticles &operator=(const CudaSphBNParticles &) = delete;

  inline GridInfo levelSetData() const { return mInfo; }

  inline float *sdfDataPtr() const { return mSDFData.data(); }
  inline float *porosityPtr() const { return mPorosity.data(); }
  inline float *overlapRatioPtr() const { return mOverlapRatio.data(); }
  inline bool *boundaryPtr() const { return mIsBoundary.data(); }
  inline float *targetRadiusPtr() const { return mTargetRadius.data(); }
  inline float3 *lastPosPtr() const { return mLastPos.data(); }

  virtual ~CudaSphBNParticles() noexcept {}

  virtual void Advect(const float dt) override;

protected:
  CudaArray<bool> mIsBoundary;
  CudaArray<float> mPorosity;
  CudaArray<float> mSDFData;
  CudaArray<float> mOverlapRatio;
  CudaArray<float> mTargetRadius;
  CudaArray<float3> mLastPos;

private:
  uint mCudaGridSize;

  GridInfo mInfo;
};

typedef SharedPtr<CudaSphBNParticles> CudaSphBNParticlesPtr;
} // namespace KIRI

#endif