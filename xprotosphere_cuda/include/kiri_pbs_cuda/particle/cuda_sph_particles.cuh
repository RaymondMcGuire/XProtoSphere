/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-01-30 11:49:22
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-01-30 14:49:26
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\particle\cuda_sph_particles.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _CUDA_SPH_PARTICLES_CUH_
#define _CUDA_SPH_PARTICLES_CUH_

#pragma once

#include <kiri_pbs_cuda/particle/cuda_particles.cuh>
namespace KIRI {
class CudaSphParticles : public CudaParticles {
public:
  explicit CudaSphParticles::CudaSphParticles(const size_t numOfMaxParticles)
      : CudaParticles(numOfMaxParticles), mVel(numOfMaxParticles),
        mAcc(numOfMaxParticles), mCol(numOfMaxParticles),
        mPressure(numOfMaxParticles), mDensity(numOfMaxParticles),
        mMass(numOfMaxParticles), mRadius(numOfMaxParticles),
        mNormal(numOfMaxParticles), mCorrectorR(numOfMaxParticles),
        mCorrectorDiff(numOfMaxParticles), mCorrectorAcr(numOfMaxParticles),
        mNablaCorrectorR(numOfMaxParticles) {}

  explicit CudaSphParticles::CudaSphParticles(const Vec_Float3 &p,
                                              const Vec_Float &mass,
                                              const Vec_Float &rad,
                                              const Vec_Float3 &col)
      : CudaParticles(p), mVel(p.size()), mAcc(p.size()), mCol(p.size()),
        mPressure(p.size()), mDensity(p.size()), mMass(p.size()),
        mRadius(p.size()), mNormal(p.size()), mCorrectorR(p.size()),
        mCorrectorDiff(p.size()), mCorrectorAcr(p.size()),
        mNablaCorrectorR(p.size()) {

    if (!p.empty()) {
      KIRI_CUCALL(cudaMemcpy(mMass.data(), &mass[0],
                             sizeof(float) * mass.size(),
                             cudaMemcpyHostToDevice));
      KIRI_CUCALL(cudaMemcpy(mRadius.data(), &rad[0],
                             sizeof(float) * rad.size(),
                             cudaMemcpyHostToDevice));
      KIRI_CUCALL(cudaMemcpy(mCol.data(), &col[0], sizeof(float3) * col.size(),
                             cudaMemcpyHostToDevice));
    }
  }

  explicit CudaSphParticles::CudaSphParticles(const size_t numOfMaxParticles,
                                              const Vec_Float3 &p,
                                              const Vec_Float &mass,
                                              const Vec_Float &rad,
                                              const Vec_Float3 &col)
      : CudaParticles(numOfMaxParticles, p), mVel(numOfMaxParticles),
        mAcc(numOfMaxParticles), mCol(numOfMaxParticles),
        mPressure(numOfMaxParticles), mDensity(numOfMaxParticles),
        mMass(numOfMaxParticles), mRadius(numOfMaxParticles),
        mNormal(numOfMaxParticles), mCorrectorR(numOfMaxParticles),
        mCorrectorDiff(numOfMaxParticles), mCorrectorAcr(numOfMaxParticles),
        mNablaCorrectorR(numOfMaxParticles) {

    if (!p.empty()) {
      KIRI_CUCALL(cudaMemcpy(mMass.data(), &mass[0],
                             sizeof(float) * mass.size(),
                             cudaMemcpyHostToDevice));
      KIRI_CUCALL(cudaMemcpy(mRadius.data(), &rad[0],
                             sizeof(float) * rad.size(),
                             cudaMemcpyHostToDevice));
      KIRI_CUCALL(cudaMemcpy(mCol.data(), &col[0], sizeof(float3) * col.size(),
                             cudaMemcpyHostToDevice));
    }
  }

  CudaSphParticles(const CudaSphParticles &) = delete;
  CudaSphParticles &operator=(const CudaSphParticles &) = delete;

  float3 *velPtr() const { return mVel.data(); }
  float3 *accPtr() const { return mAcc.data(); }
  float3 *colorPtr() const { return mCol.data(); }
  float *pressurePtr() const { return mPressure.data(); }
  float *densityPtr() const { return mDensity.data(); }
  float *massPtr() const { return mMass.data(); }
  float *radiusPtr() const { return mRadius.data(); }
  float3 *normalPtr() const { return mNormal.data(); }

  float *correctorAcrPtr() const { return mCorrectorAcr.data(); }
  float *correctorRPtr() const { return mCorrectorR.data(); }
  float *correctorDiffPtr() const { return mCorrectorDiff.data(); }
  float3 *nablaCorrectorRPtr() const { return mNablaCorrectorR.data(); }

  virtual ~CudaSphParticles() noexcept {}

  virtual void Advect(const float dt);

  void appendParticles(Vec_Float3 pos, Vec_Float radius, float3 col, float3 vel,
                       float mass);

protected:
  CudaArray<float3> mVel;
  CudaArray<float3> mAcc;
  CudaArray<float3> mCol;
  CudaArray<float> mPressure;
  CudaArray<float> mDensity;
  CudaArray<float> mMass;
  CudaArray<float> mRadius;
  CudaArray<float3> mNormal;

  CudaArray<float> mCorrectorAcr;
  CudaArray<float> mCorrectorR;
  CudaArray<float> mCorrectorDiff;
  CudaArray<float3> mNablaCorrectorR;
};

typedef SharedPtr<CudaSphParticles> CudaSphParticlesPtr;
} // namespace KIRI

#endif