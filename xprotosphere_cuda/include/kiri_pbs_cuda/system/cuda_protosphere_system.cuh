/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-05-01 20:29:06
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-06-23 16:56:21
 * @FilePath:
 * \XProtoSphere\xprotosphere_cuda\include\kiri_pbs_cuda\system\cuda_protosphere_system.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _CUDA_PROTOSPHERE_SYSTEM_CUH_
#define _CUDA_PROTOSPHERE_SYSTEM_CUH_

#pragma once
#include <kiri_pbs_cuda/particle/cuda_sph_bn_particles.cuh>
#include <kiri_pbs_cuda/searcher/cuda_neighbor_searcher.cuh>
#include <kiri_pbs_cuda/solver/sampler/cuda_dem_ms_sampler.cuh>
#include <kiri_pbs_cuda/solver/sampler/cuda_protosphere_sampler.cuh>
namespace KIRI {

class CudaProtoSphereSystem {
public:
  CudaProtoSphereSystem(CudaProtoSphereParticlesPtr &particles,
                        CudaProtoSphereSamplerPtr &solver,
                        const std::vector<float> radiusRange,
                        const std::vector<float> radiusRangeProb,
                        float boundaryVolume, float acceptOverlapRatio,
                        float halfGridSize, int maxProtoSphereIteration,
                        bool enableDEMRelax, int maxRelaxStepNum);

  CudaProtoSphereSystem(const CudaProtoSphereSystem &) = delete;
  CudaProtoSphereSystem &operator=(const CudaProtoSphereSystem &) = delete;

  float UpdateSystem();

  inline size_t GetIdealTotalNum() const { return mIdealTotalNum; }

  inline size_t GetInsertedSize() const { return mInsertedParticles->size(); }
  inline float3 *GetInsertedPositions() const {
    return mInsertedParticles->posPtr();
  }
  inline float *GetInsertedRadius() const {
    return mInsertedParticles->radiusPtr();
  }
  inline float *GetInsertedTargetRadius() const {
    return mInsertedParticles->targetRadiusPtr();
  }
  inline float3 *GetInsertedColors() const {
    return mInsertedParticles->colorPtr();
  }

  inline int GetInsertStepNum() const { return mInsertedStep; }
  inline int GetRelaxStepNum() const { return mRelaxStep; }

  inline bool GetConvergenceStatus() const { return mConvergence; }
  inline bool GetDEMRelaxationEnable() const { return mEnableDEMRelax; }

  inline int GetMaxProtoSphereIterationNum() const {
    return mMaxProtoSphereIterationNum;
  }
  inline XProtoSphereSystemStatus GetSystemStatus() const {
    return mSystemStatus;
  }
  ~CudaProtoSphereSystem();

private:
  size_t mCudaGridSize;
  bool mConvergence = false;
  size_t mNumOfSampledPoints = 0;
  float mAcceptOverlapRatio = 0.2f;
  int mMaxRelaxStepNum = 1000;
  float mHalfGridSize = 0.01f;
  bool mEnableDEMRelax = false;

  int mMaxProtoSphereIterationNum = 5;

  XProtoSphereSystemStatus mSystemStatus;

  CudaProtoSphereParticlesPtr mParticles;
  CudaSphBNParticlesPtr mInsertedParticles;

  CudaProtoSphereSamplerPtr mSolver;
  CudaDEMMultiSizedSamplerPtr mDEMMSSolver;

  CudaGNSearcherPtr mSearcher;

  std::vector<float> mCurrentRadiusRange, mCurrentRadiusRangeProb;
  std::vector<float> mPreDefinedRadiusRange, mPreDefinedRadiusRangeProb;

  int mInsertedStep = 0;
  int mRelaxStep = 0;
  size_t mIdealTotalNum = 0;
  size_t mMaxSamplerNum = 0;
  float mBoundaryVolume = 0.f;

  // statistic data
  float mRMSPE = 0.f;
  std::vector<int> mRadiusRangeCounter, mRemainSamples;

  void InsertParticles();
  void EstimateIdealTotalNum();
  void RunProtoSphereSampler();
  void RunDEMMSSampler();
};

typedef SharedPtr<CudaProtoSphereSystem> CudaProtoSphereSystemPtr;
} // namespace KIRI

#endif