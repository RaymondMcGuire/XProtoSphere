/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-05-01 20:29:07
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-05-08 21:48:18
 * @FilePath:
 * \XProtoSphere\xprotosphere_cuda\src\kiri_pbs_cuda\system\cuda_protosphere_system.cpp
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#include <kiri_pbs_cuda/system/cuda_protosphere_system.cuh>

namespace KIRI {

void CudaProtoSphereSystem::EstimateIdealTotalNum() {
  auto denominator = 0.f;
  for (auto ri = 0; ri < mPreDefinedRadiusRange.size() - 1; ri++) {
    auto rj = ri + 1;
    auto avg_radius =
        0.5f * (mPreDefinedRadiusRange[ri] + mPreDefinedRadiusRange[rj]);
    denominator += 4.f / 3.f * avg_radius * avg_radius * avg_radius * KIRI_PI *
                   mPreDefinedRadiusRangeProb[ri];
  }
  mIdealTotalNum = mBoundaryVolume / denominator;
  mIdealTotalNum /= 2.f;

  mMaxSamplerNum = static_cast<size_t>(2000000);

  KIRI_LOG_DEBUG("Maximum Samples Number={0}; mIdealTotalNum={1}",
                 mMaxSamplerNum, mIdealTotalNum);

  mRemainSamples.resize(mPreDefinedRadiusRangeProb.size(), 0);

  for (auto i = 0; i < mPreDefinedRadiusRangeProb.size(); i++) {
    mRemainSamples[i] = int(mIdealTotalNum * mPreDefinedRadiusRangeProb[i]);
  }
}

void CudaProtoSphereSystem::RunProtoSphereSampler() {
  if (mSystemStatus != XPROTOTYPE_SEARCHING)
    return;

  mSearcher->BuildGNSearcher(mInsertedParticles);
  mSolver->UpdateSolver(mParticles, mInsertedParticles, mSearcher->cellStart(),
                        CUDA_PROTOSPHERE_PARAMS, CUDA_BOUNDARY_PARAMS,
                        mInsertedStep);
  mConvergence = mSolver->CheckConvergence(mParticles);
  InsertParticles();
}

void CudaProtoSphereSystem::RunDEMMSSampler()

{
  if (mRelaxStep > mMaxRelaxStepNum) {
    mSystemStatus = DEM_RELAXATION_FINISH;
    return;
  }

  mSearcher->BuildGNSearcher(mInsertedParticles);
  mDEMMSSolver->UpdateSolver(mInsertedParticles, mSearcher->cellStart(),
                             CUDA_PROTOSPHERE_PARAMS.relax_dt,
                             CUDA_BOUNDARY_PARAMS);

  mRelaxStep++;
}

} // namespace KIRI