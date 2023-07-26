/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-02-25 14:19:22
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-04-16 16:15:51
 * @FilePath:
 * \XProtoSphere\xprotosphere_cuda\src\kiri_pbs_cuda\solver\sampler\cuda_dem_ms_sampler.cpp
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#include <kiri_pbs_cuda/solver/sampler/cuda_dem_ms_sampler.cuh>

namespace KIRI {
void CudaDEMMultiSizedSampler::UpdateSolver(CudaSphBNParticlesPtr &fluids,
                                            const CudaArray<size_t> &cellStart,
                                            float timeIntervalInSeconds,
                                            CudaBoundaryParams bparams) {

  computeOverlapRatio(fluids, cellStart, bparams.lowest_point,
                      bparams.kernel_radius, bparams.grid_size);

  ComputeMRDemLinearMomentum(fluids, 1e5f, 0.3f, std::tanf(0.5f), cellStart,
                             bparams.lowest_point, bparams.highest_point,
                             bparams.kernel_radius, bparams.grid_size);

  updateVelocities(fluids, mDt);

  identifyBoundaryParticles(fluids, mDt);

  moveParticles(fluids, mDt, 0.5f, false);

  Advect(fluids, mDt, bparams.lowest_point, bparams.highest_point);
}

} // namespace KIRI