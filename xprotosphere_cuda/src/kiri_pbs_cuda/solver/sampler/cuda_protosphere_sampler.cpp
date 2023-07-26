/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-05-01 20:29:07
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-05-05 12:05:19
 * @FilePath:
 * \XProtoSphere\xprotosphere_cuda\src\kiri_pbs_cuda\solver\sampler\cuda_protosphere_sampler.cpp
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#include <kiri_pbs_cuda/solver/sampler/cuda_protosphere_sampler.cuh>

namespace KIRI {
void CudaProtoSphereSampler::UpdateSolver(
    CudaProtoSphereParticlesPtr &particles,
    const CudaSphBNParticlesPtr &inserted, const CudaArray<size_t> &cellStart,
    const CudaProtoSphereParams &params, const CudaBoundaryParams &bparams,
    const int insertStep) {
  if (insertStep == 0)
    UpdateSDFWithOffset(particles, inserted, cellStart, bparams.lowest_point,
                        bparams.kernel_radius, bparams.grid_size);
  else
    UpdateSDF(particles, inserted, cellStart, bparams.lowest_point,
              bparams.kernel_radius, bparams.grid_size);

  AdvectProtoSphereParticles(particles, params.error_rate,
                             params.max_iteration_num, params.decay);

  CheckConvergence(particles);
}

} // namespace KIRI