/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-01-25 15:05:15
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-01-25 15:19:38
 * @FilePath: \Kiri\KiriPBSCuda\src\kiri_pbs_cuda\particle\cuda_sph_particles.cu
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#include <kiri_pbs_cuda/particle/cuda_sph_particles.cuh>
#include <kiri_pbs_cuda/solver/sph/cuda_sph_solver_common_gpu.cuh>
namespace KIRI {

void CudaSphParticles::Advect(const float dt) {

  // printf("Advect=%zd \n", mPos.Length());
  thrust::transform(
      thrust::device, mVel.data(), mVel.data() + size(), mAcc.data(),
      mVel.data(), [dt] __host__ __device__(const float3 &lv, const float3 &a) {
        return lv + dt * a;
      });

  thrust::transform(
      thrust::device, mPos.data(), mPos.data() + size(), mVel.data(),
      mPos.data(), [dt] __host__ __device__(const float3 &lp, const float3 &v) {
        return lp + dt * v;
      });

  if (mRenderColorByField)
    switch (mFieldType) {
    case 0:
      _ComputeFieldColor_CUDA<<<CuCeilDiv(mNumOfMaxParticles, KIRI_CUBLOCKSIZE),
                                KIRI_CUBLOCKSIZE>>>(
          mCol.data(), mVel.data(), mScalarFieldMin, mScalarFieldMax,
          mNumOfParticles, ColormapType::BlueWhiteRed);
      break;
    case 1:
      _ComputeFieldColor_CUDA<<<CuCeilDiv(mNumOfMaxParticles, KIRI_CUBLOCKSIZE),
                                KIRI_CUBLOCKSIZE>>>(
          mCol.data(), mDensity.data(), mScalarFieldMin, mScalarFieldMax,
          mNumOfParticles, ColormapType::BlueWhiteRed);

    default:
      break;
    }
  else
    thrust::fill(thrust::device, mCol.data(), mCol.data() + mNumOfParticles,
                 make_float3(6.f, 66.f, 115.f) / 255.f);

  KIRI_CUCALL(cudaDeviceSynchronize());
  KIRI_CUKERNAL();
}

} // namespace KIRI
