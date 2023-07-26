/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-02-25 03:59:39
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-03-04 15:42:43
 * @FilePath:
 * \XProtoSphere\xprotosphere_cuda\src\kiri_pbs_cuda\particle\cuda_sph_bn_particles.cu
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */

#include <kiri_pbs_cuda/particle/cuda_sph_bn_particles.cuh>
#include <kiri_pbs_cuda/particle/cuda_sph_bn_particles_gpu.cuh>

#include <kiri_pbs_cuda/thrust_helper/helper_thrust.cuh>
#include <thrust/device_ptr.h>

namespace KIRI {

void CudaSphBNParticles::Advect(const float dt) {
  // uint num = this->size();

  // auto densityArray = thrust::device_pointer_cast(this->densityPtr());
  // // float minDensity = *(thrust::min_element(densityArray, densityArray +
  // // num)); float maxDensity = *(thrust::max_element(densityArray,
  // densityArray
  // // + num));

  // float minDensity = 200.f;
  // float maxDensity = 2000.f;

  // BNAdvect_CUDA<<<mCudaGridSize, KIRI_CUBLOCKSIZE>>>(
  //     this->posPtr(), this->velPtr(), this->tmpPosPtr(), this->tmpVelPtr(),
  //     this->colorPtr(), this->densityPtr(), this->normalPtr(),
  //     this->boundaryPtr(), this->size(), dt, minDensity, maxDensity,
  //     mEpsilon);

  // KIRI_CUCALL(cudaDeviceSynchronize());
  // KIRI_CUKERNAL();
}

} // namespace KIRI
