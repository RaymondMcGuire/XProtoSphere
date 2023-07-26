/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-02-23 22:50:51
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-02-23 23:13:24
 * @FilePath:
 * \XProtoSphere\xprotosphere_cuda\src\kiri_pbs_cuda\system\cuda_base_system.cu
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
// clang-format off
#include <kiri_pbs_cuda/system/cuda_base_system.cuh>

// clang-format on
namespace KIRI {
CudaBaseSystem::CudaBaseSystem(CudaBaseSolverPtr &solver,
                               CudaGNSearcherPtr &searcher,
                               const size_t maxNumOfParticles)
    : mSolver(std::move(solver)), mSearcher(std::move(searcher)) {}

void CudaBaseSystem::updateWorldSize(const float3 lowestPoint,
                                     const float3 highestPoint) {

  CUDA_BOUNDARY_PARAMS.lowest_point = lowestPoint;
  CUDA_BOUNDARY_PARAMS.highest_point = highestPoint;
  CUDA_BOUNDARY_PARAMS.world_size = highestPoint - lowestPoint;
  CUDA_BOUNDARY_PARAMS.world_center = (highestPoint + lowestPoint) / 2.f;

  CUDA_BOUNDARY_PARAMS.grid_size = make_int3(
      (CUDA_BOUNDARY_PARAMS.highest_point - CUDA_BOUNDARY_PARAMS.lowest_point) /
      CUDA_BOUNDARY_PARAMS.kernel_radius);

  mSearcher->updateWorldSize(lowestPoint, highestPoint);
}

float CudaBaseSystem::updateSystem(float renderInterval) {
  cudaEvent_t start, stop;
  KIRI_CUCALL(cudaEventCreate(&start));
  KIRI_CUCALL(cudaEventCreate(&stop));
  KIRI_CUCALL(cudaEventRecord(start, 0));

  try {
    onUpdateSolver(renderInterval);
  } catch (const char *s) {
    std::cout << s << "\n";
  } catch (...) {
    std::cout << "Unknown Exception at " << __FILE__ << ": line " << __LINE__
              << "\n";
  }

  float milliseconds;
  KIRI_CUCALL(cudaEventRecord(stop, 0));
  KIRI_CUCALL(cudaEventSynchronize(stop));
  KIRI_CUCALL(cudaEventElapsedTime(&milliseconds, start, stop));
  KIRI_CUCALL(cudaEventDestroy(start));
  KIRI_CUCALL(cudaEventDestroy(stop));
  return milliseconds;
}
} // namespace KIRI
