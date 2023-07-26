/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-07-08 14:46:49
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-09-29 14:25:38
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\sampler\cuda_sampler_struct.cuh
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_SAMPLER_STRUCT_CUH_
#define _CUDA_SAMPLER_STRUCT_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI {
template <typename T> struct CudaAxisAlignedBox {
  T Min;
  T Max;
  __device__ __host__ CudaAxisAlignedBox(T min, T max) : Min(min), Max(max) {}
};

struct LevelSetShapeInfo {
  CudaAxisAlignedBox<float3> BBox;
  int3 GridSize;
  float CellSize;
  size_t NumOfFaces;

  LevelSetShapeInfo(CudaAxisAlignedBox<float3> boundingBox, float cellSize,
                    size_t numOfFaces)
      : CellSize(cellSize), BBox(boundingBox), NumOfFaces(numOfFaces) {
    GridSize = float3_to_int3((BBox.Max - BBox.Min) / cellSize);
  }
};

struct GridInfo {
  CudaAxisAlignedBox<float3> BBox;
  int3 GridSize;
  float CellSize;

  GridInfo(CudaAxisAlignedBox<float3> boundingBox, float cellSize,
           int3 gridSize)
      : CellSize(cellSize), BBox(boundingBox), GridSize(gridSize) {}
};
} // namespace KIRI

#endif