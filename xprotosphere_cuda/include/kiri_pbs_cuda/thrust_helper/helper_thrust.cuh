/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-19 00:31:33
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-01-29 02:17:45
 * @FilePath:
 * \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\thrust_helper\helper_thrust.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _THRUST_HELPER_CUH_
#define _THRUST_HELPER_CUH_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace ThrustHelper {

typedef thrust::tuple<float3, float3> TUPLE_FLOAT3X2;
typedef thrust::tuple<float3, float3, float3> TUPLE_FLOAT3X3;

template <typename T> struct Abs {
  __host__ __device__ T operator()(const T &x) const {
    return x < T(0) ? -x : x;
  }
};

template <typename T> struct Squared {
  __host__ __device__ T operator()(const T &x) const { return x * x; }
};

template <typename T> struct Plus {
  T B;
  Plus(const T b) : B(b) {}
  __host__ __device__ T operator()(const T &a) const { return a + B; }
};

template <typename T> struct AbsPlus {
  __host__ __device__ T operator()(const T &a, const T &b) const {
    return abs(a) + abs(b);
  }
};

template <typename T> struct LengthSquared {
  __host__ __device__ T operator()(const T &a, const T &b) const {
    return a * a + b * b;
  }
};

template <typename T> struct CompareLengthCuda {
  static_assert(KIRI::IsSame_Float2<T>::value ||
                    KIRI::IsSame_Float3<T>::value ||
                    KIRI::IsSame_Float4<T>::value,
                "data type is not correct");

  __host__ __device__ bool operator()(T f1, T f2) {
    return length(f1) < length(f2);
  }
};

static inline __host__ __device__ int3 ComputeGridXYZByPos3(
    const float3 &pos, const float cellSize, const int3 &gridSize) {
  int x = min(max((int)(pos.x / cellSize), 0), gridSize.x - 1),
      y = min(max((int)(pos.y / cellSize), 0), gridSize.y - 1),
      z = min(max((int)(pos.z / cellSize), 0), gridSize.z - 1);

  return make_int3(x, y, z);
}

template <typename T> struct Pos2GridHash {
  static_assert(KIRI::IsSame_Float3<T>::value || KIRI::IsSame_Float4<T>::value,
                "position data structure must be float3 or float4");

  float3 mLowestPoint;
  float mCellSize;
  int3 mGridSize;
  __host__ __device__ Pos2GridHash(const float3 lowestPoint,
                                   const float cellSize, const int3 &gridSize)
      : mLowestPoint(lowestPoint), mCellSize(cellSize), mGridSize(gridSize) {}

  __host__ __device__ uint operator()(const T &pos) {
    float3 relPos = make_float3(pos.x, pos.y, pos.z) - mLowestPoint;
    int3 grid_xyz = ComputeGridXYZByPos3(relPos, mCellSize, mGridSize);
    return grid_xyz.x * mGridSize.y * mGridSize.z + grid_xyz.y * mGridSize.z +
           grid_xyz.z;
  }
};

template <typename T> struct Pos2GridXYZ {
  static_assert(KIRI::IsSame_Float3<T>::value || KIRI::IsSame_Float4<T>::value,
                "position data structure must be float3 or float4");

  float3 mLowestPoint;
  float mCellSize;
  int3 mGridSize;
  __host__ __device__ Pos2GridXYZ(const float3 lowestPoint,
                                  const float cellSize, const int3 &gridSize)
      : mLowestPoint(lowestPoint), mCellSize(cellSize), mGridSize(gridSize) {}

  __host__ __device__ int3 operator()(const T &pos) {
    float3 relPos = make_float3(pos.x, pos.y, pos.z) - mLowestPoint;
    return ComputeGridXYZByPos3(relPos, mCellSize, mGridSize);
  }
};

struct GridXYZ2GridHash {
  int3 mGridSize;
  __host__ __device__ GridXYZ2GridHash(const int3 &gridSize)
      : mGridSize(gridSize) {}

  template <typename T> __host__ __device__ uint operator()(T x, T y, T z) {
    return (x >= 0 && x < mGridSize.x && y >= 0 && y < mGridSize.y && z >= 0 &&
            z < mGridSize.z)
               ? (((x * mGridSize.y) + y) * mGridSize.z + z)
               : (mGridSize.x * mGridSize.y * mGridSize.z);
  }
};

} // namespace ThrustHelper

#endif