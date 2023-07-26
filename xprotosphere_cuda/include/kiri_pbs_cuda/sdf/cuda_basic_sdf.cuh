/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-07-26 10:44:51
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-07-26 11:01:17
 * @FilePath:
 * \XProtoSphere\xprotosphere_cuda\include\kiri_pbs_cuda\sdf\cuda_basic_sdf.cuh
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _CUDA_BASIC_SDF_CUH_
#define _CUDA_BASIC_SDF_CUH_

#pragma once

#include <kiri_pbs_cuda/math/cuda_math_utils.cuh>
#include <kiri_pbs_cuda/sampler/cuda_sampler_struct.cuh>

namespace KIRI {

struct Mesh3SDF {
  float _scale;
  bool _inside_negative;
  GridInfo _info;

  __host__ __device__ Mesh3SDF(const GridInfo info, const float scale = 1.f,
                               const bool inside_negative = false)
      : _info(info), _scale(scale), _inside_negative(inside_negative) {}

  __device__ float SDF(const float3 pos, const float *sdf_grid) {

    auto grid_pos = (pos - _info.BBox.Min) / _info.CellSize;
    auto d =
        _scale * interpolate_value_linear(grid_pos, sdf_grid, _info.GridSize);

    return _inside_negative ? -d : d;
  }

  __device__ float SDFWithOffset(const float3 pos, const float *sdf_grid,
                                 const float rnd_offset) {

    auto grid_pos = (pos - _info.BBox.Min) / _info.CellSize;
    auto d =
        _scale * interpolate_value_linear(grid_pos, sdf_grid, _info.GridSize);

    d = _inside_negative ? -d : d;

    auto step_num = d / rnd_offset;
    auto cell_sdf = ceilf(step_num) * rnd_offset - d;
    auto floor_sdf = d - floorf(step_num) * rnd_offset;
    return min(min(d, floor_sdf), floor_sdf);
  }

  __device__ float3 closestPoint(const float3 pos, const float3 *closest_grid) {

    auto grid_pos = (pos - _info.BBox.Min) / _info.CellSize;
    auto closest = _scale * interpolate_value_linear(grid_pos, closest_grid,
                                                     _info.GridSize);

    return closest;
  }

  __device__ float3 gradSDF(const float3 pos, const float *sdf_grid,
                            const float dx = 1e-4f) {
    auto v000 = SDF(pos + make_float3(-dx), sdf_grid);
    auto v001 = SDF(pos + make_float3(-dx, -dx, dx), sdf_grid);
    auto v010 = SDF(pos + make_float3(-dx, dx, -dx), sdf_grid);
    auto v011 = SDF(pos + make_float3(-dx, dx, dx), sdf_grid);

    auto v100 = SDF(pos + make_float3(dx, -dx, -dx), sdf_grid);
    auto v101 = SDF(pos + make_float3(dx, -dx, dx), sdf_grid);
    auto v110 = SDF(pos + make_float3(dx, dx, -dx), sdf_grid);
    auto v111 = SDF(pos + make_float3(dx), sdf_grid);

    auto ddx00 = v100 - v000;
    auto ddx10 = v110 - v010;
    auto ddx01 = v101 - v001;
    auto ddx11 = v111 - v011;

    auto ddy00 = v010 - v000;
    auto ddy10 = v110 - v100;
    auto ddy01 = v011 - v001;
    auto ddy11 = v111 - v101;

    auto ddz00 = v001 - v000;
    auto ddz10 = v101 - v100;
    auto ddz01 = v011 - v010;
    auto ddz11 = v111 - v110;

    auto grad =
        (make_float3(ddx00, ddy00, ddz00) + make_float3(ddx01, ddy01, ddz01) +
         make_float3(ddx10, ddy10, ddz10) + make_float3(ddx11, ddy11, ddz11)) *
        0.25f;

    return grad;
  }
};
} // namespace KIRI

#endif /* _CUDA_BASIC_SDF_CUH_ */