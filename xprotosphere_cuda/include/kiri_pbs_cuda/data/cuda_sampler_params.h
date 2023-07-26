/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-11-24 16:21:35
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-11-24 16:22:35
 * @FilePath: \Kiri\KiriPBSCuda\include\kiri_pbs_cuda\data\cuda_sampler_params.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _CUDA_SAMPLER_PARAMS_H_
#define _CUDA_SAMPLER_PARAMS_H_

#pragma once

#include <kiri_pbs_cuda/kiri_pbs_pch.cuh>

namespace KIRI {

struct ProtoSphereVolumeData {
  Vec_Float3 pos;
  Vec_Float radius;
  Vec_Float targetRadius;
  Vec_Float3 col;
};

struct CudaProtoSphereParams {
  float error_rate;
  uint max_iteration_num;
  float decay;
  float relax_dt;
};

enum XProtoSphereSystemStatus {
  XPROTOTYPE_SEARCHING = 0,
  XPROTOTYPE_INSERTED = 1,
  DEM_RELAXATION = 2,
  DEM_RELAXATION_FINISH = 3,
  XPROTOSPHERE_FINISH = 4
};

extern CudaProtoSphereParams CUDA_PROTOSPHERE_PARAMS;
} // namespace KIRI

#endif