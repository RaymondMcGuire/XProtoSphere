/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-07-26 10:44:51
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-07-26 10:57:01
 * @FilePath: \XProtoSphere\xprotosphere\include\kiri_cuda_utils.h
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _KIRI_CUDA_UTILS_H_
#define _KIRI_CUDA_UTILS_H_

#pragma once

#include <kiri_pbs_cuda/cuda_helper/helper_math.h>
#include <kiri_pch.h>
#include <tiny_obj_loader.h>

class KiriCudaUtils {
public:
  static void ExportCSVFileFromGPU(String Folder, String FileName,
                                   float *Radius, UInt numOfParticles);
  static void ExportCSVFileFromGPU(String Folder, String FileName,
                                 float3 *Positions, float *Radius, UInt numOfParticles);

  static void ExportBgeoFileFromGPU(String Folder, String FileName,
                                    float3 *Positions, float *Radius,
                                    UInt numOfParticles);
  static void ExportBgeoFileFromGPU(String Folder, String FileName,
                                    float4 *Positions, float4 *Colors,
                                    uint *Labels, UInt numOfParticles);
  static void ExportBgeoFileFromGPU(String Folder, String FileName,
                                    float3 *Positions, float3 *Colors,
                                    float *Radius, size_t *Labels,
                                    UInt numOfParticles);
  static void ExportBgeoFileCUDA(String FolderPath, String FileName,
                                 float3 *Positions, float3 *Colors,
                                 float *Radius, size_t *Labels,
                                 UInt numOfParticles);

  static std::vector<float4> ReadBgeoFileForGPU(String Folder, String Name,
                                                bool FlipYZ = false);
  static std::vector<float4> ReadMultiBgeoFilesForGPU(String Folder,
                                                      Vec_String Names,
                                                      bool FlipYZ = false);
  static std::pair<std::vector<float4>, std::vector<float>>
  ReadBgeoFileWithMassForGPU(String Folder, String Name, bool FlipYZ = false);
  static std::vector<float4> ReadMultiBgeoFilesForGPU(Vec_String Folders,
                                                      Vec_String Names,
                                                      bool FlipYZ = false);

  static std::pair<std::vector<float4>, float4>
  ReadBgeoFileWithLowestPointForGPU(String Folder, String Name,
                                    bool FlipYZ = false);
};

#endif