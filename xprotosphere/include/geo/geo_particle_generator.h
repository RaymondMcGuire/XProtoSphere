/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-20 11:12:43
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-11-30 16:02:22
 * @FilePath: \Kiri\KiriCore\include\kiri_core\geo\geo_particle_generator.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _KIRI_GEO_PARTICLE_GENERATOR_H_
#define _KIRI_GEO_PARTICLE_GENERATOR_H_
#pragma once
#include <geo/geo_object.h>

namespace KIRI::GEO {
template <Int N, class RealType> class ParticleGenerator {
public:
  using GeometryPtr = SharedPtr<GeometryObject<N, RealType>>;
  using MeshObjectPtr = SharedPtr<MeshObject<RealType>>;
  ParticleGenerator() = delete;
  ParticleGenerator(RealType radius, RealType samplingRatio = 1.0,
                    RealType jitterRatio = 0.1);

  const auto particles() const { return this->mParticles; }
  const auto sdf() const { return this->mSDF; }
  void generate(GeometryPtr geometry, bool bNegativeInside = false);
  void GenerateByMeshObject(MeshObjectPtr mesh, bool bNegativeInside = false);

private:
  RealType mParticleRadius;
  RealType mSamplingRatio;
  RealType mJitterRatio;

  Vec_Vec4<RealType> mParticles;
  Vector<RealType> mSDF;
};
} // namespace KIRI::GEO

#include "inl/geo_particle_generator-inl.h"
#endif
