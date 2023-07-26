/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-12-03 01:39:06
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-03-26 15:57:17
 * @FilePath:
 * \Kiri\KiriCore\include\kiri_core\geo\inl\geo_particle_generator-inl.h
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#include <geo/geo_particle_generator.h>
#include <omp.h>
#include <random>
namespace KIRI::GEO {

template <Int N, class RealType>
inline ParticleGenerator<N, RealType>::ParticleGenerator(RealType radius,
                                                         RealType samplingRatio,
                                                         RealType jitterRatio)
    : mParticleRadius(radius), mSamplingRatio(samplingRatio),
      mJitterRatio(jitterRatio){

      };

template <Int N, class RealType>
void ParticleGenerator<N, RealType>::generate(GeometryPtr geometry,
                                              bool bNegativeInside) {
  auto spacing = mParticleRadius * 2.0 * mSamplingRatio;
  auto jitter = mJitterRatio * mParticleRadius;

  auto bbox = geometry->bbox();

  auto boxMin = bbox.LowestPoint;
  auto boxMax = bbox.HighestPoint;

  // KIRI_LOG_INFO(" mAABBMin={0},{1},{2}; mAABBMax={3},{4},{5}", boxMin.x,
  //               boxMin.y, boxMin.z, boxMax.x, boxMax.y, boxMax.z);

  auto pGrid = HELPER::createGrid<size_t, N, RealType>(boxMin, boxMax, spacing);

  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  std::uniform_real_distribution<RealType> dist(-1.0, 1.0);
  omp_lock_t writelock;
  omp_init_lock(&writelock);
  parallelFor(pGrid, [&](auto... idx) {
    auto ppos =
        boxMin + VectorX<N, RealType>(idx...) * static_cast<RealType>(spacing);

    auto geoPhi = geometry->sdf(ppos);

    if (geoPhi > 0.0) {
      if (geoPhi < jitter) {
        ppos +=
            jitter *
            HELPER::fRand11<N, RealType>::template vrnd<VectorX<N, RealType>>()
                .normalized();
      }

      omp_set_lock(&writelock);
      mParticles.emplace_back(
          Vector4<RealType>(Vector3<RealType>(ppos), mParticleRadius));
      if (bNegativeInside)
        mSDF.emplace_back(-geoPhi);
      else
        mSDF.emplace_back(geoPhi);
      omp_unset_lock(&writelock);
    }
  });

  // KIRI_LOG_INFO("Sampling Number={0:d}", mParticles.size());
}

template <Int N, class RealType>
void ParticleGenerator<N, RealType>::GenerateByMeshObject(
    MeshObjectPtr meshobj, bool bNegativeInside) {
  auto spacing = mParticleRadius * 2.0 * mSamplingRatio;
  auto jitter = mJitterRatio * mParticleRadius;

  auto boxMin = meshobj->aabbMin();
  auto boxMax = meshobj->aabbMax();

  KIRI_LOG_INFO(" mAABBMin={0},{1},{2}; mAABBMax={3},{4},{5}", boxMin.x,
                boxMin.y, boxMin.z, boxMax.x, boxMax.y, boxMax.z);

  auto pGrid = HELPER::createGrid<size_t, N, RealType>(boxMin, boxMax, spacing);

  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  std::uniform_real_distribution<RealType> dist(-1.0, 1.0);
  omp_lock_t writelock;
  omp_init_lock(&writelock);
  parallelFor(pGrid, [&](auto... idx) {
    auto ppos =
        boxMin + VectorX<N, RealType>(idx...) * static_cast<RealType>(spacing);

    auto geoPhi = meshobj->sdf(ppos);

    if (geoPhi > 0.0) {
      if (geoPhi < jitter) {
        ppos +=
            jitter *
            HELPER::fRand11<N, RealType>::template vrnd<VectorX<N, RealType>>()
                .normalized();
      }

      omp_set_lock(&writelock);
      mParticles.emplace_back(
          Vector4<RealType>(Vector3<RealType>(ppos), mParticleRadius));
      if (bNegativeInside)
        mSDF.emplace_back(-geoPhi);
      else
        mSDF.emplace_back(geoPhi);
      omp_unset_lock(&writelock);
    }
  });

  // KIRI_LOG_INFO("Sampling Number={0:d}", mParticles.size());
}
} // namespace KIRI::GEO