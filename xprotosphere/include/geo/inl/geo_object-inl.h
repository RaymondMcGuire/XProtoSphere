/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-07-26 10:44:51
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-07-26 10:54:37
 * @FilePath: \XProtoSphere\xprotosphere\include\geo\inl\geo_object-inl.h
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _KIRI_GEO_OBJECT_INL_H_
#define _KIRI_GEO_OBJECT_INL_H_
#pragma once

namespace KIRI::GEO {

template <Int N, class RealType>
VectorX<N, RealType>
GeometryObject<N, RealType>::transform(const VectorX<N, RealType> &ppos) const {
  if (!mTransformed) {
    return ppos;
  } else {
    VectorX<N, RealType> trans_vec;
    trans_vec.init(mTransformationMatrix * VectorX<N + 1, RealType>(ppos, 1.0));
    return trans_vec;
  }
}

template <Int N, class RealType>
VectorX<N, RealType> GeometryObject<N, RealType>::invTransform(
    const VectorX<N, RealType> &ppos) const {
  if (!mTransformed) {
    return ppos;
  } else {
    VectorX<N, RealType> trans_vec;
    trans_vec.init(mInvTransformationMatrix *
                   VectorX<N + 1, RealType>(ppos, 1.0));
    return trans_vec;
  }
}

template <Int N, class RealType>
BoundingBoxX<N, RealType> GeometryObject<N, RealType>::bbox() const {
  auto min_aabb = this->transform(VectorX<N, RealType>(0)) -
                  VectorX<N, RealType>(mUniformScale) *
                      std::sqrt(VectorX<N, RealType>(1.0).sum());
  auto max_aabb = this->transform(VectorX<N, RealType>(0)) +
                  VectorX<N, RealType>(mUniformScale) *
                      std::sqrt(VectorX<N, RealType>(1.0).sum());
  return BoundingBoxX<N, RealType>(min_aabb, max_aabb);
}

template <class RealType>
MeshObject<RealType>::MeshObject(const String &meshName, const RealType scale) {
  loadMeshData(meshName, scale);
  computeSDF();
}

template <class RealType>
MeshObject<RealType>::MeshObject(const String &meshName,
                                 const Vector3<RealType> scale) {
  loadMeshData(meshName, scale);
  computeSDF();
}

template <class RealType>
void MeshObject<RealType>::loadMeshData(const String &meshName,
                                        const RealType scale) {
  mMeshData = std::make_shared<MeshObjectLoader>(meshName, "models", ".obj");
  mMeshData->scaleToBox(scale, 3.0 * mStep);

  auto bbox_min = mMeshData->getAABBMin();
  auto bbox_max = mMeshData->getAABBMax();

  mAABBMin = Vector3<RealType>(bbox_min.x, bbox_min.y, bbox_min.z);
  mAABBMax = Vector3<RealType>(bbox_max.x, bbox_max.y, bbox_max.z);
}

template <class RealType>
void MeshObject<RealType>::loadMeshData(const String &meshName,
                                        const Vector3<RealType> scale) {
  mMeshData = std::make_shared<MeshObjectLoader>(meshName, "models", ".obj");

  // need padding
  mMeshData->scaleToBox(scale);

  auto bbox_min = mMeshData->getAABBMin();
  auto bbox_max = mMeshData->getAABBMax();

  mAABBMin = Vector3<RealType>(bbox_min.x, bbox_min.y, bbox_min.z) -
             Vector3<RealType>(3.0 * mStep);
  mAABBMax = Vector3<RealType>(bbox_max.x, bbox_max.y, bbox_max.z) +
             Vector3<RealType>(3.0 * mStep);
}

template <class RealType> void MeshObject<RealType>::computeSDF() {
  mGrid3D.SetGrid(mAABBMin, mAABBMax, mStep);

  Vector<Vector3<RealType>> vertexList(mMeshData->getNVertices());
  Vector<Vector3<RealType>> faceList(mMeshData->getNFaces());

  auto vertices = mMeshData->getVertices();
  auto faces = mMeshData->getFaces();
  for (auto i = 0; i < vertexList.size(); i++)
    vertexList[i] = Vector3<RealType>(vertices[i * 3], vertices[i * 3 + 1],
                                      vertices[i * 3 + 2]);

  for (auto i = 0; i < faceList.size(); i++)
    faceList[i] =
        Vector3<RealType>(faces[i * 3], faces[i * 3 + 1], faces[i * 3 + 2]);

  auto bbox_min = mMeshData->getAABBMin();
  // Compute SDF data
  computeSDFMesh(faceList, vertexList,
                 Vector3<RealType>(bbox_min.x, bbox_min.y, bbox_min.z), mStep,
                 mGrid3D.getNCells()[0], mGrid3D.getNCells()[1],
                 mGrid3D.getNCells()[2], mSDFData);
  mSDFGenerated = true;
}

template <class RealType>
void MeshObject<RealType>::computeSDFMesh(
    const Vector<Vector3<RealType>> &faces,
    const Vector<Vector3<RealType>> &vertices, const Vector3<RealType> &origin,
    RealType CellSize, RealType ni, RealType nj, RealType nk,
    Array3<RealType> &SDF, Int exactBand) {
  // KIRI_LOG_INFO("ni,nj,nk={0},{1},{2};CellSize={3}", ni, nj, nk, CellSize);

  kiri_math_mini::setMaxNumberOfThreads(kiri_math_mini::maxNumberOfThreads());
  SDF.resize(ni, nj, nk, (ni + nj + nk) * CellSize); // upper bound on distance
  mSDFClosestPoint.resize(ni, nj, nk, Vector3<RealType>(Huge<RealType>()));

  Array3UI closest_tri(ni, nj, nk, 0xffffffff);

  Array3UI intersectionCount(ni, nj, nk, 0u);

  for (UInt face = 0, faceEnd = static_cast<UInt>(faces.size()); face < faceEnd;
       ++face) {
    UInt p = faces[face][0];
    UInt q = faces[face][1];
    UInt r = faces[face][2];

    // coordinates in grid to high precision
    Vector3<RealType> fp = (vertices[p] - origin) / CellSize;
    Vector3<RealType> fq = (vertices[q] - origin) / CellSize;
    Vector3<RealType> fr = (vertices[r] - origin) / CellSize;

    // do distances nearby
    Int i0 = kiri_math_mini::clamp(
        static_cast<Int>(kiri_math_mini::min3(fp[0], fq[0], fr[0])) - exactBand,
        0, static_cast<Int>(ni - 1));
    Int i1 = kiri_math_mini::clamp(
        static_cast<Int>(kiri_math_mini::max3(fp[0], fq[0], fr[0])) +
            exactBand + 1,
        0, static_cast<Int>(ni - 1));
    Int j0 = kiri_math_mini::clamp(
        static_cast<Int>(kiri_math_mini::min3(fp[1], fq[1], fr[1])) - exactBand,
        0, static_cast<Int>(nj - 1));
    Int j1 = kiri_math_mini::clamp(
        static_cast<Int>(kiri_math_mini::max3(fp[1], fq[1], fr[1])) +
            exactBand + 1,
        0, static_cast<Int>(nj - 1));
    Int k0 = kiri_math_mini::clamp(
        static_cast<Int>(kiri_math_mini::min3(fp[2], fq[2], fr[2])) - exactBand,
        0, static_cast<Int>(nk - 1));
    Int k1 = kiri_math_mini::clamp(
        static_cast<Int>(kiri_math_mini::max3(fp[2], fq[2], fr[2])) +
            exactBand + 1,
        0, static_cast<Int>(nk - 1));

    kiri_math_mini::parallelFor(
        i0, i1 + 1, j0, j1 + 1, k0, k1 + 1, [&](Int i, Int j, Int k) {
          Vector3<RealType> gx = Vector3<RealType>(i, j, k) * CellSize + origin;
          // RealType d = KIRI::GEO::HELPER::point_triangle_distance(
          //     gx, vertices[p], vertices[q], vertices[r]);
          auto [d, closest_p] = KIRI::GEO::HELPER::point_project2triangle(
              gx, vertices[p], vertices[q], vertices[r]);

          if (d < SDF(i, j, k)) {
            SDF(i, j, k) = d;
            closest_tri(i, j, k) = face;
            mSDFClosestPoint(i, j, k) = closest_p;
          }
        });

    Int expand_val = 1;

    j0 = kiri_math_mini::clamp(
        static_cast<Int>(std::ceil(kiri_math_mini::min3(fp[1], fq[1], fr[1]))) -
            expand_val,
        0, static_cast<Int>(nj - 1));
    j1 = kiri_math_mini::clamp(static_cast<Int>(std::floor(
                                   kiri_math_mini::max3(fp[1], fq[1], fr[1]))) +
                                   expand_val,
                               0, static_cast<Int>(nj - 1));
    k0 = kiri_math_mini::clamp(
        static_cast<Int>(std::ceil(kiri_math_mini::min3(fp[2], fq[2], fr[2]))) -
            expand_val,
        0, static_cast<Int>(nk - 1));
    k1 = kiri_math_mini::clamp(static_cast<Int>(std::floor(
                                   kiri_math_mini::max3(fp[2], fq[2], fr[2]))) +
                                   expand_val,
                               0, static_cast<Int>(nk - 1));

    for (Int k = k0; k <= k1; ++k) {

      for (Int j = j0; j <= j1; ++j) {
        RealType a, b, c;

        if (KIRI::GEO::HELPER::point_in_triangle_2d<RealType>(
                j, k, fp[1], fp[2], fq[1], fq[2], fr[1], fr[2], a, b, c)) {
          auto fi = a * fp[0] + b * fq[0] + c * fr[0];

          Int i_interval = std::max(static_cast<Int>(std::ceil(fi)), 0);

          if (i_interval < static_cast<Int>(ni)) {
            ++intersectionCount(i_interval, j, k);
          }
        }
      }
    }
  }

  for (UInt pass = 0; pass < 2; ++pass) {
    KIRI::GEO::HELPER::sweep(faces, vertices, SDF, closest_tri,
                             mSDFClosestPoint, origin, CellSize, +1, +1, +1);
    KIRI::GEO::HELPER::sweep(faces, vertices, SDF, closest_tri,
                             mSDFClosestPoint, origin, CellSize, -1, -1, -1);
    KIRI::GEO::HELPER::sweep(faces, vertices, SDF, closest_tri,
                             mSDFClosestPoint, origin, CellSize, +1, +1, -1);
    KIRI::GEO::HELPER::sweep(faces, vertices, SDF, closest_tri,
                             mSDFClosestPoint, origin, CellSize, -1, -1, +1);
    KIRI::GEO::HELPER::sweep(faces, vertices, SDF, closest_tri,
                             mSDFClosestPoint, origin, CellSize, +1, -1, +1);
    KIRI::GEO::HELPER::sweep(faces, vertices, SDF, closest_tri,
                             mSDFClosestPoint, origin, CellSize, -1, +1, -1);
    KIRI::GEO::HELPER::sweep(faces, vertices, SDF, closest_tri,
                             mSDFClosestPoint, origin, CellSize, +1, -1, -1);
    KIRI::GEO::HELPER::sweep(faces, vertices, SDF, closest_tri,
                             mSDFClosestPoint, origin, CellSize, -1, +1, +1);
  }

  kiri_math_mini::parallelFor<UInt>(0, static_cast<UInt>(nk), [&](UInt k) {
    for (UInt j = 0; j < nj; ++j) {
      UInt total_count = 0;

      for (UInt i = 0; i < ni; ++i) {
        total_count += intersectionCount(i, j, k);

        if (!(total_count & 1)) {
          SDF(i, j, k) = -SDF(i, j, k);
        }
      }
    }
  });
}

template <class RealType>
RealType MeshObject<RealType>::sdf(const VectorX<3, RealType> &ppos0,
                                   bool bNegativeInside) const {

  KIRI_ASSERT(mSDFGenerated);

  // TODO transform
  // auto ppos = this->invTransform(ppos0);
  auto ppos = ppos0;

  auto gridPos = mGrid3D.getGridCoordinate(ppos);
  RealType d = this->mUniformScale *
               KIRI::GEO::HELPER::interpolateValueLinear(gridPos, mSDFData);
  return bNegativeInside ? -d : d;
}

template <class RealType>
Vector3<RealType>
MeshObject<RealType>::sdfClosestPoint(const Vector3<RealType> &pos) const {

  KIRI_ASSERT(mSDFGenerated);

  // TODO transform
  // auto ppos = this->invTransform(ppos0);
  auto ppos = pos;

  auto gridPos = mGrid3D.getGridCoordinate(ppos);
  auto closest =
      this->mUniformScale *
      KIRI::GEO::HELPER::interpolateValueLinear(gridPos, mSDFClosestPoint);
  return closest;
}

} // namespace KIRI::GEO

#endif