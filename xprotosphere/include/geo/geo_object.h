/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-07-26 10:44:51
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-07-26 10:53:50
 * @FilePath: \XProtoSphere\xprotosphere\include\geo\geo_object.h
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _KIRI_GEO_OBJECT_H_
#define _KIRI_GEO_OBJECT_H_
#pragma once
#include <geo/geo_grid.h>
#include <geo/geo_helper.h>
#include <geo/geo_math_helper.h>
#include <model/mesh_loader.h>
namespace KIRI::GEO {

template <class RealType> class ProtoSphere {
public:
  Vector3<RealType> pos;
  RealType radius;
  RealType targetRadius;

  ProtoSphere(Vector3<RealType> p, RealType rad, RealType tarRad)
      : pos(p), radius(rad), targetRadius(tarRad) {}
};

template <Int N, class RealType> class GeometryObject {
public:
  GeometryObject() {}
  static constexpr UInt dimension() noexcept { return static_cast<UInt>(N); }
  virtual String name() = 0;
  virtual RealType sdf(const VectorX<N, RealType> &ppos0,
                       bool bNegativeInside = false) const = 0;

  VectorX<N, RealType> transform(const VectorX<N, RealType> &ppos) const;
  VectorX<N, RealType> invTransform(const VectorX<N, RealType> &ppos) const;

  BoundingBoxX<N, RealType> bbox() const;

  void setScale(RealType scale) { mUniformScale = scale; }

protected:
  bool mTransformed = false;
  RealType mUniformScale = RealType(1.0);
  MatrixXX<N + 1, RealType> mTransformationMatrix =
      MatrixXX<N + 1, RealType>(1.0);
  MatrixXX<N + 1, RealType> mInvTransformationMatrix =
      MatrixXX<N + 1, RealType>(1.0);
};

template <class RealType>
class MeshObject : public GeometryObject<3, RealType> {
public:
  MeshObject() = delete;
  MeshObject(const String &meshName,
             const RealType scale = static_cast<RealType>(0.5));
  MeshObject(const String &meshName,
             const Vector3<RealType> scale = Vector3<RealType>(0.5));

  virtual String name() override { return String("MeshObject"); }
  virtual RealType sdf(const VectorX<3, RealType> &ppos0,
                       bool bNegativeInside = false) const override;
  Vector3<RealType> sdfClosestPoint(const Vector3<RealType> &pos) const;

  RealType sdfStep() { return mStep; }

  Vector3<RealType> aabbMin() { return mAABBMin; }
  Vector3<RealType> aabbMax() { return mAABBMax; }

  Vector<RealType> sdfData() {
    Vector<RealType> data;
    for (auto i = 0;
         i < mSDFData.width() * mSDFData.height() * mSDFData.depth(); i++) {
      data.emplace_back(mSDFData[i]);
    }
    return data;
  }

  Vector<Vector3<RealType>> sdfClosestPointData() {
    Vector<Vector3<RealType>> data;
    for (auto i = 0; i < mSDFClosestPoint.width() * mSDFClosestPoint.height() *
                             mSDFClosestPoint.depth();
         i++) {
      data.emplace_back(mSDFClosestPoint[i]);
    }
    return data;
  }

  GeoGrid<RealType> gridData() { return mGrid3D; };
  const SharedPtr<MeshObjectLoader> &meshData() { return mMeshData; }

protected:
  void computeSDF();

  void loadMeshData(const String &meshName, const RealType scale);
  void loadMeshData(const String &meshName, const Vector3<RealType> scale);
  void computeSDFMesh(const Vector<Vector3<RealType>> &faces,
                      const Vector<Vector3<RealType>> &vertices,
                      const Vector3<RealType> &origin, RealType CellSize,
                      RealType ni, RealType nj, RealType nk,
                      Array3<RealType> &SDF, Int exactBand = 1);

  bool mSDFGenerated = false;
  RealType mStep = 0.01f;

  Array3<RealType> mSDFData;
  Array3<Vector3<RealType>> mSDFClosestPoint;
  GeoGrid<RealType> mGrid3D;

  Vector3<RealType> mAABBMin, mAABBMax;
  SharedPtr<MeshObjectLoader> mMeshData;
};
typedef SharedPtr<MeshObject<float>> MeshObject3FPtr;
typedef SharedPtr<MeshObject<double>> MeshObject3DPtr;

} // namespace KIRI::GEO

#include "inl/geo_object-inl.h"
#endif
