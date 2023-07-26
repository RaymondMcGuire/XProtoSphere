/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-06-12 08:44:06
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-06-12 08:53:16
 * @FilePath: \XProtoSphere\xprotosphere\include\model\mesh_loader.h
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef _KIRI_MESH_OBJECT_LOADER_H_
#define _KIRI_MESH_OBJECT_LOADER_H_
#pragma once
#include <kiri_pch.h>
#include <tiny_obj_loader.h>

class MeshObjectLoader {
public:
  MeshObjectLoader() { clearData(); }
  MeshObjectLoader(const String &name, const String &folder = "models",
                   const String &ext = ".obj", const String &fileName = "");

  bool Load(const String &filePath);
  void scaleToBox(tinyobj::real_t BoxScale = 1.0, float padding = 0.1f);
  void
  scaleToBox(Vector3<tinyobj::real_t> BoxScale = Vector3<tinyobj::real_t>(0.0));
  void Normalize();
  tinyobj::real_t volume();

  auto getMeshCenter() const {
    KIRI_ASSERT(mMeshReady);
    return tinyobj::real_t(0.5) * (mAABBMin + mAABBMax);
  }
  auto getNTriangles() const { return mNumTriangles; }
  const auto &getAABBMin() const {
    KIRI_ASSERT(mMeshReady);
    return mAABBMin;
  }
  const auto &getAABBMax() const {
    KIRI_ASSERT(mMeshReady);
    return mAABBMax;
  }

  const auto &getVertices() const {
    KIRI_ASSERT(mMeshReady);
    return mVertices;
  }
  const auto &getNormals() const {
    KIRI_ASSERT(mMeshReady);
    return mNormals;
  }
  const auto &getTexCoord2D() const {
    KIRI_ASSERT(mMeshReady);
    return mTexCoord2D;
  }

  const auto &getFaces() const {
    KIRI_ASSERT(mMeshReady);
    return mFaces;
  }
  const auto &getFaceVertices() const {
    KIRI_ASSERT(mMeshReady);
    return mFaceVertices;
  }
  const auto &getFaceVertexNormals() const {
    KIRI_ASSERT(mMeshReady);
    return mFaceVertexNormals;
  }
  const auto &getFaceVertexColors() const {
    KIRI_ASSERT(mMeshReady);
    return mFaceVertexColors;
  }
  const auto &getFaceVTexCoord2D() const {
    KIRI_ASSERT(mMeshReady);
    return mFaceVertexTexCoord2D;
  }

  auto getNFaces() const noexcept {
    KIRI_ASSERT(mMeshReady);
    return (mFaces.size() / 3);
  }
  auto getNVertices() const noexcept {
    KIRI_ASSERT(mMeshReady);
    return (mVertices.size() / 3);
  }
  auto getNFaceVertices() const noexcept {
    KIRI_ASSERT(mMeshReady);
    return (mFaceVertices.size() / 3);
  }

  std::vector<Vector3<tinyobj::real_t>> pos;
  std::vector<Vector3<tinyobj::real_t>> normal;
  std::vector<int> indices;

private:
  void computeFaceVertexData();
  void clearData();

  String mName;
  String mExtension;
  String mFolder;

  bool mMeshReady = false;
  UInt mNumTriangles = 0;

  Vector3<tinyobj::real_t> mAABBMin;
  Vector3<tinyobj::real_t> mAABBMax;

  Vector<tinyobj::real_t> mFaces;
  Vector<tinyobj::real_t> mVertices;
  Vector<tinyobj::real_t> mNormals;
  Vector<tinyobj::real_t> mTexCoord2D;
  Vector<tinyobj::real_t> mFaceVertices;
  Vector<tinyobj::real_t> mFaceVertexNormals;
  Vector<tinyobj::real_t> mFaceVertexColors;
  Vector<tinyobj::real_t> mFaceVertexTexCoord2D;
};

typedef SharedPtr<MeshObjectLoader> MeshObjectLoaderPtr;

#endif
