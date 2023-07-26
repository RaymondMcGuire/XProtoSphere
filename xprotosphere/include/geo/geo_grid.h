/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-19 00:31:31
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-11-21 22:45:43
 * @FilePath: \Kiri\KiriCore\include\kiri_core\geo\geo_grid.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */

#ifndef _KIRI_GEO_GRID_H_
#define _KIRI_GEO_GRID_H_
#pragma once
#include <kiri_pch.h>

namespace KIRI::GEO {
template <class RealType> class GeoGrid {
public:
  GeoGrid() = default;
  GeoGrid(const Vector3<RealType> &bMin, const Vector3<RealType> &bMax,
          RealType CellSize)
      : mBMin(bMin), mBMax(bMax) {
    SetCellSize(CellSize);
  }

  void SetGrid(const Vector3<RealType> &bMin, const Vector3<RealType> &bMax,
               RealType CellSize);
  void SetCellSize(RealType CellSize);

  const auto &getBMin() const noexcept { return mBMin; }
  const auto &getBMax() const noexcept { return mBMax; }

  const auto &getNCells() const noexcept { return mNCells; }
  const auto &getNNodes() const noexcept { return mNNodes; }
  auto getNTotalCells() const noexcept { return mNTotalCells; }
  auto getNTotalNodes() const noexcept { return mNTotalNodes; }

  auto getCellSize() const noexcept { return mCellSize; }
  auto getInvCellSize() const noexcept { return mInvCellSize; }
  auto getHalfCellSize() const noexcept { return mHalfCellSize; }
  auto getCellSizeSquared() const noexcept { return mCellSizeSqr; }

  // Particle processing
  auto getGridCoordinate(const Vector3<RealType> &ppos) const {
    return (ppos - mBMin) / mCellSize;
  }

protected:
  Vector3<RealType> mBMin;
  Vector3<RealType> mBMax;

  Vector3<RealType> mNCells;
  Vector3<RealType> mNNodes;

  UInt mNTotalCells = 1u;
  UInt mNTotalNodes = 1u;
  RealType mCellSize = 1.0;
  RealType mInvCellSize = 1.0;
  RealType mHalfCellSize = 0.5;
  RealType mCellSizeSqr = 1.0;

  bool mbCellIdxNeedResize = false;
};

} // namespace KIRI::GEO

#include "inl/geo_grid-inl.h"
#endif
