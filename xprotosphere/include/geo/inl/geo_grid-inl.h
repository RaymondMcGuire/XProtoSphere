/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-10-19 00:31:31
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-11-21 22:46:27
 * @FilePath: \Kiri\KiriCore\src\kiri_core\geo\geo_grid.cpp
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#include <geo/geo_grid.h>

namespace KIRI::GEO {
template <class RealType>
void GeoGrid<RealType>::SetGrid(const Vector3<RealType> &bMin,
                                const Vector3<RealType> &bMax,
                                RealType CellSize) {
  mBMin = bMin;
  mBMax = bMax;
  SetCellSize(CellSize);
}

template <class RealType>
void GeoGrid<RealType>::SetCellSize(RealType CellSize) {
  KIRI_ASSERT(CellSize > 0);
  mCellSize = CellSize;
  mInvCellSize = 1.0 / mCellSize;
  mHalfCellSize = 0.5 * mCellSize;
  mCellSizeSqr = mCellSize * mCellSize;
  mNTotalCells = 1;

  for (Int i = 0; i < mNCells.size(); ++i) {
    mNCells[i] = static_cast<UInt>(ceil((mBMax[i] - mBMin[i]) / mCellSize));
    mNNodes[i] = mNCells[i] + 1u;

    mNTotalCells *= mNCells[i];
    mNTotalNodes *= mNNodes[i];
  }
  mbCellIdxNeedResize = true;
}
} // namespace KIRI::GEO