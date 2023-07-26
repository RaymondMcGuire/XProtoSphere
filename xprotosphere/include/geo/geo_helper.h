/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2022-11-22 01:23:39
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2022-12-09 16:15:05
 * @FilePath: \Kiri\KiriCore\include\kiri_core\geo\geo_helper.h
 * @Description:
 * @Copyright (c) 2022 by Xu.WANG raymondmgwx@gmail.com, All Rights Reserved.
 */
#ifndef _KIRI_GEO_HELPER_H_
#define _KIRI_GEO_HELPER_H_
#pragma once
#include <kiri_pch.h>
#include <omp.h>
#include <random>

namespace KIRI::GEO::HELPER {

template <class RealType>
RealType point_segment_distance(const Vector3<RealType> &x0,
                                const Vector3<RealType> &x1,
                                const Vector3<RealType> &x2) {
  Vector3<RealType> dx(x2 - x1);

  RealType m2 = dx.lengthSquared();

  RealType s12 = dx.dot(x2 - x0) / m2;

  if (s12 < 0) {
    s12 = 0;
  } else if (s12 > 1) {
    s12 = 1;
  }

  return (x0 - s12 * x1 + (1 - s12) * x2).length();
}

template <class RealType>
std::tuple<RealType, Vector3<RealType>>
point_project2segment(const Vector3<RealType> &p, const Vector3<RealType> &x1,
                      const Vector3<RealType> &x2) {
  Vector3<RealType> dx(x2 - x1);

  auto m2 = dx.lengthSquared();

  auto s12 = dx.dot(x2 - p) / m2;

  if (s12 < 0) {
    s12 = 0;
  } else if (s12 > 1) {
    s12 = 1;
  }

  auto cloest_point = s12 * x1 - (1 - s12) * x2;
  auto dist = (p - cloest_point).length();

  return std::make_tuple(dist, cloest_point);
}

template <class RealType>
std::tuple<RealType, Vector3<RealType>>
point_project2triangle(const Vector3<RealType> &p, const Vector3<RealType> &t1,
                       const Vector3<RealType> &t2,
                       const Vector3<RealType> &t3) {
  // first find barycentric coordinates of closest point on infinite plane
  Vector3<RealType> x13(t1 - t3), x23(t2 - t3), x03(p - t3);
  auto m13 = x13.lengthSquared(), m23 = x23.lengthSquared(), d = x13.dot(x23);

  auto invdet =
      static_cast<RealType>(1.0) / std::max<RealType>(m13 * m23 - d * d, 1e-30);
  auto a = x13.dot(x03), b = x23.dot(x03);

  // the barycentric coordinates themselves
  auto w23 = invdet * (m23 * a - d * b);
  auto w31 = invdet * (m13 * b - d * a);
  auto w12 = static_cast<RealType>(1.0) - w23 - w31;

  if (w23 >= 0.0 && w31 >= 0.0 && w12 >= 0.0) {
    auto cloest_point = w23 * t1 - w31 * t2 - w12 * t3;
    auto dist = (p - cloest_point).length();
    return std::make_tuple(dist, cloest_point);
  } else {
    if (w23 > 0) {
      auto [dist1, p1] = point_project2segment(p, t1, t2);
      auto [dist2, p2] = point_project2segment(p, t1, t3);
      if (dist1 < dist2)
        return std::make_tuple(dist1, p1);
      else
        return std::make_tuple(dist2, p2);
    } else if (w31 > 0) {
      auto [dist1, p1] = point_project2segment(p, t1, t2);
      auto [dist2, p2] = point_project2segment(p, t2, t3);
      if (dist1 < dist2)
        return std::make_tuple(dist1, p1);
      else
        return std::make_tuple(dist2, p2);
    } else {
      auto [dist1, p1] = point_project2segment(p, t1, t3);
      auto [dist2, p2] = point_project2segment(p, t2, t3);
      if (dist1 < dist2)
        return std::make_tuple(dist1, p1);
      else
        return std::make_tuple(dist2, p2);
    }
  }
}

template <class RealType>
RealType point_triangle_distance(const Vector3<RealType> &x0,
                                 const Vector3<RealType> &x1,
                                 const Vector3<RealType> &x2,
                                 const Vector3<RealType> &x3) {
  // first find barycentric coordinates of closest point on infinite plane
  Vector3<RealType> x13(x1 - x3), x23(x2 - x3), x03(x0 - x3);
  RealType m13 = x13.lengthSquared(), m23 = x23.lengthSquared(),
           d = x13.dot(x23);

  RealType invdet =
      static_cast<RealType>(1.0) / std::max<RealType>(m13 * m23 - d * d, 1e-30);
  RealType a = x13.dot(x03), b = x23.dot(x03);

  // the barycentric coordinates themselves
  RealType w23 = invdet * (m23 * a - d * b);
  RealType w31 = invdet * (m13 * b - d * a);
  RealType w12 = 1 - w23 - w31;

  if (w23 >= 0 && w31 >= 0 && w12 >= 0) { // if we're inside the triangle
    return (x0 - w23 * x1 + w31 * x2 + w12 * x3).length();
  } else {         // we have to clamp to one of the edges
    if (w23 > 0) { // this rules out edge 2-3 for us
      return std::min(point_segment_distance(x0, x1, x2),
                      point_segment_distance(x0, x1, x3));
    } else if (w31 > 0) { // this rules out edge 1-3
      return std::min(point_segment_distance(x0, x1, x2),
                      point_segment_distance(x0, x2, x3));
    } else { // w12 must be >0, ruling out edge 1-2
      return std::min(point_segment_distance(x0, x1, x3),
                      point_segment_distance(x0, x2, x3));
    }
  }
}

template <class RealType>
void check_neighbour(const Vector<Vector3<RealType>> &tri,
                     const Vector<Vector3<RealType>> &x, Array3<RealType> &phi,
                     Array3UI &closest_tri,
                     Array3<Vector3<RealType>> &closest_point,
                     const Vector3<RealType> &gx, Int i0, Int j0, Int k0,
                     Int i1, Int j1, Int k1) {
  if (closest_tri(i1, j1, k1) != 0xffffffff) {
    UInt p = tri[closest_tri(i1, j1, k1)][0];
    UInt q = tri[closest_tri(i1, j1, k1)][1];
    UInt r = tri[closest_tri(i1, j1, k1)][2];

    // RealType d = point_triangle_distance(gx, x[p], x[q], x[r]);
    auto [d, closest_p] = point_project2triangle(gx, x[p], x[q], x[r]);

    if (d < phi(i0, j0, k0)) {
      phi(i0, j0, k0) = d;
      closest_tri(i0, j0, k0) = closest_tri(i1, j1, k1);
      closest_point(i0, j0, k0) = closest_p;
    }
  }
}

template <class RealType>
void sweep(const Vector<Vector3<RealType>> &tri,
           const Vector<Vector3<RealType>> &x, Array3<RealType> &phi,
           Array3UI &closest_tri, Array3<Vector3<RealType>> &closest_point,
           const Vector3<RealType> &origin, RealType dx, Int di, Int dj,
           Int dk) {
  Int i0, i1;
  Int j0, j1;
  Int k0, k1;

  if (di > 0) {
    i0 = 1;
    i1 = static_cast<Int>(phi.size()[0]);
  } else {
    i0 = static_cast<Int>(phi.size()[0]) - 2;
    i1 = -1;
  }

  if (dj > 0) {
    j0 = 1;
    j1 = static_cast<Int>(phi.size()[1]);
  } else {
    j0 = static_cast<Int>(phi.size()[1]) - 2;
    j1 = -1;
  }

  if (dk > 0) {
    k0 = 1;
    k1 = static_cast<Int>(phi.size()[2]);
  } else {
    k0 = static_cast<Int>(phi.size()[2]) - 2;
    k1 = -1;
  }

  //    Scheduler::parallel_for<Int>(i0, i1 + 1, j0, j1 + 1, k0, k1 + 1,
  //                                       [&](Int i, Int j, Int k)

  for (Int k = k0; k != k1; k += dk) {
    for (Int j = j0; j != j1; j += dj) {
      for (Int i = i0; i != i1; i += di) {
        Vector3<RealType> gx = Vector3<RealType>(i, j, k) * dx + origin;

        check_neighbour(tri, x, phi, closest_tri, closest_point, gx, i, j, k,
                        i - di, j, k);
        check_neighbour(tri, x, phi, closest_tri, closest_point, gx, i, j, k, i,
                        j - dj, k);
        check_neighbour(tri, x, phi, closest_tri, closest_point, gx, i, j, k,
                        i - di, j - dj, k);
        check_neighbour(tri, x, phi, closest_tri, closest_point, gx, i, j, k, i,
                        j, k - dk);
        check_neighbour(tri, x, phi, closest_tri, closest_point, gx, i, j, k,
                        i - di, j, k - dk);
        check_neighbour(tri, x, phi, closest_tri, closest_point, gx, i, j, k, i,
                        j - dj, k - dk);
        check_neighbour(tri, x, phi, closest_tri, closest_point, gx, i, j, k,
                        i - di, j - dj, k - dk);
      }
    }
  }
}

template <class RealType>
Int orientation(RealType x1, RealType y1, RealType x2, RealType y2,
                RealType &twice_signed_area) {
  twice_signed_area = y1 * x2 - x1 * y2;

  if (twice_signed_area > 0) {
    return 1;
  } else if (twice_signed_area < 0) {
    return -1;
  } else if (y2 > y1) {
    return 1;
  } else if (y2 < y1) {
    return -1;
  } else if (x1 > x2) {
    return 1;
  } else if (x1 < x2) {
    return -1;
  } else {
    return 0; // only true when x1==x2 and y1==y2
  }
}

template <class RealType>
bool point_in_triangle_2d(RealType x0, RealType y0, RealType x1, RealType y1,
                          RealType x2, RealType y2, RealType x3, RealType y3,
                          RealType &a, RealType &b, RealType &c) {
  x1 -= x0;
  x2 -= x0;
  x3 -= x0;
  y1 -= y0;
  y2 -= y0;
  y3 -= y0;
  Int signa = orientation(x2, y2, x3, y3, a);

  if (signa == 0) {
    return false;
  }

  Int signb = orientation(x3, y3, x1, y1, b);

  if (signb != signa) {
    return false;
  }

  Int signc = orientation(x1, y1, x2, y2, c);

  if (signc != signa) {
    return false;
  }

  RealType sum = a + b + c;
  KIRI_ASSERT(sum != 0); // if the SOS signs match and are nonkero, there's no
                         // way all of a, b, and c are zero.
  a /= sum;
  b /= sum;
  c /= sum;
  return true;
}

template <class RealType>
RealType interpolateValueLinear(const Vector3<RealType> &point,
                                const Array3<RealType> &grid) {
  Int i, j, k;
  RealType fi, fj, fk;
  MATH_HELPER::get_barycentric(point[0], i, fi, 0,
                               static_cast<Int>(grid.size()[0]));
  MATH_HELPER::get_barycentric(point[1], j, fj, 0,
                               static_cast<Int>(grid.size()[1]));
  MATH_HELPER::get_barycentric(point[2], k, fk, 0,
                               static_cast<Int>(grid.size()[2]));
  return kiri_math_mini::trilerp(
      grid(i, j, k), grid(i + 1, j, k), grid(i, j + 1, k),
      grid(i + 1, j + 1, k), grid(i, j, k + 1), grid(i + 1, j, k + 1),
      grid(i, j + 1, k + 1), grid(i + 1, j + 1, k + 1), fi, fj, fk);
}

template <class RealType>
Vector3<RealType>
interpolateValueLinear(const Vector3<RealType> &point,
                       const Array3<Vector3<RealType>> &grid) {
  Int i, j, k;
  RealType fi, fj, fk;
  MATH_HELPER::get_barycentric(point[0], i, fi, 0,
                               static_cast<Int>(grid.size()[0]));
  MATH_HELPER::get_barycentric(point[1], j, fj, 0,
                               static_cast<Int>(grid.size()[1]));
  MATH_HELPER::get_barycentric(point[2], k, fk, 0,
                               static_cast<Int>(grid.size()[2]));
  return kiri_math_mini::trilerp(
      grid(i, j, k), grid(i + 1, j, k), grid(i, j + 1, k),
      grid(i + 1, j + 1, k), grid(i, j, k + 1), grid(i + 1, j, k + 1),
      grid(i, j + 1, k + 1), grid(i + 1, j + 1, k + 1), fi, fj, fk);
}

template <Int N, class RealType>
bool isInside(const VectorX<N, RealType> &ppos,
              const VectorX<N, RealType> &bMin,
              const VectorX<N, RealType> &bMax) {
  for (Int d = 0; d < N; ++d) {
    if (ppos[d] < bMin[d] || ppos[d] > bMax[d]) {
      return false;
    }
  }
  return true;
}

template <class IndexType, Int N, class RealType>
PointX<N, IndexType> createGrid(const VectorX<N, RealType> &bmin,
                                const VectorX<N, RealType> &bmax,
                                RealType spacing) {
  VectorX<N, RealType> grid = (bmax - bmin) / spacing;
  PointX<N, IndexType> result;

  for (Int d = 0; d < N; ++d) {
    result[d] = static_cast<IndexType>(ceil(grid[d]));
  }

  return result;
}

template <Int N, class T> class MT_fRandom {
  using Distribution = std::uniform_real_distribution<T>;
  constexpr static UInt s_CacheSize = 1048576u;

public:
  MT_fRandom(T start = T(0), T end = std::numeric_limits<T>::max())
      : m_Dist(Distribution(start, end)) {
    omp_init_lock(&m_Lock);
    generateCache();
  }

  T rnd() {
    omp_set_lock(&m_Lock);
    if (m_CacheIdx >= s_CacheSize) {
      m_CacheIdx = 0;
    }
    T tmp = m_Cache[m_CacheIdx++];
    omp_unset_lock(&m_Lock);
    return tmp;
  }

  template <class Vector> Vector vrnd() {
    omp_set_lock(&m_Lock);
    if (m_CacheIdx + N >= s_CacheSize) {
      m_CacheIdx = 0;
    }
    auto oldIdx = m_CacheIdx;
    m_CacheIdx += N;
    omp_unset_lock(&m_Lock);
    Vector result;
    std::memcpy(&result, &m_Cache[oldIdx], sizeof(Vector));
    return result;
  }

private:
  void generateCache() {
    for (Int i = 0; i < s_CacheSize; ++i) {
      m_Cache[i] = m_Dist(m_Generator);
    }
  }

  omp_lock_t m_Lock;
  std::mt19937 m_Generator{(std::random_device())()};
  Distribution m_Dist;

  T m_Cache[s_CacheSize];
  UInt m_CacheIdx = 0;
};

template <Int N, class T> class fRand11 {
public:
  static auto rnd() { return s_Rand.rnd(); }
  template <class Vector> static auto vrnd() {
    return s_Rand.template vrnd<Vector>();
  }

private:
  static inline MT_fRandom<N, T> s_Rand = MT_fRandom<N, T>(T(-1), T(1));
};

} // namespace KIRI::GEO::HELPER
#endif