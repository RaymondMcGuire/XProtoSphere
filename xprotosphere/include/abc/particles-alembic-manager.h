/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-02-19 15:45:28
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-02-19 23:43:33
 * @FilePath: \Kiri\KiriExamples\include\abc\particles-alembic-manager.h
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#ifndef PBF_PARTICLES_ALEMBIC_MANAGER_HPP
#define PBF_PARTICLES_ALEMBIC_MANAGER_HPP

#include <kiri_pbs_cuda/cuda_helper/helper_math.h>
#include <abc/alembic-manager-base.h>


class ParticlesAlembicManager : public AlembicManagerBase {
public:
  ParticlesAlembicManager(const std::string &output_file_path,
                          const double delta_time,
                          const std::string &object_name)
      : AlembicManagerBase(output_file_path, delta_time, object_name) {}

  void SubmitCurrentStatusFloat(float *radius_array, const uint num) {
    using namespace Alembic::Abc;
    using namespace Alembic::AbcGeom;

    uint f_bytes = num * sizeof(float);
    float *cpu_radius = (float *)malloc(f_bytes);
    cudaMemcpy(cpu_radius, radius_array, f_bytes, cudaMemcpyDeviceToHost);

    std::vector<float> radius_vector;
    radius_vector.assign(cpu_radius, cpu_radius + num);

    if (m_is_first) {
      SubmitCurrentStatusFirstTimeFloat(radius_vector, num);

      m_is_first = false;

      return;
    }

    std::vector<float> data;
    for (auto i = 0; i < num; i++) {
      data.emplace_back(radius_vector[i]);
      data.emplace_back(radius_vector[i]);
      data.emplace_back(radius_vector[i]);
    }

    OPointsSchema::Sample sample;
    sample.setPositions(P3fArraySample((const V3f *)&data.front(), num));
    m_points.getSchema().set(sample);
  }

  void SubmitCurrentStatusFloat3(float3 *position_array, const uint num) {
    using namespace Alembic::Abc;
    using namespace Alembic::AbcGeom;

    uint f3_bytes = num * sizeof(float3);
    float3 *cpu_positions = (float3 *)malloc(f3_bytes);

    cudaMemcpy(cpu_positions, position_array, f3_bytes, cudaMemcpyDeviceToHost);

    std::vector<float3> position_vector;
    position_vector.assign(cpu_positions, cpu_positions + num);

    if (m_is_first) {
      SubmitCurrentStatusFirstTimeFloat3(position_vector, num);

      m_is_first = false;

      return;
    }

    const V3fArraySample position_array_sample(
        reinterpret_cast<const V3f *>(&position_vector.front()), num);

    OPointsSchema::Sample sample;
    sample.setPositions(position_array_sample);
    m_points.getSchema().set(sample);
  }

  void SubmitCurrentStatusPositionWithSection(float3 *position_array,
                                              const uint num) {
    using namespace Alembic::Abc;
    using namespace Alembic::AbcGeom;

    uint f3_bytes = num * sizeof(float3);
    float3 *cpu_positions = (float3 *)malloc(f3_bytes);

    cudaMemcpy(cpu_positions, position_array, f3_bytes, cudaMemcpyDeviceToHost);

    std::vector<float3> position_vector;
    for (auto i = 0; i < num; i++) {
      if (cpu_positions[i].z >= 0.f)
        position_vector.emplace_back(cpu_positions[i]);
    }

    auto current_num = position_vector.size();

    if (m_is_first) {
      SubmitCurrentStatusFirstTimeFloat3(position_vector, current_num);

      m_is_first = false;

      return;
    }

    const V3fArraySample position_array_sample(
        reinterpret_cast<const V3f *>(&position_vector.front()), current_num);

    OPointsSchema::Sample sample;
    sample.setPositions(position_array_sample);
    m_points.getSchema().set(sample);
  }

  void SubmitCurrentStatusColorWithSection(float3 *position_array,
                                           float3 *color_array,
                                           const uint num) {
    using namespace Alembic::Abc;
    using namespace Alembic::AbcGeom;

    uint f3_bytes = num * sizeof(float3);
    float3 *cpu_positions = (float3 *)malloc(f3_bytes);
    float3 *cpu_colors = (float3 *)malloc(f3_bytes);
    cudaMemcpy(cpu_positions, position_array, f3_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_colors, color_array, f3_bytes, cudaMemcpyDeviceToHost);

    std::vector<float3> color_vector;
    for (auto i = 0; i < num; i++) {
      if (cpu_positions[i].z >= 0.f)
        color_vector.emplace_back(cpu_colors[i]);
    }

    auto current_num = color_vector.size();

    if (m_is_first) {
      SubmitCurrentStatusFirstTimeFloat3(color_vector, current_num);

      m_is_first = false;

      return;
    }

    const V3fArraySample position_array_sample(
        reinterpret_cast<const V3f *>(&color_vector.front()), current_num);

    OPointsSchema::Sample sample;
    sample.setPositions(position_array_sample);
    m_points.getSchema().set(sample);
  }

  void SubmitCurrentStatusRadiusWithSection(float3 *position_array,
                                            float *radius_array,
                                            const uint num) {
    using namespace Alembic::Abc;
    using namespace Alembic::AbcGeom;

    uint f_bytes = num * sizeof(float);
    float *cpu_radius = (float *)malloc(f_bytes);
    cudaMemcpy(cpu_radius, radius_array, f_bytes, cudaMemcpyDeviceToHost);

    uint f3_bytes = num * sizeof(float3);
    float3 *cpu_positions = (float3 *)malloc(f3_bytes);
    cudaMemcpy(cpu_positions, position_array, f3_bytes, cudaMemcpyDeviceToHost);

    std::vector<float> radius_vector;
    for (auto i = 0; i < num; i++) {
      if (cpu_positions[i].z >= 0.f)
        radius_vector.emplace_back(cpu_radius[i]);
    }

    auto current_num = radius_vector.size();

    if (m_is_first) {
      SubmitCurrentStatusFirstTimeFloat(radius_vector, current_num);

      m_is_first = false;

      return;
    }

    std::vector<float> data;
    for (auto i = 0; i < current_num; i++) {
      data.emplace_back(radius_vector[i]);
      data.emplace_back(radius_vector[i]);
      data.emplace_back(radius_vector[i]);
    }

    OPointsSchema::Sample sample;
    sample.setPositions(
        P3fArraySample((const V3f *)&data.front(), current_num));
    m_points.getSchema().set(sample);
  }

private:
  void SubmitCurrentStatusFirstTimeFloat(std::vector<float> radius_array,
                                         const uint num) {
    using namespace Alembic::Abc;
    using namespace Alembic::AbcGeom;

    const std::vector<std::int32_t> counts(num, 1);

    std::vector<uint64_t> index_buffer(num);
    for (std::size_t elem_index = 0; elem_index < num; ++elem_index) {
      index_buffer[elem_index] = elem_index;
    }

    const UInt64ArraySample index_array_sample(index_buffer.data(), num);

    std::vector<float> data;
    for (auto i = 0; i < num; i++) {
      data.emplace_back(radius_array[i]);
      data.emplace_back(radius_array[i]);
      data.emplace_back(radius_array[i]);
    }

    OPointsSchema::Sample sample;
    sample.setIds(index_array_sample);
    sample.setPositions(P3fArraySample((const V3f *)&data.front(), num));

    m_points.getSchema().set(sample);
  }

  void
  SubmitCurrentStatusFirstTimeFloat3(const std::vector<float3> position_array,
                                     const uint num) {
    using namespace Alembic::Abc;
    using namespace Alembic::AbcGeom;

    const std::vector<std::int32_t> counts(num, 1);

    std::vector<uint64_t> index_buffer(num);
    for (std::size_t elem_index = 0; elem_index < num; ++elem_index) {
      index_buffer[elem_index] = elem_index;
    }

    const V3fArraySample position_array_sample(
        reinterpret_cast<const V3f *>(&position_array.front()), num);
    const UInt64ArraySample index_array_sample(index_buffer.data(), num);

    OPointsSchema::Sample sample;
    sample.setIds(index_array_sample);
    sample.setPositions(position_array_sample);

    m_points.getSchema().set(sample);
  }
};

typedef SharedPtr<ParticlesAlembicManager> ParticlesAlembicManagerPtr;

#endif
