/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-07-26 10:44:51
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-07-26 11:09:38
 * @FilePath: \XProtoSphere\xprotosphere\src\main.cpp
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
// clang-format off
#include<root_directory.h>
#include <kiri_cuda_utils.h>
#include <geo/geo_particle_generator.h>
#include <abc/particles-alembic-manager.h>
#include <filesystem>
#include <kiri_pbs_cuda/system/cuda_protosphere_system.cuh>
#include <cuda/cuda_helper.h>
// clang-format on

using namespace KIRI;

int main(int argc, char *argv[]) {

  if (argc != 8) {
    std::cout << "Usage: " << argv[0]
              << "-input_model_name -input_model_scale -distribution_type "
                 "-protosphere_max_iter "
                 "-enable_dem_relax -overlap_ratio_thresold -dsr_max_iter"
              << " " << argc << std::endl;
    return 1;
  }

  String input_model = argv[1];
  float input_model_scale = std::stof(argv[2]);
  String distribution_type = argv[3];
  int protosphere_max_iter = std::stoi(argv[4]);
  bool enable_dem_relax = static_cast<bool>(std::stoi(argv[5]));
  float overlap_ratio = std::stof(argv[6]);
  int dsr_max_iter = std::stoi(argv[7]);

  if (!enable_dem_relax) {
    overlap_ratio = 0.f;
    dsr_max_iter = 0;
  }

  // predefined model scale
  if (input_model_scale == 0.f)
    input_model_scale = 0.8f;

  // setup path
  String export_abc_pth = String(EXPORT_PATH) + "abc/" + input_model;
  String export_pos_abc = export_abc_pth + "/" + "pos" + ".abc";
  String export_scale_abc = export_abc_pth + "/" + "scale" + ".abc";
  String export_color_abc = export_abc_pth + "/" + "color" + ".abc";

  String export_log_pth = String(EXPORT_PATH) + "log/";
  String export_log_file = export_log_pth + "/" + input_model + ".txt";

  std::error_code ec;
  std::filesystem::create_directories(export_abc_pth, ec);
  std::filesystem::create_directories(export_log_pth, ec);

  // abc exporter
  auto abc_pos_data = std::make_shared<ParticlesAlembicManager>(
      export_pos_abc, 0.01, "particle_pos");
  auto abc_scale_data = std::make_shared<ParticlesAlembicManager>(
      export_scale_abc, 0.01, "particle_scale");
  auto abc_cd_data = std::make_shared<ParticlesAlembicManager>(
      export_color_abc, 0.01, "particle_cd");

  KiriLog::Init(export_log_file);

  KIRI_LOG_INFO(
      "Input Params: Model Name={0}, Model Scale={1}, ProtoSphereIterNum={2}, "
      "EnableDEMRelaxation={3}, OverlapRatioThreshold={4}",
      input_model, input_model_scale, protosphere_max_iter, enable_dem_relax,
      overlap_ratio);

  auto xprotosphere_particle_size = 0.002f;
  CUDA_PROTOSPHERE_PARAMS.decay = 0.01f;
  CUDA_PROTOSPHERE_PARAMS.error_rate = 0.01f;
  CUDA_PROTOSPHERE_PARAMS.max_iteration_num = 200;
  CUDA_PROTOSPHERE_PARAMS.relax_dt = 0.00005f;

  auto mesh3d =
      std::make_shared<GEO::MeshObject<float>>(input_model, input_model_scale);

  Vector3F boundingbox_min(mesh3d->gridData().getBMin()),
      boundingbox_max(mesh3d->gridData().getBMax());
  auto grid_data = mesh3d->gridData().getNCells();
  GridInfo info(CudaAxisAlignedBox<float3>(KiriToCUDA(boundingbox_min),
                                           KiriToCUDA(boundingbox_max)),
                mesh3d->gridData().getCellSize(),
                make_int3(grid_data.x, grid_data.y, grid_data.z));

  // predefined particle size distribution
  std::vector<float> radius_range;
  std::vector<float> radius_range_prob;

  if (distribution_type == "A") {
    radius_range.push_back(0.003);
    radius_range.push_back(0.006);
    radius_range.push_back(0.01);
    radius_range.push_back(0.015);

    radius_range_prob.push_back(0.9);
    radius_range_prob.push_back(0.09);
    radius_range_prob.push_back(0.01);
  } else if (distribution_type == "B") {

    radius_range.push_back(0.003);
    radius_range.push_back(0.004);
    radius_range.push_back(0.0045);
    radius_range.push_back(0.0055);
    radius_range.push_back(0.01);

    radius_range_prob.push_back(0.45);
    radius_range_prob.push_back(0.01);
    radius_range_prob.push_back(0.45);
    radius_range_prob.push_back(0.09);
  } else if (distribution_type == "C") {
    radius_range.push_back(0.003);
    radius_range.push_back(0.004);
    radius_range.push_back(0.005);
    radius_range.push_back(0.006);
    radius_range.push_back(0.007);
    radius_range.push_back(0.008);
    radius_range.push_back(0.01);

    radius_range_prob.push_back(0.3);
    radius_range_prob.push_back(0.01);
    radius_range_prob.push_back(0.3);
    radius_range_prob.push_back(0.01);
    radius_range_prob.push_back(0.3);
    radius_range_prob.push_back(0.08);
  } else {
    radius_range.push_back(0.003);
    radius_range.push_back(0.006);
    radius_range.push_back(0.01);
    radius_range.push_back(0.015);

    radius_range_prob.push_back(0.9);
    radius_range_prob.push_back(0.09);
    radius_range_prob.push_back(0.01);
  }

  // recompute dem relax dt
  CUDA_PROTOSPHERE_PARAMS.relax_dt =
      0.5f * radius_range[0] / std::sqrtf(1e5f / 2700.f);

  std::random_device engine;
  std::mt19937 gen(engine());
  std::piecewise_constant_distribution<float> pcdis{
      std::begin(radius_range), std::end(radius_range),
      std::begin(radius_range_prob)};

  auto generator3d = std::make_shared<GEO::ParticleGenerator<3, float>>(
      xprotosphere_particle_size, 1.f, 0.1f);
  generator3d->GenerateByMeshObject(mesh3d);
  auto data = generator3d->particles();

  Vec_Float3 pos_array;
  Vec_Float3 color_array;
  Vec_Float radius_array;
  Vec_Float target_radius_array;
  Vec_Float3 closest_points_array;

  for (auto i = 0; i < data.size(); i++) {
    pos_array.emplace_back(make_float3(data[i].x, data[i].y, data[i].z));
    color_array.emplace_back(make_float3(0.f, 0.f, 1.f));
    radius_array.emplace_back(mesh3d->sdfStep() / 2.f);
    target_radius_array.emplace_back(pcdis(gen));
  }

  auto closest_data = mesh3d->sdfClosestPointData();
  for (auto i = 0; i < closest_data.size(); i++) {
    closest_points_array.emplace_back(
        make_float3(closest_data[i].x, closest_data[i].y, closest_data[i].z));
  }

  ProtoSphereVolumeData volume_data;
  volume_data.col = color_array;
  volume_data.pos = pos_array;
  volume_data.radius = radius_array;
  volume_data.targetRadius = target_radius_array;

  auto protosphere_particles = std::make_shared<CudaProtoSphereParticles>(
      volume_data.pos, volume_data.col, volume_data.radius,
      volume_data.targetRadius, info, mesh3d->sdfData(), closest_points_array,
      radius_range[radius_range.size() - 1], radius_range[0]);

  KIRI_LOG_DEBUG("info grid size={0}; sdf size={1}; protospheres={2}",
                 info.GridSize.x * info.GridSize.y * info.GridSize.z,
                 mesh3d->sdfData().size(), pos_array.size());

  auto solver =
      std::make_shared<CudaProtoSphereSampler>(protosphere_particles->size());

  auto system = std::make_shared<CudaProtoSphereSystem>(
      protosphere_particles, solver, radius_range, radius_range_prob,
      mesh3d->meshData()->volume(), overlap_ratio, xprotosphere_particle_size,
      protosphere_max_iter, enable_dem_relax, dsr_max_iter);

  KiriTimer timer;
  while (true) {
    auto current_insert_step = system->GetInsertStepNum();

    if (system->GetDEMRelaxationEnable()) {

      if (system->GetSystemStatus() == DEM_RELAXATION_FINISH) {
        // KIRI_LOG_DEBUG("Insert ProtoSphere and Write to File!!");
        abc_pos_data->SubmitCurrentStatusFloat3(system->GetInsertedPositions(),
                                                system->GetInsertedSize());

        abc_scale_data->SubmitCurrentStatusFloat(

            system->GetInsertedRadius(), system->GetInsertedSize());

        abc_cd_data->SubmitCurrentStatusFloat3(system->GetInsertedColors(),
                                               system->GetInsertedSize());

        auto inserted_time = timer.Elapsed();
        KIRI_LOG_INFO("Current Insert Step={0}, Elapsed Time={1}",
                      current_insert_step, inserted_time);
      }
    } else {
      if (system->GetSystemStatus() == XPROTOTYPE_INSERTED) {
        // KIRI_LOG_DEBUG("Insert ProtoSphere and Write to File!!");
        abc_pos_data->SubmitCurrentStatusFloat3(system->GetInsertedPositions(),
                                                system->GetInsertedSize());

        abc_scale_data->SubmitCurrentStatusFloat(

            system->GetInsertedRadius(), system->GetInsertedSize());

        abc_cd_data->SubmitCurrentStatusFloat3(system->GetInsertedColors(),
                                               system->GetInsertedSize());

        auto inserted_time = timer.Elapsed();
        KIRI_LOG_INFO("Current Insert Step={0}, Elapsed Time={1}",
                      current_insert_step, inserted_time);
      }
    }

    if (system->GetSystemStatus() == XPROTOSPHERE_FINISH) {

      auto inserted_time = timer.Elapsed();
      KIRI_LOG_INFO("Current Insert Step={0}, Elapsed Time={1}",
                    current_insert_step, inserted_time);

      // export
      KiriCudaUtils::ExportBgeoFileFromGPU(
          "protosphere", input_model, system->GetInsertedPositions(),
          system->GetInsertedRadius(), system->GetInsertedSize());

      // position
      KiriCudaUtils::ExportCSVFileFromGPU(
          "protosphere", input_model + "_pos_and_size", system->GetInsertedPositions(),system->GetInsertedRadius(),
          system->GetInsertedSize());

      // radius & target radius
      KiriCudaUtils::ExportCSVFileFromGPU(
          "protosphere", input_model + "_radius", system->GetInsertedRadius(),
          system->GetInsertedSize());

      KiriCudaUtils::ExportCSVFileFromGPU(
          "protosphere", input_model + "_target_radius",
          system->GetInsertedTargetRadius(), system->GetInsertedSize());

      break;
    }

    system->UpdateSystem();
  }

  return 0;
}
