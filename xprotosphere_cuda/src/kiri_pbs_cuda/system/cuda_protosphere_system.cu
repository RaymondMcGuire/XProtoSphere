/***
 * @Author: Xu.WANG raymondmgwx@gmail.com
 * @Date: 2023-07-26 10:44:51
 * @LastEditors: Xu.WANG raymondmgwx@gmail.com
 * @LastEditTime: 2023-07-26 11:07:12
 * @FilePath:
 * \XProtoSphere\xprotosphere_cuda\src\kiri_pbs_cuda\system\cuda_protosphere_system.cu
 * @Description:
 * @Copyright (c) 2023 by Xu.WANG, All Rights Reserved.
 */
#include <kiri_pbs_cuda/system/cuda_base_system_gpu.cuh>
#include <kiri_pbs_cuda/system/cuda_protosphere_system.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <random>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#include <kiri_pbs_cuda/math/cuda_math_colormap.cuh>

template <typename T> struct power3_value {
  __host__ __device__ T operator()(const T &x) const { return x * x * x; }
};

namespace KIRI {

float bhattacharyya_distance(std::vector<float> p, std::vector<float> q) {
  float bc = 0.f;
  for (auto i = 0; i < p.size(); i++)
    bc += std::sqrt(p[i] * q[i]);

  return -std::log(bc);
}

float kl_divergence(std::vector<float> p, std::vector<float> q) {
  float divergence = 0;
  for (auto i = 0; i < p.size(); i++)
    divergence += p[i] * std::log(p[i] / q[i]);

  return divergence;
}

float jsd_distance(std::vector<float> p, std::vector<float> q) {
  auto n = p.size();
  std::vector<float> m(n);
  for (auto i = 0; i < n; i++)
    m[i] = (p[i] + q[i]) / 2.f;

  float dpm = kl_divergence(p, m);
  float dqm = kl_divergence(q, m);
  float distance = 0.5f * dpm + 0.5f * dqm;
  return distance;
}

CudaProtoSphereSystem::~CudaProtoSphereSystem() {}

CudaProtoSphereSystem::CudaProtoSphereSystem(
    CudaProtoSphereParticlesPtr &particles, CudaProtoSphereSamplerPtr &solver,
    const std::vector<float> radiusRange,
    const std::vector<float> radiusRangeProb, float boundaryVolume,
    float acceptOverlapRatio, float halfGridSize, int maxProtoSphereIteration,
    bool enableDemRelax, bool enableDistConstrain, int maxRelaxStepNum)
    : mParticles(std::move(particles)), mSolver(std::move(solver)),
      mCurrentRadiusRange(radiusRange),
      mCurrentRadiusRangeProb(radiusRangeProb),
      mPreDefinedRadiusRange(radiusRange),
      mPreDefinedRadiusRangeProb(radiusRangeProb),
      mBoundaryVolume(boundaryVolume), mAcceptOverlapRatio(acceptOverlapRatio),
      mHalfGridSize(halfGridSize),
      mMaxProtoSphereIterationNum(maxProtoSphereIteration),
      mEnableDEMRelax(enableDemRelax),mInsertDistConstrain(enableDistConstrain), mMaxRelaxStepNum(maxRelaxStepNum) {
  auto num = mParticles->size();
  mCudaGridSize = CuCeilDiv(mParticles->maxSize(), KIRI_CUBLOCKSIZE);

  // realloc generator
  if (mInsertDistConstrain)
    mCurrentRadiusRangeProb[0] = 0.f;

  mRadiusRangeCounter.resize(mPreDefinedRadiusRange.size() - 1, 0);
  this->EstimateIdealTotalNum();

  CUDA_BOUNDARY_PARAMS.lowest_point = mParticles->levelSetData().BBox.Min;
  CUDA_BOUNDARY_PARAMS.highest_point = mParticles->levelSetData().BBox.Max;
  CUDA_BOUNDARY_PARAMS.world_size =
      CUDA_BOUNDARY_PARAMS.highest_point - CUDA_BOUNDARY_PARAMS.lowest_point;
  CUDA_BOUNDARY_PARAMS.world_center =
      (CUDA_BOUNDARY_PARAMS.highest_point + CUDA_BOUNDARY_PARAMS.lowest_point) /
      2.f;
  CUDA_BOUNDARY_PARAMS.kernel_radius = mPreDefinedRadiusRange.back() * 4.f;
  CUDA_BOUNDARY_PARAMS.grid_size = make_int3(
      CUDA_BOUNDARY_PARAMS.world_size / CUDA_BOUNDARY_PARAMS.kernel_radius);

  mInsertedParticles = std::make_shared<CudaSphBNParticles>(
      mMaxSamplerNum, mParticles->levelSetData(), mParticles->sdfCPUData());
  mSearcher = std::make_shared<CudaGNSearcher>(
      CUDA_BOUNDARY_PARAMS.lowest_point, CUDA_BOUNDARY_PARAMS.highest_point,
      mMaxSamplerNum, CUDA_BOUNDARY_PARAMS.kernel_radius);

  // TODO move to function (init data)
  thrust::fill(thrust::device, mParticles->iterationNumPtr(),
               mParticles->iterationNumPtr() + num, 0);

  thrust::fill(thrust::device, mParticles->convergencePtr(),
               mParticles->convergencePtr() + num, 0);

  thrust::fill(thrust::device, mParticles->learningRatePtr(),
               mParticles->learningRatePtr() + num, 1.f);

  // start prototype searching
  mSystemStatus = XPROTOTYPE_SEARCHING;
}

void CudaProtoSphereSystem::InsertParticles() {
    if (!mConvergence)
        return;

    // Transform gpu data to cpu
    auto num = mParticles->size();
    thrust::sort_by_key(thrust::device, mParticles->currentRadiusPtr(),
                       mParticles->currentRadiusPtr() + num,
                       thrust::make_zip_iterator(thrust::make_tuple(
                           mParticles->posPtr(), mParticles->targetRadiusPtr())),
                       thrust::greater<float>());

    thrust::device_vector<float> radius(mParticles->currentRadiusPtr(),
                                      mParticles->currentRadiusPtr() + num);
    thrust::device_vector<float> target_radius(
        mParticles->targetRadiusPtr(), mParticles->targetRadiusPtr() + num);
    thrust::device_vector<float3> pos(mParticles->posPtr(),
                                    mParticles->posPtr() + num);

    thrust::host_vector<float3> poshost = pos;
    thrust::host_vector<float> radius_host = radius;
    thrust::host_vector<float> target_radius_host = target_radius;

    thrust::host_vector<float3> inserted_sphere_pos;
    thrust::host_vector<float> inserted_sphere_radius;
    thrust::host_vector<float> inserted_sphere_target_radius;
    thrust::host_vector<float3> inserted_sphere_color;

    std::random_device seed_gen;
    std::mt19937 color_engine(seed_gen());

    // Track successful insertions per range for this iteration
    std::vector<int> current_iteration_insertions(mRadiusRangeCounter.size(), 0);

    for (auto i = 0; i < radius_host.size(); i++) {
        if (radius_host[i] < mParticles->minRadius() ||
            radius_host[i] > mParticles->maxRadius())
            continue;

        // Check overlapping with existing particles
        auto overlapping = false;
        for (auto j = 0; j < inserted_sphere_pos.size(); j++) {
            auto other_pos = inserted_sphere_pos[j];
            auto other_radius = inserted_sphere_radius[j];
            auto dist =
                length(other_pos - poshost[i]) - (radius_host[i] + other_radius);
            if (dist < 0 && abs(dist) > mAcceptOverlapRatio * radius_host[i]) {
                overlapping = true;
                break;
            }
        }

        // Insert new spheres if conditions are met
        if (!overlapping && abs(radius_host[i] - target_radius_host[i]) <
                               target_radius_host[i] * 0.01f) {
            bool can_insert = true;
            int range_index = -1;

            // Determine which range this particle belongs to
            for (auto ri = 0; ri < mPreDefinedRadiusRange.size() - 1; ri++) {
                auto rj = ri + 1;
                if (radius_host[i] > mPreDefinedRadiusRange[ri] &&
                    radius_host[i] <= mPreDefinedRadiusRange[rj]) {
                    range_index = ri;
                    
                    // Check if we've already reached the target for this range
                    if (mInsertDistConstrain && 
                        mRadiusRangeCounter[ri] >= mRemainSamples[ri]) {
                        can_insert = false;
                    }
                    break;
                }
            }

            if (can_insert && range_index >= 0) {
                inserted_sphere_pos.push_back(poshost[i]);
                inserted_sphere_radius.push_back(radius_host[i]);
                inserted_sphere_target_radius.push_back(target_radius_host[i]);

                auto cur_color =
                    GetColor<float>(
                        (radius_host[i] - mParticles->minRadius()) /
                            (mParticles->maxRadius() - mParticles->minRadius()),
                        ColormapType::Parula)
                        .Data();
                inserted_sphere_color.push_back(cur_color);
                
                mRMSPE +=
                    powf((radius_host[i] - target_radius_host[i]) / target_radius_host[i],
                         2.f);

                // Update counters
                mRadiusRangeCounter[range_index]++;
                current_iteration_insertions[range_index]++;
            }
        }
    }

    // Update inserted particles data
    auto inserted_size = mInsertedParticles->size();
    KIRI_LOG_INFO("New Insert Particles Num={0}; Current Particles Num={1};",
                  inserted_sphere_pos.size(), inserted_size);

    // Copy new particles to GPU
    KIRI_CUCALL(cudaMemcpy(
        mInsertedParticles->posPtr() + inserted_size, &inserted_sphere_pos[0],
        sizeof(float3) * inserted_sphere_pos.size(), cudaMemcpyHostToDevice));

    KIRI_CUCALL(cudaMemcpy(mInsertedParticles->radiusPtr() + inserted_size,
                          &inserted_sphere_radius[0],
                          sizeof(float) * inserted_sphere_radius.size(),
                          cudaMemcpyHostToDevice));

    KIRI_CUCALL(cudaMemcpy(mInsertedParticles->targetRadiusPtr() + inserted_size,
                          &inserted_sphere_target_radius[0],
                          sizeof(float) * inserted_sphere_target_radius.size(),
                          cudaMemcpyHostToDevice));

    KIRI_CUCALL(cudaMemcpy(
        mInsertedParticles->colorPtr() + inserted_size, &inserted_sphere_color[0],
        sizeof(float3) * inserted_sphere_color.size(), cudaMemcpyHostToDevice));

    mNumOfSampledPoints += inserted_sphere_pos.size();
    mInsertedParticles->SetSize(mNumOfSampledPoints);

    // Compute mass and update other parameters...
    auto radius_array_device =
        thrust::device_pointer_cast(mInsertedParticles->radiusPtr());
    float max_rad = *(thrust::max_element(
        radius_array_device, radius_array_device + mNumOfSampledPoints));
    thrust::host_vector<float> all_inserted_mass;

    thrust::device_vector<float> inserted_radius(mInsertedParticles->radiusPtr(),
                                                mInsertedParticles->radiusPtr() +
                                                    mNumOfSampledPoints);
    thrust::host_vector<float> inserted_radius_host = inserted_radius;

    for (auto i = 0; i < mNumOfSampledPoints; i++) {
      all_inserted_mass.push_back(inserted_radius_host[i] / max_rad);
    }

    KIRI_CUCALL(cudaMemcpy(mInsertedParticles->massPtr(), &all_inserted_mass[0],
                          sizeof(float) * all_inserted_mass.size(),
                          cudaMemcpyHostToDevice));

    // Calculate and log statistics
    this->CalculateAndLogStatistics();

    // Adjust distribution probability for next iteration
    if (mInsertDistConstrain) {
        AdjustDistributionProbability();
    }

    // Re-initialize particles for next iteration
    this->ReinitializeParticles(num);

    mSystemStatus = XPROTOTYPE_INSERTED;
    mInsertedStep++;
}

void CudaProtoSphereSystem::CalculateAndLogStatistics() {
    // Calculate volume statistics
    float radius_sum = thrust::transform_reduce(
        thrust::device_ptr<float>(mInsertedParticles->radiusPtr()),
        thrust::device_ptr<float>(mInsertedParticles->radiusPtr() +
                                 mInsertedParticles->size()),
        power3_value<float>(), 0.f, thrust::plus<float>());

    auto porosity = 1.f - (4.f / 3.f * KIRI_PI * radius_sum) / mBoundaryVolume;

    // Calculate distribution statistics
    std::vector<float> predict_dist;
    std::string distribution_str = "";
    for (auto ri = 0; ri < mPreDefinedRadiusRange.size() - 1; ri++) {
        auto dist = mRadiusRangeCounter[ri] / (float)mNumOfSampledPoints;
        distribution_str += std::to_string(mPreDefinedRadiusRange[ri]) + "---" +
                          std::to_string(mPreDefinedRadiusRange[ri + 1]) + ":" +
                          std::to_string(dist) + "; ";
        predict_dist.emplace_back(dist);
    }

    // Calculate evaluation metrics
    auto bhatta_val = bhattacharyya_distance(mPreDefinedRadiusRangeProb, predict_dist);
    auto kl_val = kl_divergence(mPreDefinedRadiusRangeProb, predict_dist);
    auto jsd_val = jsd_distance(mPreDefinedRadiusRangeProb, predict_dist);
    auto average_rmspe = std::sqrt(mRMSPE / (float)mNumOfSampledPoints);

    KIRI_LOG_INFO("Distribution Data={0}; Porosity={1}; Sphere Numbers={2}; "
                  "RMSPE={3}; BHATTA={4}; KL Divergence={5}; JSD Distance={6};",
                  distribution_str, porosity, mNumOfSampledPoints, average_rmspe,
                  bhatta_val, kl_val, jsd_val);
}

// particle reinitialization
void CudaProtoSphereSystem::ReinitializeParticles(size_t num) {
    thrust::fill(thrust::device, mParticles->iterationNumPtr(),
                 mParticles->iterationNumPtr() + num, 0);

    thrust::fill(thrust::device, mParticles->convergencePtr(),
                 mParticles->convergencePtr() + num, 0);

    thrust::fill(thrust::device, mParticles->learningRatePtr(),
                 mParticles->learningRatePtr() + num, 1.f);

    thrust::copy(thrust::device, mParticles->GetInitialPosPtr(),
                 mParticles->GetInitialPosPtr() + num, mParticles->posPtr());
    thrust::copy(thrust::device, mParticles->GetInitialPosPtr(),
                 mParticles->GetInitialPosPtr() + num, mParticles->lastPosPtr());

    // Generate new target radii based on updated distribution
    Vec_Float targetRadius;
    std::random_device engine;
    std::mt19937 gen(engine());
    std::piecewise_constant_distribution<float> pcdis{
        std::begin(mCurrentRadiusRange), std::end(mCurrentRadiusRange),
        std::begin(mCurrentRadiusRangeProb)};

    for (auto i = 0; i < num; i++)
        targetRadius.emplace_back(pcdis(gen));

    KIRI_CUCALL(cudaMemcpy(mParticles->targetRadiusPtr(), &targetRadius[0],
                          sizeof(float) * targetRadius.size(),
                          cudaMemcpyHostToDevice));

    mSolver->SphereMoveRandom(mParticles, mHalfGridSize);
}

void CudaProtoSphereSystem::AdjustDistributionProbability() {
    // Calculate remaining space and volume-based weights
    float totalVolume = 0.f;
    std::vector<float> volumeWeights(mRadiusRangeCounter.size());
    
    for (size_t i = 0; i < mRadiusRangeCounter.size(); i++) {
        float remainNum = mRemainSamples[i] - mRadiusRangeCounter[i];
        remainNum = std::max(remainNum, 0.f);
        
        // Calculate average radius for this range
        float avgRadius = (mPreDefinedRadiusRange[i] + mPreDefinedRadiusRange[i + 1]) / 2.f;
        // Weight by volume (r^3) and remaining count
        float volumeWeight = std::pow(avgRadius, 3.f) * remainNum;
        volumeWeights[i] = volumeWeight;
        totalVolume += volumeWeight;
    }

    // Update probabilities based on volume-weighted distribution
    if (totalVolume > 0.f) {
        for (size_t i = 0; i < mRadiusRangeCounter.size(); i++) {
            // Base probability from volume weights
            float baseProb = volumeWeights[i] / totalVolume;
            
            // Apply progressive bias against filled ranges
            float fillRatio = static_cast<float>(mRadiusRangeCounter[i]) / mRemainSamples[i];
            float biasedProb = baseProb * std::exp(-fillRatio * 2.f); // Exponential decay
            
            mCurrentRadiusRangeProb[i] = biasedProb;
        }
        
        // Normalize probabilities
        float sumProb = 0.f;
        for (auto prob : mCurrentRadiusRangeProb) {
            sumProb += prob;
        }
        
        if (sumProb > 0.f) {
            for (size_t i = 0; i < mCurrentRadiusRangeProb.size(); i++) {
                mCurrentRadiusRangeProb[i] /= sumProb;
            }
        }
        
        // Apply minimum probability threshold to ensure diversity
        const float minProb = 0.05f;
        bool needsRenormalization = false;
        
        for (size_t i = 0; i < mCurrentRadiusRangeProb.size(); i++) {
            if (mRadiusRangeCounter[i] < mRemainSamples[i] && 
                mCurrentRadiusRangeProb[i] < minProb) {
                mCurrentRadiusRangeProb[i] = minProb;
                needsRenormalization = true;
            }
        }
        
        // Renormalize if minimum thresholds were applied
        if (needsRenormalization) {
            sumProb = 0.f;
            for (auto prob : mCurrentRadiusRangeProb) {
                sumProb += prob;
            }
            
            for (size_t i = 0; i < mCurrentRadiusRangeProb.size(); i++) {
                mCurrentRadiusRangeProb[i] /= sumProb;
            }
        }
    }
    
    // Debug output
    std::string distributionStr = "Updated distribution probabilities: ";
    for (size_t i = 0; i < mCurrentRadiusRangeProb.size(); i++) {
        distributionStr += std::to_string(mCurrentRadiusRangeProb[i]) + " ";
    }
    KIRI_LOG_INFO("{0}", distributionStr);
}

float CudaProtoSphereSystem::UpdateSystem() {
  cudaEvent_t start, stop;
  KIRI_CUCALL(cudaEventCreate(&start));
  KIRI_CUCALL(cudaEventCreate(&stop));
  KIRI_CUCALL(cudaEventRecord(start, 0));

  switch (mSystemStatus) {
  case XPROTOTYPE_SEARCHING:
    this->RunProtoSphereSampler();
    break;
  case XPROTOTYPE_INSERTED:
    if (mEnableDEMRelax) {
      mSystemStatus = DEM_RELAXATION;
      mRelaxStep = 0;
      mDEMMSSolver =
          std::make_shared<CudaDEMMultiSizedSampler>(mNumOfSampledPoints);
    } else {
      if (mInsertedStep >= mMaxProtoSphereIterationNum)
        mSystemStatus = XPROTOSPHERE_FINISH;
      else
        mSystemStatus = XPROTOTYPE_SEARCHING;
    }

    break;
  case DEM_RELAXATION:
    this->RunDEMMSSampler();
    break;

  case DEM_RELAXATION_FINISH:
    if (mInsertedStep >= mMaxProtoSphereIterationNum)
      mSystemStatus = XPROTOSPHERE_FINISH;
    else
      mSystemStatus = XPROTOTYPE_SEARCHING;
    break;
  default:
    break;
  }

  cudaDeviceSynchronize();
  KIRI_CUKERNAL();

  float milliseconds;
  KIRI_CUCALL(cudaEventRecord(stop, 0));
  KIRI_CUCALL(cudaEventSynchronize(stop));
  KIRI_CUCALL(cudaEventElapsedTime(&milliseconds, start, stop));
  KIRI_CUCALL(cudaEventDestroy(start));
  KIRI_CUCALL(cudaEventDestroy(stop));
  return milliseconds;
}

} // namespace KIRI
