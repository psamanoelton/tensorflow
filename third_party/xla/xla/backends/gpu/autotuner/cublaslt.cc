/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/gpu/autotuner/cublaslt.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;
using se::gpu::BlasLt;

using CublasLtBackendConfig = AutotuneResult::GemmKey;

namespace {

absl::StatusOr<BlasLt::Epilogue> AsBlasLtEpilogue(
    GemmBackendConfig_Epilogue epilogue) {
  switch (epilogue) {
    case GemmBackendConfig::DEFAULT:
      return BlasLt::Epilogue::kDefault;
    case GemmBackendConfig::RELU:
      return BlasLt::Epilogue::kReLU;
    case GemmBackendConfig::GELU:
      return BlasLt::Epilogue::kGELU;
    case GemmBackendConfig::GELU_AUX:
      return BlasLt::Epilogue::kGELUWithAux;
    case GemmBackendConfig::BIAS:
      return BlasLt::Epilogue::kBias;
    case GemmBackendConfig::BIAS_RELU:
      return BlasLt::Epilogue::kBiasThenReLU;
    case GemmBackendConfig::BIAS_GELU:
      return BlasLt::Epilogue::kBiasThenGELU;
    case GemmBackendConfig::BIAS_GELU_AUX:
      return BlasLt::Epilogue::kBiasThenGELUWithAux;
    default:
      return Internal("Unsupported Epilogue.");
  }
}

bool IsSupported(const HloInstruction& instr) {
  return IsCublasLtMatmul(instr) || IsCublasLtMatmulF8(instr);
}

}  // namespace

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
CublasLtBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return std::vector<std::unique_ptr<BackendConfig>>();
  }

  GpuBackendConfig gpu_config =
      instr.backend_config<GpuBackendConfig>().value();
  const GemmBackendConfig& backend_config = gpu_config.gemm_backend_config();

  TF_ASSIGN_OR_RETURN(
      GemmConfig gemm_config,
      GemmConfig::For(
          &instr, target_config().device_description.gpu_compute_capability()));

  TF_ASSIGN_OR_RETURN(BlasLt::Epilogue epilogue,
                      AsBlasLtEpilogue(backend_config.epilogue()));

  auto allocator =
      std::make_unique<se::StreamExecutorMemoryAllocator>(stream_executor());
  TF_ASSIGN_OR_RETURN(
      se::Stream * stream,
      allocator->GetStream(stream_executor()->device_ordinal()));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<BlasLt::MatmulPlan> plan,
      se::gpu::BlasLt::GetMatmulPlan(stream, gemm_config, epilogue));

  auto create_matrix_desc = [](const se::gpu::MatrixLayout& layout)
      -> absl::StatusOr<se::gpu::MatrixDescriptor> {
    TF_ASSIGN_OR_RETURN(se::blas::DataType type,
                        se::gpu::AsBlasDataType(layout.dtype));
    return se::gpu::MatrixDescriptor{
        /*data=*/se::DeviceMemoryBase(), layout.leading_dim_stride,
        layout.batch_stride, type,
        // BLAS is column-major by default.
        (layout.order == se::gpu::MatrixLayout::Order::kColumnMajor
             ? se::blas::Transpose::kNoTranspose
             : se::blas::Transpose::kTranspose)};
  };

  TF_ASSIGN_OR_RETURN(se::gpu::MatrixDescriptor lhs_desc,
                      create_matrix_desc(gemm_config.lhs_layout));
  TF_ASSIGN_OR_RETURN(se::gpu::MatrixDescriptor rhs_desc,
                      create_matrix_desc(gemm_config.rhs_layout));
  TF_ASSIGN_OR_RETURN(se::gpu::MatrixDescriptor output_desc_base,
                      create_matrix_desc(gemm_config.output_layout));

  se::gpu::OutputMatrixDescriptor out_desc(std::move(output_desc_base));
  out_desc.batch_size = gemm_config.output_layout.batch_size;
  out_desc.m = gemm_config.output_layout.num_rows;
  out_desc.n = gemm_config.output_layout.num_cols;
  out_desc.k = gemm_config.lhs_layout.num_cols;
  TF_ASSIGN_OR_RETURN(
      out_desc.compute_type,
      se::gpu::GetBlasComputationType(
          gemm_config.precision_algorithm, gemm_config.lhs_layout.dtype,
          gemm_config.output_layout.dtype, gemm_config.compute_precision));

  se::blas::BlasSupport* blas = stream_executor()->AsBlas();
  if (blas == nullptr) {
    return absl::InternalError("Failed to getBlas support.");
  }
  std::vector<se::blas::AlgorithmType> algorithms;
  blas->GetBlasGemmAlgorithms(stream, lhs_desc, rhs_desc, &out_desc,
                              &gemm_config.alpha, &gemm_config.beta,
                              &algorithms);
  int num_algorithms = algorithms.size();
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.reserve(num_algorithms);
  for (int i = 0; i < num_algorithms; ++i) {
    CublasLtBackendConfig gemm_key;
    gemm_key.set_algorithm(i);
    auto any = std::make_unique<google::protobuf::Any>();
    any->PackFrom(gemm_key);
    configs.push_back(std::move(any));
  }

  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>>
CublasLtBackend::GetDefaultConfig(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "Not a CublasLt custom call instruction.");
  }

  AutotuneResult::GemmKey gemm_key;
  gemm_key.set_algorithm(0);
  auto any = std::make_unique<google::protobuf::Any>();
  any->PackFrom(gemm_key);
  return any;
}

absl::Status CublasLtBackend::ApplyConfig(HloInstruction& instr,
                                          const BackendConfig& config) {
  CublasLtBackendConfig gemm_key;
  if (!config.UnpackTo(&gemm_key)) {
    return absl::InvalidArgumentError(
        "Failed to unpack CublasLtBackendConfig from Any.");
  }
  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      instr.backend_config<GpuBackendConfig>());
  GemmBackendConfig& backend_config = *gpu_config.mutable_gemm_backend_config();
  backend_config.set_selected_algorithm(gemm_key.algorithm());
  TF_RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
