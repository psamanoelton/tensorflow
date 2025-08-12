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

#include "xla/backends/gpu/autotuner/native_emitter.h"

#include <memory>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
NativeEmitterBackend::GetSupportedConfigs(const HloInstruction& instr) {
  NativeEmitterBackendConfig config;
  auto any = std::make_unique<google::protobuf::Any>();
  any->PackFrom(config);
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(std::move(any));
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>>
NativeEmitterBackend::GetDefaultConfig(const HloInstruction& instr) {
  NativeEmitterBackendConfig config;
  auto any = std::make_unique<google::protobuf::Any>();
  any->PackFrom(config);
  return any;
}

absl::Status NativeEmitterBackend::ApplyConfig(HloInstruction& instr,
                                               const BackendConfig& config) {
  auto fusion_instr = Cast<HloFusionInstruction>(&instr);
  fusion_instr->set_fusion_kind(HloInstruction::FusionKind::kInput);
  fusion_instr->clear_backend_config();
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
