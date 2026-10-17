#include "scheduler.hpp"
#include "ddim_scheduler.hpp"
#include "dpm_scheduler.hpp"

#include <stdexcept>

namespace sd15::detail {

std::unique_ptr<Scheduler> make_scheduler(SamplingPreset preset, int steps) {
    switch (preset) {
    case SamplingPreset::DdimLeading:
        return std::make_unique<DdimScheduler>(steps);
    case SamplingPreset::Dpmpp2mKarras:
        return std::make_unique<DpmScheduler>(steps);
    }
    throw std::invalid_argument("Invalid sampling preset");
}

} // namespace sd15::detail
