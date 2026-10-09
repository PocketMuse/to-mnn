#pragma once

#include "sd15/converter.hpp"

#include <memory>
#include <mutex>
#include <thread>

namespace sd15 {

enum class JobState { Running, Cancelling, Succeeded, Cancelled, Failed };

struct ConversionSnapshot {
    uint64_t job_id = 0;
    JobState state = JobState::Running;
    ConvertProgress progress;
    std::optional<ConvertResult> result;
};

struct StartResult {
    uint64_t job_id = 0;
    ConvertError error;
};

using ConversionObserver = std::function<void(const ConversionSnapshot&)>;

/// Expo 모듈 인스턴스가 소유한다. 가장 최근 작업 하나의 상태를 보관한다.
class ConversionService {
public:
    ConversionService() = default;
    ConversionService(const ConversionService&) = delete;
    ConversionService& operator=(const ConversionService&) = delete;
    ~ConversionService();

    StartResult start_conversion(const ConvertOptions& options, ConversionObserver observer = {});
    std::optional<ConversionSnapshot> get_conversion_status(uint64_t job_id, ConvertError* error = nullptr) const;
    bool cancel_conversion(uint64_t job_id);

private:
    mutable std::mutex mutex_;
    std::mutex lifecycle_;
    std::thread worker_;
    std::shared_ptr<CancellationToken> token_;
    ConversionSnapshot snapshot_;
    bool active_ = false;
};

}  // namespace sd15
