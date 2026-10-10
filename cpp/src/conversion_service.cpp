#include "sd15/conversion_service.hpp"

namespace sd15 {
namespace {
std::atomic<uint64_t> next_job_id{1};
}

ConversionService::~ConversionService() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (token_) {
            token_->request_cancel();
        }
    }
    if (worker_.joinable()) {
        worker_.join();
    }
}

StartResult ConversionService::start_conversion(const ConvertOptions& options, ConversionObserver observer) {
    StartResult result;
    std::unique_lock<std::mutex> lifecycle(lifecycle_, std::try_to_lock);
    if (!lifecycle.owns_lock()) {
        result.error.code = ErrorCode::Busy;
        return result;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_) {
            result.error.code = ErrorCode::Busy;
            return result;
        }
    }
    try {
        if (worker_.joinable()) {
            worker_.join();
        }
        auto token = std::make_shared<CancellationToken>();
        const auto id = next_job_id.fetch_add(1, std::memory_order_relaxed);
        // 첫 알림이 시작 상태보다 먼저 전달되지 않도록 잠근다.
        std::lock_guard<std::mutex> lock(mutex_);
        worker_ = std::thread([this, options, token, observer = std::move(observer)] {
            ConvertResult outcome;
            try {
                outcome = convert(
                    options,
                    [&](const ConvertProgress& progress) {
                        ConversionSnapshot current;
                        {
                            std::lock_guard<std::mutex> guard(mutex_);
                            snapshot_.progress = progress;
                            current = snapshot_;
                        }
                        if (observer) {
                            observer(current);
                        }
                    },
                    *token);
            }
            catch (const std::bad_alloc&) {
                outcome.error.code = ErrorCode::OutOfMemory;
            }
            catch (...) {
                outcome.error.code = ErrorCode::InternalError;
            }
            {
                std::lock_guard<std::mutex> guard(mutex_);
                switch (outcome.status) {
                case ConvertStatus::Succeeded:
                    snapshot_.state = JobState::Succeeded;
                    break;
                case ConvertStatus::Cancelled:
                    snapshot_.state = JobState::Cancelled;
                    break;
                case ConvertStatus::Failed:
                    snapshot_.state = JobState::Failed;
                    break;
                }
                snapshot_.progress.elapsed_ms = outcome.elapsed_ms;
                snapshot_.progress.remaining_ms.reset();
                if (outcome.status == ConvertStatus::Succeeded) {
                    snapshot_.progress.remaining_ms = 0;
                }
                snapshot_.result = std::move(outcome);
            }
            // 종료 알림 실패는 확정된 결과를 변경하지 않는다.
            try {
                if (observer) {
                    const auto current = get_conversion_status(snapshot_.job_id);
                    if (current) {
                        observer(*current);
                    }
                }
            }
            catch (...) {
            }
            std::lock_guard<std::mutex> guard(mutex_);
            active_ = false;
        });
        token_ = std::move(token);
        snapshot_ = {};
        snapshot_.job_id = id;
        active_ = true;
        result.job_id = id;
    }
    catch (const std::bad_alloc&) {
        result.error.code = ErrorCode::OutOfMemory;
    }
    catch (...) {
        result.error.code = ErrorCode::InternalError;
    }
    return result;
}

std::optional<ConversionSnapshot> ConversionService::get_conversion_status(uint64_t job_id, ConvertError* error) const {
    if (error != nullptr) {
        *error = {};
    }
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (job_id != 0 && snapshot_.job_id == job_id) {
            return snapshot_;
        }
    }
    catch (const std::bad_alloc&) {
        if (error != nullptr) {
            error->code = ErrorCode::OutOfMemory;
        }
    }
    catch (...) {
        if (error != nullptr) {
            error->code = ErrorCode::InternalError;
        }
    }
    return std::nullopt;
}

bool ConversionService::cancel_conversion(uint64_t job_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (job_id == 0 || snapshot_.job_id != job_id) {
        return false;
    }
    if (!snapshot_.result) {
        snapshot_.state = JobState::Cancelling;
        token_->request_cancel();
    }
    return true;
}

} // namespace sd15
