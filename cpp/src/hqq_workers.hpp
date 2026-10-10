#pragma once

#include "conversion_progress.hpp"

#include <condition_variable>
#include <mutex>
#include <thread>

namespace sd15::detail {

struct HqqBatch {
    const float* values;
    uint8_t* payload;
    uint8_t* scales;
    std::size_t groups;
    std::size_t area;
    int iterations;
};

// 변환 한 번의 작업 스레드를 소유한다. run은 호출 스레드에서만 사용한다.
class HqqWorkers {
public:
    explicit HqqWorkers(unsigned thread_count);
    ~HqqWorkers();
    HqqWorkers(const HqqWorkers&) = delete;
    HqqWorkers& operator=(const HqqWorkers&) = delete;

    // 반환 또는 예외 전달 전에 모든 작업이 버퍼 사용을 끝낸다.
    HqqError run(const HqqBatch& batch, Progress& progress);

private:
    void stop();
    void worker(unsigned index);
    HqqError compute(unsigned index, Progress* progress);

    std::array<std::thread, kMaxThreadCount - 1> threads_;
    std::array<HqqError, kMaxThreadCount> errors_{};
    std::mutex mutex_;
    std::condition_variable ready_, done_;
    unsigned thread_count_;
    unsigned active_ = 1;
    unsigned pending_ = 0;
    uint64_t generation_ = 0;
    bool stopping_ = false;
    std::atomic<bool> abort_{false};
    HqqBatch batch_{};
    const CancellationToken* cancellation_ = nullptr;
};

}  // namespace sd15::detail
