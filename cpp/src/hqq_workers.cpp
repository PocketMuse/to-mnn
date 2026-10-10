#include "hqq_workers.hpp"
#include "tensor_transform.hpp"

#include <exception>
#include <stdexcept>

namespace sd15::detail {

HqqWorkers::HqqWorkers(unsigned thread_count) : thread_count_(thread_count) {
    if (thread_count < 1 || thread_count > kMaxThreadCount) {
        throw std::invalid_argument("Thread count must be in [1, 8]");
    }
    try {
        for (unsigned i = 1; i < thread_count_; ++i) {
            threads_[i - 1] = std::thread(&HqqWorkers::worker, this, i);
        }
    }
    catch (...) {
        stop();
        throw;
    }
}

HqqWorkers::~HqqWorkers() {
    stop();
}

void HqqWorkers::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    ready_.notify_all();
    for (auto& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

HqqError HqqWorkers::compute(unsigned index, Progress* progress) {
    if (index >= active_) {
        return HqqError::None;
    }
    const std::size_t begin = batch_.groups * index / active_;
    const std::size_t end = batch_.groups * (index + 1) / active_;
    for (std::size_t i = begin; i < end; ++i) {
        if (abort_.load(std::memory_order_relaxed) || cancellation_->is_requested()) {
            break;
        }
        if (progress != nullptr) {
            progress->emit();
            progress->check();
        }
        const auto error = encode_hqq_group(batch_.values + i * batch_.area, batch_.area, batch_.iterations,
                                            batch_.payload + i * batch_.area, batch_.scales + i * kHqqAlphaBytes);
        if (error != HqqError::None) {
            return error;
        }
    }
    return HqqError::None;
}

void HqqWorkers::worker(unsigned index) {
    uint64_t observed = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        ready_.wait(lock, [&] { return stopping_ || generation_ != observed; });
        if (stopping_) {
            return;
        }
        observed = generation_;
        lock.unlock();
        const auto error = compute(index, nullptr);
        lock.lock();
        errors_[index] = error;
        --pending_;
        done_.notify_one();
    }
}

HqqError HqqWorkers::run(const HqqBatch& batch, Progress& progress) {
    progress.check();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        batch_ = batch;
        cancellation_ = &progress.cancellation;
        active_ = static_cast<unsigned>(std::min<std::size_t>(thread_count_, batch.groups));
        errors_.fill(HqqError::None);
        abort_.store(false, std::memory_order_relaxed);
        pending_ = thread_count_ - 1;
        ++generation_;
    }
    ready_.notify_all();
    std::exception_ptr failure;
    try {
        errors_[0] = compute(0, &progress);
        std::unique_lock<std::mutex> lock(mutex_);
        while (pending_ != 0) {
            done_.wait_for(lock, std::chrono::milliseconds(kProgressIntervalMs));
            lock.unlock();
            progress.emit();
            progress.check();
            lock.lock();
        }
    }
    catch (...) {
        abort_.store(true, std::memory_order_relaxed);
        failure = std::current_exception();
    }
    // 콜백 예외도 작업 완료까지 보관한다. 스택 해제는 대기 이후에만 허용한다.
    {
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [&] { return pending_ == 0; });
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
    progress.check();
    for (unsigned i = 0; i < active_; ++i) {
        if (errors_[i] != HqqError::None) {
            return errors_[i];
        }
    }
    return HqqError::None;
}

} // namespace sd15::detail
