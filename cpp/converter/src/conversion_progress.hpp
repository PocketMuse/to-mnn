#pragma once

#include "sd15/converter.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>

namespace sd15::detail {

constexpr uint64_t kProgressIntervalMs = 100;
using Clock = std::chrono::steady_clock;

struct Cancelled {};
struct CallbackFailed {};

enum class WorkKind : std::size_t { Copy, Float, Hqq, Count };

/// 저장 바이트 진행률과 작업 종류별 실측 속도를 관리한다.
struct Progress {
    const ProgressCallback& callback;
    const CancellationToken& cancellation;
    Clock::time_point started = Clock::now();
    Clock::time_point notified = started;
    ConvertProgress value;
    std::chrono::duration<double> hqq_compute_time{};
    std::array<uint64_t, static_cast<std::size_t>(WorkKind::Count)> total{};
    std::array<uint64_t, static_cast<std::size_t>(WorkKind::Count)> done{};
    std::array<double, static_cast<std::size_t>(WorkKind::Count)> seconds{};

    Progress(const ProgressCallback& observer, const CancellationToken& token)
        : callback(observer), cancellation(token) {}

    uint64_t elapsed() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started).count();
    }

    void check() const {
        if (cancellation.is_requested()) {
            throw Cancelled{};
        }
    }

    void emit(bool force = false) {
        const auto now = Clock::now();
        if (!force && std::chrono::duration_cast<std::chrono::milliseconds>(now - notified).count() <
                          static_cast<int64_t>(kProgressIntervalMs)) {
            return;
        }
        value.elapsed_ms = elapsed();
        value.fraction =
            value.total_bytes == 0
                ? 0
                : std::min(0.99, static_cast<double>(value.completed_bytes) / static_cast<double>(value.total_bytes));
        value.remaining_ms.reset();
        double remaining = 0;
        bool known = value.stage == ConvertStage::Converting;
        for (std::size_t i = 0; i < total.size(); ++i) {
            if (total[i] == done[i]) {
                continue;
            }
            if (done[i] == 0 || seconds[i] <= 0) {
                known = false;
                continue;
            }
            remaining += seconds[i] * static_cast<double>(total[i] - done[i]) / static_cast<double>(done[i]);
        }
        if (known && std::isfinite(remaining) && remaining * 1000 < static_cast<double>(UINT64_MAX)) {
            value.remaining_ms = static_cast<uint64_t>(std::ceil(remaining * 1000));
        }
        if (value.stage == ConvertStage::Completed) {
            value.fraction = 1;
            value.remaining_ms = 0;
        }
        notified = now;
        if (callback) {
            try {
                callback(value);
            }
            catch (...) {
                throw CallbackFailed{};
            }
        }
    }

    void advance(WorkKind kind, uint64_t bytes, Clock::time_point begin) {
        const auto i = static_cast<std::size_t>(kind);
        done[i] += bytes;
        seconds[i] += std::chrono::duration<double>(Clock::now() - begin).count();
        value.completed_bytes += bytes;
        emit();
    }
};

} // namespace sd15::detail
