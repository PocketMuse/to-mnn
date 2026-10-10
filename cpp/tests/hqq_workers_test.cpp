#include "hqq_workers.hpp"
#include "tensor_transform.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>

using namespace sd15;
using namespace sd15::detail;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main() {
    try {
        for (unsigned threads : {1, 2, 3, 4, 8}) {
            HqqWorkers workers(threads);
            for (std::size_t groups : {1, 3, 17, 65}) {
                constexpr std::size_t area = 128;
                std::vector<float> values(groups * area);
                for (std::size_t i = 0; i < values.size(); ++i) {
                    values[i] = static_cast<float>(static_cast<int>(i % 131) - 65) / 17;
                }
                std::vector<uint8_t> expected(values.size()), alpha(groups * kHqqAlphaBytes);
                for (std::size_t i = 0; i < groups; ++i) {
                    require(encode_hqq_group(values.data() + i * area, area, 20, expected.data() + i * area,
                                             alpha.data() + i * kHqqAlphaBytes) == HqqError::None,
                            "serial reference");
                }
                std::vector<uint8_t> payload(expected.size() + 2, 0xa5), scales(alpha.size() + 2, 0xa5);
                CancellationToken token;
                const auto caller = std::this_thread::get_id();
                unsigned callbacks = 0;
                ProgressCallback callback = [&](const auto&) {
                    require(std::this_thread::get_id() == caller, "callback stays on caller");
                    ++callbacks;
                };
                Progress progress(callback, token);
                progress.notified -= std::chrono::seconds(1);
                HqqBatch batch{values.data(), payload.data() + 1, scales.data() + 1, groups, area, 20};
                require(workers.run(batch, progress) == HqqError::None, "parallel succeeds");
                require(callbacks != 0, "callback exercised during computation");
                require(std::equal(expected.begin(), expected.end(), payload.begin() + 1), "exact payload");
                require(std::equal(alpha.begin(), alpha.end(), scales.begin() + 1), "exact alpha");
                require(payload.front() == 0xa5 && payload.back() == 0xa5 && scales.front() == 0xa5 &&
                            scales.back() == 0xa5,
                        "output bounds");

                // 마지막 그룹은 호출 스레드가 아닌 작업 스레드의 오류 경로를 검사한다.
                values[values.size() - 2] = -std::numeric_limits<float>::max();
                values.back() = std::numeric_limits<float>::max();
                require(workers.run(batch, progress) == HqqError::RangeOverflow, "worker error returned");
                values[values.size() - 2] = 0;
                values.back() = 1;
                require(workers.run(batch, progress) == HqqError::None, "reuse after failure");

                ProgressCallback throws = [](const auto&) { throw 42; };
                Progress failed(throws, token);
                failed.notified -= std::chrono::seconds(1);
                bool caught = false;
                try {
                    workers.run(batch, failed);
                }
                catch (const CallbackFailed&) {
                    caught = true;
                }
                require(caught, "callback exception drained");
                require(workers.run(batch, progress) == HqqError::None, "reuse after callback failure");

                CancellationToken cancelled;
                ProgressCallback cancel = [&](const auto&) { cancelled.request_cancel(); };
                Progress stopped(cancel, cancelled);
                stopped.notified -= std::chrono::seconds(1);
                caught = false;
                try {
                    workers.run(batch, stopped);
                }
                catch (const Cancelled&) {
                    caught = true;
                }
                require(caught, "cancel while batch dispatched");
                require(workers.run(batch, progress) == HqqError::None, "reuse after cancellation");

                CancellationToken external;
                std::mutex gate;
                std::condition_variable ready;
                bool entered = false;
                bool requested = false;
                ProgressCallback signal = [&](const auto&) {
                    std::unique_lock<std::mutex> lock(gate);
                    entered = true;
                    ready.notify_one();
                    ready.wait(lock, [&] { return requested; });
                };
                Progress asynchronous(signal, external);
                asynchronous.notified -= std::chrono::seconds(1);
                std::thread canceller([&] {
                    std::unique_lock<std::mutex> lock(gate);
                    ready.wait(lock, [&] { return entered; });
                    external.request_cancel();
                    requested = true;
                    ready.notify_one();
                });
                caught = false;
                try {
                    workers.run(batch, asynchronous);
                }
                catch (const Cancelled&) {
                    caught = true;
                }
                canceller.join();
                require(caught, "external cancellation during batch");
                require(workers.run(batch, progress) == HqqError::None, "reuse after external cancellation");
            }
        }
        std::cout << "HQQ worker tests passed\n";
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
