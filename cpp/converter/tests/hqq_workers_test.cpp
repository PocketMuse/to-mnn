#include "hqq_workers.hpp"
#include "tensor_transform.hpp"

#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <stdexcept>

using namespace sd15;
using namespace sd15::detail;

TEST_CASE("HQQ workers preserve output and recover after errors") {
    for (unsigned threads : {1, 2, 3, 4, 8}) {
        CAPTURE(threads);
        HqqWorkers workers(threads);
        for (std::size_t groups : {1, 3, 17, 65}) {
            CAPTURE(groups);
            constexpr std::size_t area = 128;
            std::vector<float> values(groups * area);
            for (std::size_t i = 0; i < values.size(); ++i) {
                values[i] = static_cast<float>(static_cast<int>(i % 131) - 65) / 17;
            }
            std::vector<uint8_t> expected(values.size()), alpha(groups * kHqqAlphaBytes);
            for (std::size_t i = 0; i < groups; ++i) {
                UNSCOPED_INFO("serial reference");
                REQUIRE(encode_hqq_group(values.data() + i * area, area, 20, expected.data() + i * area,
                                         alpha.data() + i * kHqqAlphaBytes) == HqqError::None);
            }
            std::vector<uint8_t> payload(expected.size() + 2, 0xa5), scales(alpha.size() + 2, 0xa5);
            CancellationToken token;
            const auto caller = std::this_thread::get_id();
            unsigned callbacks = 0;
            ProgressCallback callback = [&](const auto&) {
                UNSCOPED_INFO("callback stays on caller");
                REQUIRE(std::this_thread::get_id() == caller);
                ++callbacks;
            };
            Progress progress(callback, token);
            progress.notified -= std::chrono::seconds(1);
            HqqBatch batch{values.data(), payload.data() + 1, scales.data() + 1, groups, area, 20};
            UNSCOPED_INFO("parallel succeeds");
            REQUIRE(workers.run(batch, progress) == HqqError::None);
            UNSCOPED_INFO("callback exercised during computation");
            REQUIRE(callbacks != 0);
            UNSCOPED_INFO("exact payload");
            REQUIRE(std::equal(expected.begin(), expected.end(), payload.begin() + 1));
            UNSCOPED_INFO("exact alpha");
            REQUIRE(std::equal(alpha.begin(), alpha.end(), scales.begin() + 1));
            UNSCOPED_INFO("output bounds");
            REQUIRE(payload.front() == 0xa5);
            REQUIRE(payload.back() == 0xa5);
            REQUIRE(scales.front() == 0xa5);
            REQUIRE(scales.back() == 0xa5);

            // 마지막 그룹은 호출 스레드가 아닌 작업 스레드의 오류 경로를 검사한다.
            values[values.size() - 2] = -std::numeric_limits<float>::max();
            values.back() = std::numeric_limits<float>::max();
            UNSCOPED_INFO("worker error returned");
            REQUIRE(workers.run(batch, progress) == HqqError::RangeOverflow);
            values[values.size() - 2] = 0;
            values.back() = 1;
            UNSCOPED_INFO("reuse after failure");
            REQUIRE(workers.run(batch, progress) == HqqError::None);

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
            UNSCOPED_INFO("callback exception drained");
            REQUIRE(caught);
            UNSCOPED_INFO("reuse after callback failure");
            REQUIRE(workers.run(batch, progress) == HqqError::None);

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
            UNSCOPED_INFO("cancel while batch dispatched");
            REQUIRE(caught);
            UNSCOPED_INFO("reuse after cancellation");
            REQUIRE(workers.run(batch, progress) == HqqError::None);

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
            UNSCOPED_INFO("external cancellation during batch");
            REQUIRE(caught);
            UNSCOPED_INFO("reuse after external cancellation");
            REQUIRE(workers.run(batch, progress) == HqqError::None);
        }
    }
}
