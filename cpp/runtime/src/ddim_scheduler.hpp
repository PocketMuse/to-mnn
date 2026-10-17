#pragma once

#include <cstddef>
#include <vector>

namespace sd15::detail {

// 요청마다 생성한다. SD1.5 DDIM의 시간표와 갱신 규칙을 소유한다.
class DdimScheduler {
public:
    explicit DdimScheduler(int steps);
    const std::vector<int>& timesteps() const noexcept { return timesteps_; }
    float init_noise_sigma() const noexcept { return 1.0f; }
    const std::vector<float>& scale_model_input(const std::vector<float>& sample, std::size_t) const noexcept {
        return sample;
    }
    // index는 timesteps의 위치이며 model_output은 CFG 적용 후 예측이다.
    void step(std::vector<float>& latent, const std::vector<float>& model_output, std::size_t index,
              std::vector<float>* clean_prediction = nullptr) const;

private:
    const std::vector<int> timesteps_;
    const std::vector<float> alphas_;
};

} // namespace sd15::detail
