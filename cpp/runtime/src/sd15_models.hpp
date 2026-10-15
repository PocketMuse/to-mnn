#pragma once

#include "mnn_model_session.hpp"

namespace sd15::detail {

struct RuntimeModelOptions {
    ModelOptions clip;
    ModelOptions unet;
    ModelOptions vae;
};

// 공개 설정을 모델별 설정으로 한 번만 변환한다.
RuntimeModelOptions resolve_model_options(const RuntimeOptions& options);

class ClipEncoder {
public:
    ClipEncoder(const std::filesystem::path& path, const ModelOptions& options);
    std::vector<float> encode(const std::vector<int32_t>& tokens);

private:
    MnnModelSession session_;
};

class UNetDenoiser {
public:
    UNetDenoiser(const std::filesystem::path& path, const ModelOptions& options);
    std::vector<float> predict(const std::vector<float>& latent, int timestep, const std::vector<float>& context);

private:
    MnnModelSession session_;
};

class VaeDecoder {
public:
    VaeDecoder(const std::filesystem::path& path, const ModelOptions& options);
    // latent를 받아 내부에서 VAE scale을 적용한다.
    Image decode(std::vector<float> latent);

private:
    MnnModelSession session_;
};

} // namespace sd15::detail
