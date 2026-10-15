#pragma once

#include "mnn_model_session.hpp"

namespace sd15::detail {

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
