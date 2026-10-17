#pragma once

#include "mnn_model_session.hpp"

namespace sd15::detail {

class ClipEncoder {
public:
    ClipEncoder(const std::filesystem::path& path, const ModelOptions& options,
                const GenerationCancellationToken* cancellation = nullptr);
    std::vector<float> encode(const std::vector<int32_t>& tokens);

private:
    MnnModelSession session_;
    MnnModelSession::Input<int32_t> tokens_;
    MnnModelSession::Output context_;
};

class UNetDenoiser {
public:
    UNetDenoiser(const std::filesystem::path& path, const ModelOptions& options,
                 const GenerationCancellationToken* cancellation = nullptr);
    std::vector<float> predict(const std::vector<float>& latent, int timestep, const std::vector<float>& context);

private:
    MnnModelSession session_;
    MnnModelSession::Input<float> latent_;
    MnnModelSession::Input<int32_t> timestep_;
    MnnModelSession::Input<float> context_;
    MnnModelSession::Output noise_;
};

class PreviewDecoder {
public:
    PreviewDecoder(const std::filesystem::path& path, const ModelOptions& options,
                   const GenerationCancellationToken* cancellation = nullptr);
    Image decode(const std::vector<float>& latent);

private:
    MnnModelSession session_;
    MnnModelSession::Input<float> latent_;
    MnnModelSession::Output image_;
};

class VaeDecoder {
public:
    VaeDecoder(const std::filesystem::path& path, const ModelOptions& options,
               const GenerationCancellationToken* cancellation = nullptr);
    // latent를 받아 내부에서 VAE scale을 적용한다.
    Image decode(std::vector<float> latent);

private:
    MnnModelSession session_;
    MnnModelSession::Input<float> latent_;
    MnnModelSession::Output image_;
};

} // namespace sd15::detail
