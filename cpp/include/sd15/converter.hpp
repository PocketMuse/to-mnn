#pragma once

#include <cstddef>
#include <string>

namespace sd15 {

constexpr std::size_t kDefaultChunkBytes = 1024 * 1024;

/// 입력 체크포인트·템플릿 경로와 새 출력 디렉터리, 작업 청크 크기.
struct ConvertOptions {
    std::string checkpoint;
    std::string template_dir;
    std::string output_dir;
    /// 입력 버퍼 크기. 4바이트~4MiB의 4의 배수이며 출력 버퍼는 두 배다.
    std::size_t chunk_bytes = kDefaultChunkBytes;
};

/// F32/F16 safetensors와 템플릿으로 MNN 파일을 청크 단위 복원한다.
/// 성공 시 true, 실패 시 false와 error를 반환한다. 기존 출력은 거부한다.
/// 완성 전 결과는 .partial에 쓰며 일반 오류 시 이번 작업 파일만 정리한다.
bool convert(const ConvertOptions& options, std::string& error);

}  // namespace sd15
