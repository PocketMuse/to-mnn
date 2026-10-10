# SD1.5 Converter API

## 1. 입력 준비

* C++17 정적 라이브러리: `sd15_converter`
* 헤더: [converter.hpp](../cpp/include/sd15/converter.hpp), [conversion_service.hpp](../cpp/include/sd15/conversion_service.hpp)
* 입력: F32/F16 safetensors, `graph.bin`, `manifest.json`
* 출력: `{text_encoder,unet,vae_decoder}.mnn`과 필요한 외부 가중치
* MNNConvert나 MNN Runtime 없이 변환. FP16 템플릿 v1, HQQ 템플릿 v2 지원
* 양자화 정책: [SD1.5 ADR](sd15.md)

```cpp
sd15::ConvertOptions options;
options.checkpoint = "/app/models/model.safetensors";
options.template_dir = "/app/templates/sd15-hqq-b128";
options.output_dir = "/app/models/converted";
options.chunk_bytes = sd15::kDefaultChunkBytes;
options.hqq_iterations = 20;
```

* 앱이 관리하는 로컬 경로 사용. 파일 선택 URI는 로컬 파일로 준비
* 출력 부모 폴더는 미리 생성하고 입력·템플릿은 작업 중 변경하지 않음
* 입력 청크 기본 1MiB, 허용 범위 4바이트~4MiB의 4의 배수
* `hqq_iterations`: 0~20, 기본 20. 0은 초기 min/max로 W8 양자화하고 HQQ 보정을 생략
* 기존 manifest의 20회는 참조 설정으로 유지. 실행 횟수만 덮어쓰며 템플릿 재생성은 불필요
* FP16 전용 템플릿에는 적용되지 않음. 기본 20회만 기존 참조 모델과 바이트 일치를 기대
* CLI: `--hqq-iterations 10`. 0회는 MNNConvert의 `--hqq` 미적용과 바이트 일치를 보장하지 않음

> 기존 출력 폴더는 덮어쓰지 않음. 청크 크기는 전체 메모리 상한이 아님

## 2. 검사 및 변환

### `inspect` — 변환 전 검사

```cpp
const auto info = sd15::inspect(options);
```

* 템플릿 기준 key·shape·dtype, 데이터 범위, 저장 구간·양자화 설정 확인
* 파일 생성 없이 헤더와 manifest만 검사. UI 스레드 밖에서 호출
* NaN/Inf와 HQQ 범위 초과는 변환 중 검사
* 가용 공간이 출력 크기보다 작으면 `INSUFFICIENT_SPACE`. 조회 불가시 변환 중 쓰기 오류로 확인

| 반환 필드 | 내용 |
| --- | --- |
| `ok`, `error` | 검사 결과·오류 |
| `model_family` | 지원 범위 `sd15`. 자동 모델 분류 결과는 아님 |
| `manifest_version`, `mnn_version` | 템플릿·MNN 버전 |
| `unet_quantization` | manifest 정책 식별자. 없으면 `unspecified` |
| `files` | 출력 파일별 `name`, `size_bytes` |
| `output_bytes`, `available_bytes` | 예상 출력 크기·가용 공간. 조회 불가시 가용 공간은 값 없음 |
| `partial_exists` | 동일 출력 경로의 `.partial` 존재 여부 |
| `hqq_iterations` | 적용할 보정 횟수. 양자화 대상이 없으면 값 없음 |

> 신뢰하는 배포 템플릿과의 호환성 검사. 입력 URI 복사·파일 시스템 메타데이터 용량은 별도

### `convert` — 동기 변환

```cpp
sd15::CancellationToken cancellation;
const auto result = sd15::convert(options, [](const sd15::ConvertProgress& progress) {
    // UI에 전달할 진행 정보를 복사
}, cancellation);

// 다른 스레드에서 취소: cancellation.request_cancel();
```

* 호출 스레드에서 입력을 다시 검사하고 `.partial`에 기록. 완료시 출력 폴더 이름으로 변경
* 반환: `status`, `files`, `elapsed_ms`, `error`
* 추가 반환: `hqq_iterations`, `hqq_compute_ms`. 횟수는 검사 완료 후 설정하며 대상이 없으면 값 없음
* `hqq_compute_ms`는 완료된 HQQ 계산 배치의 누적 경과 시간. 초기화·보정·payload/alpha 생성과 진행 확인을 포함하고 입력 변환·파일 I/O는 제외
* 계산 배치 내 진행 콜백 시간은 포함되므로 성능 측정은 빈 콜백으로 실행. 실패·취소 결과는 부분 집계일 수 있음
* 상태: `Succeeded`, `Cancelled`, `Failed`. 실패·취소시 `files`는 비어 있음
* 취소 토큰은 변환마다 새로 만들고 완료까지 유지. 반복 요청 가능
* 청크·HQQ 그룹 경계와 출력 공개 직전에 취소 확인. 검사·파일 I/O 중에는 지연 가능
* 정상 취소는 `Cancelled / None`, 정리 실패는 `Failed / CleanupFailed`
* 기존 `bool convert(options, std::string& error)`와 CLI도 사용 가능

> 최종 취소 확인을 지난 작업은 성공으로 끝날 수 있음. 성공 여부는 최종 결과로 확인

## 3. 진행률 및 시간

| `ConvertProgress` 필드 | 내용 |
| --- | --- |
| `stage` | `Validating` → `Converting` → `Finalizing` → `Completed` |
| `component` | 현재 구성 요소: `text_encoder`, `unet`, `vae_decoder` 등 |
| `completed_bytes`, `total_bytes` | 기록한 출력 바이트·전체량. 검사 중 전체량은 0일 수 있음 |
| `fraction` | 0~1 진행률. 출력 공개 전 최대 0.99, 완료시 1 |
| `elapsed_ms` | 경과 시간 (`steady_clock`) |
| `remaining_ms` | 예상 남은 시간. 추정 불가시 값 없음 |

* 약 100ms 간격으로 알림. 최초·구성 요소 변경·최종 단계는 즉시 전달
* 별도 타이머 없음. 파일 I/O·HQQ 그룹 처리 중에는 알림이 지연될 수 있음
* 남은 시간은 복사/zero, F16·F32 변환, HQQ의 종류별 누적 처리 속도로 계산
* 남은 작업 종류의 측정값이 없으면 추정 보류. 마지막 close/rename 시간은 제외
* UI 표시: 추정 전 `8초 경과`, 추정 후 `약 2분 남음`
* 콜백은 변환 스레드에서 실행. 참조는 콜백 안에서만 유효하므로 전달할 데이터는 복사
* 공개 전 콜백 예외는 `CALLBACK_FAILED`와 출력 정리. 공개 후 알림 예외는 성공 유지

> 진행률은 시간 비율이 아님. 예상 남은 시간은 실행 중 늘어나거나 줄어들 수 있음

## 4. 비동기 작업

```cpp
sd15::ConversionService service; // 네이티브 모듈 수명 동안 유지
const auto started = service.start_conversion(options, [](const sd15::ConversionSnapshot& update) {
    // job_id와 상태를 이벤트 큐에 전달
});

sd15::ConvertError query_error;
const auto state = service.get_conversion_status(started.job_id, &query_error);
// 취소 버튼: service.cancel_conversion(started.job_id);
```

| API | 결과 |
| --- | --- |
| `start_conversion` | 접수시 `job_id`, 접수 실패시 ID 0과 `error` |
| `get_conversion_status` | `job_id`, `state`, `progress`, 선택적인 `result` |
| `cancel_conversion` | 알려진 ID는 `true`, 알 수 없는 ID는 `false` |

* 상태: `Running`, `Cancelling`, `Succeeded`, `Cancelled`, `Failed`. `result`가 있으면 종료 상태
* 입력 오류는 접수 이후 최종 결과로 전달. 이벤트와 조회는 같은 snapshot 구조 사용
* 앱에서 서비스 하나를 공유. 동시 작업은 하나이며 중복 실행은 `BUSY`
* 최근 결과는 다음 작업 접수 또는 서비스 파괴까지 보관
* 알 수 없거나 교체된 ID 조회는 값 없음. 조회 실패는 `query_error`로 구분
* 반복 취소 가능. 요청 직후 조회에 `Cancelling` 반영, 이벤트는 다음 알림 또는 종료시 전달
* 이벤트는 작업 스레드에서 순서대로 전달. 첫 이벤트가 함수 반환보다 먼저 올 수 있으므로 시작 전에 구독
* 콜백에서 상태 조회·취소 가능. 마지막 콜백 중 새 작업 접수는 `BUSY`일 수 있음
* 서비스 파괴시 취소 후 스레드 종료를 기다림. 콜백 안에서 파괴하거나 파괴와 API 호출을 동시에 실행하지 않음

> 완료 후 취소는 결과를 변경하지 않음. 앱 종료 후 상태 복구·이어서 변환은 미지원

## 5. 오류 및 임시 파일

* `ConvertError`: `code`, `message`, 관련 `path`, `tensor`
* 앱 분기는 `error_code_name(error.code)` 사용. 설명·경로·텐서 정보는 비어 있을 수 있음
* 변환 중 C++ 예외는 결과로 반환. OS에 의한 프로세스 종료는 반환 불가

| 오류 코드 | 원인 |
| --- | --- |
| `INVALID_INPUT` | 옵션·safetensors·양자화 값 범위 오류 |
| `INCOMPATIBLE_TEMPLATE` | manifest·버전·구간·레시피 오류 |
| `TENSOR_MISMATCH` | 필수 key 누락, shape·dtype·원본 불일치 |
| `OUTPUT_EXISTS`, `STALE_OUTPUT` | 기존 출력 또는 `.partial` 존재 |
| `BUSY` | 다른 변환·정리·서비스 작업 실행 중 |
| `READ_FAILED`, `WRITE_FAILED` | 파일 열기·읽기·쓰기·종료·공개 실패 |
| `INSUFFICIENT_SPACE` | 사전 공간 부족 또는 쓰기 중 ENOSPC |
| `NON_FINITE_WEIGHT` | 원본 가중치의 NaN/Inf |
| `OUT_OF_MEMORY` | C++ 할당 실패 |
| `CALLBACK_FAILED` | 출력 공개 전 콜백 예외 |
| `CLEANUP_FAILED` | 임시 출력 정리 실패 또는 정리 조건 불충족 |
| `INTERNAL_ERROR` | 그 외 내부 오류 |

### `cleanup_partial` — 중단된 출력 정리

```cpp
const auto cleanup = sd15::cleanup_partial(options);
```

* 앱 재시작 후 `partial_exists` 또는 `STALE_OUTPUT` 확인시 호출
* `.partial/.sd15-converter.json`의 파일 목록을 확인한 뒤 정리
* 반환: `ok`, `error`. `.partial`이 없어도 성공. 최종 출력은 보존
* 표식 없음·손상, 알 수 없는 파일, 링크·하위 폴더는 거부. 재귀 삭제 없음
* 표식 기록 전 종료된 작업은 앱에서 별도 확인

> 앱 소유 경로에서만 사용. 변환·정리는 프로세스 내에서 상호 배제하며 프로세스 간 잠금은 미지원

## 6. Expo 연결

| Expo 이름 | C++ 연결 |
| --- | --- |
| `inspectConversion(options)` | `inspect` |
| `startConversion(options)` | `service.start_conversion` |
| `getConversionStatus(jobId)` | `service.get_conversion_status` |
| `cancelConversion(jobId)` | `service.cancel_conversion` |
| `cleanupPartial(options)` | `cleanup_partial` |
| `conversionUpdated` | `ConversionObserver`의 snapshot |

* 검사·정리는 UI 스레드 밖에서 실행
* JS 작업 ID는 문자열, 값 없는 예상 시간은 `null`로 전달
* 바이트 수·시간은 JS 안전 정수 범위 확인
* 직렬화·이벤트 전달 예외, 권한·백그라운드 실행·모듈 수명은 네이티브 계층에서 처리

> 연결 설계이며 실제 Expo/Kotlin/Swift 바인딩은 미구현

## 테스트

```bash
docker compose -f compose.sd15.yaml build convert test
docker compose -f compose.sd15.yaml run --rm test
```

* 작은 데이터로 검사·진행률·시간 추정·취소·정리·오류·작업 상태와 기존 변환 검사
* 빌드 단계에서도 CTest 실행. 실제 모델 파일은 불필요

2026-10-09 검증:

| 항목 | 결과 |
| --- | --- |
| 테스트 | 60개 통과, Windows 네이티브 Runtime 1개 건너뜀 |
| 저장 텐서 | CLIP 209/209, UNet 1117/1117, VAE 154/154 이름·shape·dtype·바이트 일치 |
| 고정 그래프 | 모든 바이트 일치 |
| 변환 시간 / peak RSS | 203.19초 / 12,184KiB |

* 입력: `v1-5-pruned-emaonly.safetensors`, 템플릿: `sd15-hqq-b128-manual`
* 조건: Docker 64MiB 제한, 입력 청크 1MiB, 단일 실행
* 보고서: `artifacts/tensor-comparison-api-20261009.json`

> PC/Docker 결과이며 Android·Expo 통합 검증은 별도
