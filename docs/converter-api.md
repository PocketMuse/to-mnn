# SD1.5 Converter API

## 1. 개요
F32/F16 safetensors와 템플릿으로 MNN 모델을 만든다.

* 입력: Checkpoint + `graph.bin` + `manifest.json`
* 출력: `{text_encoder,unet,vae_decoder}.mnn`과 필요한 외부 가중치
* 지원: FP16 템플릿 v1, HQQ 템플릿 v2, MNN 3.6.1 레이아웃
* 공개 헤더: [converter.hpp](../cpp/converter/include/sd15/converter.hpp), [hqq.hpp](../cpp/converter/include/sd15/hqq.hpp)

## 2. API 목록

`sd15` 네임스페이스 기준

| API | 용도 |
| --- | --- |
| `convert` | 동기 변환·진행 알림 |
| `CancellationToken::request_cancel` / `is_requested` | 동기 변환 취소 요청·확인 |
| `cleanup_partial` | 중단된 임시 출력 정리 |
| `error_code_name` | 오류 코드를 문자열로 변환 |
| `quantize_hqq` / `hqq_error_message` | 저수준 HQQ 그룹 계산·오류 설명 |

## 3. 공통 입력 — `ConvertOptions`

| 필드 | 타입 | 기본값 | 조건 |
| --- | --- | --- | --- |
| `checkpoint` | `std::string` | 빈 문자열 | F32/F16 safetensors 로컬 파일 |
| `template_dir` | `std::string` | 빈 문자열 | 배포 템플릿 디렉터리 |
| `output_dir` | `std::string` | 빈 문자열 | 아직 존재하지 않는 출력 디렉터리 |
| `chunk_bytes` | `std::size_t` | `kDefaultChunkBytes` = 1MiB | 4바이트~4MiB, 4의 배수 |
| `hqq_iterations` | `int` | `kHqqIterations` = 20 | 0~20 |
| `thread_count` | `unsigned` | 1 | 1~`kMaxThreadCount` = 8, 호출 스레드 포함 |

```cpp
sd15::ConvertOptions options;
options.checkpoint = "/app/models/model.safetensors";
options.template_dir = "/app/templates/sd15-hqq-b128";
options.output_dir = "/app/models/converted";
options.thread_count = 4;
```

기존 `output_dir`은 덮어쓰지 않으며 `OutputExists`로 실패한다.

> w8 != w8 + hqq=0

## 4. 세부사항

### 4-1. 실행 스레드

| 기능 | 실행 방식·주의점 |
| --- | --- |
| `convert` | 동기 변환. UI 밖의 작업 스레드에서 호출하며 진행 콜백도 같은 스레드에서 실행 |
| `cleanup_partial` | 동기 파일 정리. UI 밖의 작업 스레드에서 호출 |

### 4-2. 동기 변환·진행 알림

호출자가 준비한 작업 스레드에서 실행하며 반환될 때 변환·정리가 끝난다.

```cpp
sd15::CancellationToken cancellation;
const auto result = sd15::convert(options, [](const sd15::ConvertProgress& progress) {
    // UI에 전달할 진행 정보는 복사해 큐에 넣는다.
}, cancellation);
if (result.status == sd15::ConvertStatus::Failed) {
    // result.error 확인
}
```

* 진행률은 출력 바이트 기준이며 남은 시간은 추정값이다. `fraction=1` 알림보다 최종 반환값의 `status`를 완료 판단에 사용한다.
* 콜백은 `convert` 호출 스레드에서 실행한다. 참조는 콜백 동안만 유효하므로 보관할 데이터는 복사한다.
* 공개 전 콜백 예외는 `CallbackFailed`로 변환을 중단한다. 출력 공개 후 알림 예외는 성공 결과를 변경하지 않는다.
* Expo 연결 계층은 작업 실행·토큰 수명·진행 이벤트·최종 결과 전달을 담당한다. JS에는 비동기 호출로 노출하고 작업 스레드에서 동기 `convert`를 호출한다.
* 기존 `ConversionService` 호출자는 동기 `convert(options, callback, token)`로 이전한다. 작업 ID·상태 보관이 필요하면 호출 계층에서 관리한다.

> 입력·템플릿·저장 공간 검사는 `convert()` 내부에서 수행한다. HQQ 계산 병렬화는 `thread_count`로 유지한다.

### 4-3. 취소·수명·중단된 파일

* 다른 스레드나 콜백에서 `cancellation.request_cancel()`로 요청한다. 취소 경로를 변환과 같은 직렬 작업 큐에 넣지 않는다.
* 파일 I/O 중에도 취소 요청은 가능하다. 실제 중단은 I/O·검사 이후 취소 확인 지점에서 이루어지며, 마지막 확인을 지난 요청은 성공으로 끝날 수도 있다.
* 취소 요청은 완료가 아니다. `convert` 반환 후 `ConvertStatus::Cancelled`를 확인한다.
* 토큰은 변환마다 새로 만들고 `convert` 반환까지 유지한다. 앱·모듈 종료 시에도 취소 요청만으로 토큰이나 콜백이 참조하는 자원을 해제하지 않는다.
* 변환·정리는 프로세스 내 하나만 실행한다. 완료 전 중복 실행은 `Busy`이며 프로세스 간 잠금은 없다.
* 기존 `.partial`은 자동으로 덮어쓰지 않는다. `convert()`의 `StaleOutput` 오류 확인 후 앱 소유 경로에서 `cleanup_partial(options)`로 정리한다. 소유 표식이 없거나 알 수 없는 파일·링크·하위 폴더가 있으면 정리를 거부한다.
* 저수준 `quantize_hqq`를 직접 호출하면 유한 입력과 버퍼 용량을 호출자가 보장한다. `payload`는 입력 원소 수만큼의 바이트, `alpha`는 float 2개가 필요하며 실패한 출력은 사용하지 않는다.

## 5. 공통 오류

`ConvertError`는 `ErrorCode code`, `std::string message`, `path`, `tensor`로 구성한다. 설명·경로·텐서 정보는 비어 있을 수 있다. 변환 중 예외는 결과로 전달하며 OS에 의한 프로세스 종료는 반환할 수 없다.

| `ErrorCode` / 문자열 | 원인 |
| --- | --- |
| `None` / `NONE` | 오류 없음 |
| `InvalidInput` / `INVALID_INPUT` | 옵션·safetensors·양자화 값 범위 오류 |
| `IncompatibleTemplate` / `INCOMPATIBLE_TEMPLATE` | manifest·버전·구간·레시피 오류 |
| `TensorMismatch` / `TENSOR_MISMATCH` | 필수 key 누락, shape·dtype·원소 수 불일치 |
| `OutputExists` / `OUTPUT_EXISTS` | 기존 출력 디렉터리 |
| `StaleOutput` / `STALE_OUTPUT` | 기존 `.partial` |
| `Busy` / `BUSY` | 프로세스 내 변환·정리와의 경합 |
| `ReadFailed` / `READ_FAILED` | 파일 열기·읽기 실패 |
| `WriteFailed` / `WRITE_FAILED` | 출력 열기·쓰기·닫기·공개 실패 |
| `InsufficientSpace` / `INSUFFICIENT_SPACE` | 사전 공간 부족 또는 쓰기 중 ENOSPC |
| `NonFiniteWeight` / `NON_FINITE_WEIGHT` | 원본 가중치의 NaN/Inf |
| `OutOfMemory` / `OUT_OF_MEMORY` | C++ 할당 실패 |
| `CallbackFailed` / `CALLBACK_FAILED` | 출력 공개 전 진행 콜백 예외 |
| `CleanupFailed` / `CLEANUP_FAILED` | 임시 출력 정리 실패 또는 정리 조건 불충족 |
| `InternalError` / `INTERNAL_ERROR` | 그 외 내부 오류 |
