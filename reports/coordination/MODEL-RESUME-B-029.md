# MODEL-RESUME-B-029: 예약 원문 S3 이전 실행 + CPU·S3 첫 배치 수집 (완료)

- **결론:** 두 작업 모두 실제로 실행해 끝냈다.
  - 예약 로컬 원문과 서명 파일 **10,714객체(2,247,488,519 B)를 S3 예약 전용 prefix로 올렸다.** 10,714객체 모두 S3가 저장한 SHA-256과 길이로 검증했고, 대표 6객체는 GET으로 되읽어 sha256이 일치했다.
  - 이어서 비예약 ALIO 규정 **첫 배치 452계열을 S3에 수집해 CPU+S3 수집 재개를 입증했다.** 수집량은 응답 835,170,668 B, 저장 979,611,467 B다. 요청은 2,756건이고 모두 200, 순차 처리, 간격은 최소 1.046초였다.
- **결함 1건:** 배치는 1 GiB 쓰기 상한에서 멈췄다. 그러나 정지가 마지막 체크포인트 안에서 일어나 종료 코드가 1이었다. 메타데이터 덮어쓰기까지 상한에 셌기 때문이다. 수집기를 최소 수정했고(6절), 요약은 저장된 상태로 다시 게시했다. 이때 새 요청은 0건이다.
- **하지 않은 것:**
  - 루트 사용 0. STS 루트 조회도 반복하지 않았다.
  - 버킷 정책·보안 설정 변경 0. 유료 서비스 활성화 0.
  - 기존 S3 객체 덮어쓰기 0. 로컬 원본 삭제 0. 보호 해제 0. GPU·EC2 0.

## 1. 인증 (trid-b-corpus)

- **호출자:** `arn:aws:iam::148761639313:user/trid-b-corpus`(`s3_session_client`의 STS 확인). `AWS_PROFILE=trid-b-corpus`과 `--profile trid-b-corpus`를 함께 명시했다.
- **코드 수정 1건:** 환경변수 검사에서 `AWS_PROFILE`을 무조건 거부하던 것을, `--profile`과 같은 값이면 허용하도록 바꿨다. 다른 값이면 거부한다(테스트 추가). 키 환경변수(`AWS_ACCESS_KEY_ID` 등)는 계속 거부한다.
- **버킷 설정** (`bucket-check`, 읽기 전용, 실측):

| 항목 | 값 |
|---|---|
| 리전 | ap-northeast-2 |
| 기본 암호화 | SSE-S3 AES256, BucketKey off, SSE-C 차단 |
| 공개 차단 | BPA 4개 true |
| 객체 소유권 | BucketOwnerEnforced (ACL 비활성) |
| 버전 관리 | 미설정 |
| 서버 접근 로그 | 미설정 |
| 버킷 정책 | 없음(`NoSuchBucketPolicy`) |

- HTTPS는 IAM 조건(`aws:SecureTransport`)으로 강제하고 있다. 버킷 정책은 바꾸지 않았다.

## 2. 예약 원문 이전 (실행 완료)

**업로드 위치** (모두 예약 전용, 비예약 prefix와 분리)
- 로컬 저장소: `corpus/reserved/benchmark-reserve-001/local-b026/`
- 서명 파일: `corpus/reserved/benchmark-reserve-001/signatures/`
- 실행 시각: 2026-11-02T03:54:03Z ~ 04:21:08Z (27분).

| 구분 | 객체 | 바이트 |
|---|---|---|
| 로컬 저장소 (raw 4,825 · text 5,028 · state 852 · 상태·요약·예산 3) | 10,708 | 1,599,115,384 |
| 서명 입력·manifest (148 봉인본 · B-026 · B-027) | 6 | 648,373,135 |
| **합계** | **10,714** | **2,247,488,519** |
| 이전 manifest (`local-b026/_migration-manifest.jsonl`, 같은 내용) | +1 | — |

- **업로드 방식:** `put_new`(If-None-Match: `*`)에 `ChecksumSHA256`를 붙였다. 그래서 S3가 업로드 바이트를 검증한다. 기존 객체는 하나도 바뀌지 않았다.
- **검증 수:** 10,714 / 10,714.
  - HeadObject(`ChecksumMode=ENABLED`)가 돌려준 **S3 저장 ChecksumSHA256**과 로컬 sha256을 대조했다. 길이도 대조했다.
  - 단일 PUT이라 전체 객체 체크섬이다. 사용자 메타데이터 비교가 아니다.
  - 암호화는 AES256 10,714건, 스토리지 클래스는 STANDARD 10,714건이다.
- **독립 대조:** ListObjectsV2로 다시 셌다.
  - `local-b026/`: 10,709객체, 1,603,841,529 B = 저장소 10,708 + manifest 1.
  - `signatures/`: 6객체, 648,373,135 B.
- **대표 GET 되읽기 6건 sha256 일치:** `budget.json`, raw HWP 1건, text 2건(그중 큰 원문 1건), state 1건, B-027 서명 입력 1건.
- **로컬 manifest:** `MODEL-RESUME-B-029/migration-manifest.jsonl`. 10,714행, 4,726,145 B, sha `94a51efffc3909e06e05d732ea9de959f782b2181e0c58a02aff20936b74c208`이다. 행마다 key·bytes·sha256·verified·SSE·클래스를 담았다. 크기 때문에 git에는 넣지 않았다. 같은 내용이 S3 `local-b026/_migration-manifest.jsonl`에 있다.
- **로컬 원본:** 2 GiB 보존물은 그대로다. manifest 해시는 변경 없이 S3 사본과 같다.
  - 148 `0ed3cc07…`
  - B-026 `584da062…`
  - B-027 `c3310d7f…`

## 3. 첫 비예약 배치 (실행 완료, 학습 hold)

- **실행:** `rules-batch --profile trid-b-corpus --interval 1.0`
  - 위치: `corpus/raw/institutions/alio-rule-b028/`
  - 시각: 2026-11-02T04:21:28Z ~ 05:24:14Z
- **대상:** 인벤토리 ALIO 규정 7,990계열에서 예약 1,000을 뺀 6,990계열이다. 업무군을 번갈아 처리했다. 예약 계열 요청은 0이다.
- **요청 규약 준수** (`logs/requests-www.alio.go.kr.jsonl` 실측)
  - 2,756건: robots 1, 상세 453, 파일 2,302.
  - 모두 200. 순차 1개씩, 최소 간격 1.046초. 차단·제한 응답 0. 프록시·재시도 없음.

| 항목 | 값 |
|---|---|
| 상태 기록된 계열 | 452 (text_acquired 448, partial 4) |
| 중단돼 미완인 계열 | 1 (상태 없음 → 다음 실행에서 처음부터) |
| 업무군 | 9개 군, 각 49~50계열 |
| 파일 | 2,297 (hwp 1,448 · pdf 721 · hwpx 70 · zip 57 · xls 1). 다운로드 전부 ok |
| 추출 | ok 2,293, unextractable 3, failed 1. ZIP 내부 1,038 전부 ok |
| 텍스트 | 61,348,440자 |
| S3 저장 | raw 2,754객체 834,368,365 B · text 3,278객체 136,410,818 B · state 452 · 요약·예산·목록 3 = **6,487객체, 979,611,467 B** |
| 응답 바이트 | 835,170,668 (1 GiB 상한 이내) |
| 사용권 | `unconfirmed` 452 → 모든 상태에 `training_hold: [licence_unconfirmed]`, `reservation: null` |

- **요약 파일:** `reports/coordination/alio-rule-b028-acquisition-summary.json`.
- **커밋하지 않은 파일:** 계열별 목록 `…-acquisition.jsonl`. 파일명과 시행 문구가 들어 있어 예약 상태 목록과 같은 기준을 적용했다.
- **아직 하지 않은 것:**
  - 이 배치의 PDF 721건에는 B-030의 이미지 페이지 OCR 점검을 아직 적용하지 않았다. 배치는 B-029 코드로 추출했다.
  - 학습 승격은 사용권 확인 전이라 하지 않는다.

## 4. 한도

| 한도 | 사용 |
|---|---|
| S3 전체 10 GiB | 2,247,488,519 + 979,611,467 = 3,227,099,986 B (약 3.0 GiB) |
| 첫 배치 응답 1 GiB | 835,170,668 B |
| 첫 배치 S3 쓰기 1 GiB | 내용(raw+text) 970,779,183 B. 체크포인트 덮어쓰기는 별도 |
| 로컬 신규 임시 512 MiB | 배치는 메모리에서 S3로 바로 썼다. 로컬 디스크 증가는 요청 로그와 요약 파일뿐이다 |

## 5. 비용 (서울 공식 단가, B-028 `price-evidence.json`)

| 항목 | 양 | USD |
|---|---|---|
| 이전 PUT | 10,715 | 0.048 |
| 이전 HEAD·GET | 약 10,720 | 0.004 |
| 배치 PUT (객체 6,487 + 체크포인트 덮어쓰기 약 600 [추정]) | 약 7,100 | 0.032 |
| 배치 GET·LIST·HEAD (요약 재게시 453 등) | 약 500 | 0.000 |
| 저장 (월) | 약 3.0 GiB | 0.075/월 |
| 인터넷 반출 | 되읽기 6건·상태 452건, 수 MB | 0.001 미만 |
| **첫 달 합계** | | **약 0.16 USD ≈ 230원** (1,450원/USD [추정 환율]) |

- 스토리지 한도 1만원, 추가 예산 30만원 이내다. 실제 청구액은 아직 보지 못했다. Cost Explorer 권한이 없다.
- **관측 서비스 권고** (활성화 안 함): 서버 접근 로그와 버전 관리가 미설정이다.
  - 접근 로그는 저장비만 든다.
  - CloudTrail 데이터 이벤트를 세 prefix에만 켜면, 이번 규모 약 3만 이벤트 ≈ 0.03 USD다.
  - 켤지는 감독이 정한다.

## 6. 수정한 결함

- **`S3Store` 쓰기 상한 계산:**
  - **현상:** state·요약·예산·acquisition 목록은 10계열마다 덮어쓴다. 이를 쓰기 상한에 세면서 카운터가 1,072,999,926 B에 이르렀다. 실제 내용은 970,779,183 B였다. 마지막 체크포인트의 `publish`가 `Stop`을 냈고 루프 밖에서 잡히지 않아 종료 코드 1로 끝났다.
  - **영향:**
    - 데이터 손실은 없다. 계열 상태 452개는 모두 저장됐다.
    - 응답 1 GiB 상한도 지켰다.
    - 실행 중에 쓴 요약과 목록은 2계열 늦은 상태였다.
  - **수정:** `put`(덮어쓰는 메타데이터)은 상한에 세지 않는다. raw·text만 센다. 최소 수정이다.
  - **조치:** 저장된 452 상태로 `publish`를 한 번 다시 실행해 요약·목록·예산을 고쳤다(ALIO 요청 0). 예산은 보수적으로 잡았다. 응답 바이트는 마지막 체크포인트 값과 저장 raw 중 큰 값이고, 요청 수는 요청 로그 실측이다.
- **`AWS_PROFILE` 검사:** 1절과 같다.

## 7. 재현

```
export AWS_PROFILE=trid-b-corpus
R="uv run -q --no-project --python 3.12 --with boto3 --with 'botocore[crt]' --with requests python"
$R scripts/collect_institutions.py bucket-check --profile trid-b-corpus
$R scripts/collect_institutions.py reserved-migrate --profile trid-b-corpus --local data/benchmark-reserve-001/local \
   --extra data/benchmark-reserve-001/{sign-input-local-852-b026.jsonl.gz,signatures-candidate-local-852-b026.jsonl,sign-input-local-reextract-b027.jsonl.gz,signatures-candidate-local-reextract-b027.jsonl,sign-input-partial-148.jsonl,signatures-partial-148.jsonl} \
   --manifest-out reports/coordination/MODEL-RESUME-B-029/migration-manifest.jsonl --roundtrip <6 keys>
$R scripts/collect_institutions.py rules-batch --profile trid-b-corpus --interval 1.0   # (+ --with ko-pii==1.22.15)
```

- 같은 이전을 다시 실행하면 같은 바이트는 완료로 보고, 다른 바이트면 멈춘다.
- 같은 배치 명령을 다시 실행하면 452계열을 건너뛰고 다음 계열부터 받는다. S3 `budget.json`에 응답 바이트가 누적된다.
- 다음 배치는 감독 지시 후에 실행한다.

## 8. 산출물

- **코드:** `scripts/collect_institutions.py` (`S3Store.put`, `AWS_PROFILE` 검사).
- **테스트:** `tests/test_collect_institutions.py`.
- **보고서:** `alio-rule-b028-acquisition-summary.json`. 이전 manifest는 로컬과 S3에만 있다(2절).
- **S3:** 위 세 prefix.

SUPERVISION_REPORT_READY MODEL-RESUME-B-029
