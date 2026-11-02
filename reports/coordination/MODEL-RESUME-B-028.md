# MODEL-RESUME-B-028: CPU 수집·정제 + S3 저장 전환 (구현 완료, 인증 차단으로 실제 업로드 미실행)

- **결론:** 실행 완료분과 인증 블로커를 나눠 적는다.
  - **완료:** 수집기의 S3 경로를 구현했고, 오프라인으로 검증했다. 실제 AWS 경로도 스모크했다(비루트 세션, HTTPS, 권한 거부를 정상 기록). 이전 계획 건조 실행, 공식 단가·비용 산정, 필요 IAM 권한 정의, 문서까지 마쳤다.
  - **차단:** corpus prefix에 권한이 있는 **비루트 프로필이 이 기기에 없다.** 그래서 실제 업로드·체크섬 검증·GET 되읽기·다음 배치 수집·버킷 보안 설정 확인은 실행하지 못했다.
- **하지 않은 것:**
  - 루트 사용 0. 새 로그인·권한 추가 0. 환경변수 자격증명 우회 0.
  - 버킷·계정 보안 설정 변경 0. 유료 서비스 활성화 0.
  - S3 쓰기 0. ALIO 요청 0. 로컬 원문 삭제 0.
  - GPU·EC2 0. A 운영 코드 변경 0.

## 1. 인증·권한 확인 (실측)

| 프로필 | 식별자 | corpus 사용 |
|---|---|---|
| `default` | 루트(장기 키) | **금지** |
| `kms2026` | `arn:aws:iam::148761639313:root`. 이름만으로는 알 수 없어 식별 목적으로 `sts get-caller-identity`를 1회 호출했다. 다른 호출은 하지 않았다 | **금지(루트)** |
| `trid-aws-eval-006-operator` | `user/trid-aws-eval-006-operator` | 버킷 HeadBucket 403. GetBucketLocation·암호화·공개차단·소유권·버전·로깅·정책상태, `corpus/reserved/`·`corpus/clean/` ListBucket 모두 AccessDenied. 자기 정책 조회(iam:ListAttached/UserPolicies)도 거부 |
| `trid-aws-eval-006-c-runner` | `user/trid-aws-eval-006-c-runner` | 위와 같이 전부 AccessDenied |

- 관측 도구 읽기(CloudTrail DescribeTrails, CloudWatch ListMetrics, 계정 수준 공개 차단)도 operator로 모두 거부됐다. 그래서 **접근 로그·CloudTrail 데이터 이벤트·CloudWatch 현황은 확인하지 못했다.**
- 기존 `trid-sagemaker-lora` 역할은 corpus 권한이 있지만 SageMaker용 역할이라 이 기기에서 쓸 주체가 아니다.

**필요한 것(감독·계정 관리자 결정)**
1. corpus 전용 비루트 IAM 주체 1개. 예: 사용자 또는 역할 `trid-b-corpus`와 이 기기의 `login_session` 프로필.
2. 위 주체에 붙일 최소 권한: `MODEL-RESUME-B-028/required-iam-policy.json`.
   - 버킷 설정 읽기 7종.
   - ListBucket. 아래 세 prefix로 조건을 건다.
   - Put·GetObject. 세 prefix에만, TLS 조건을 붙이고, **Delete는 없다**.
     - `corpus/reserved/benchmark-reserve-001/local-b026/`
     - `corpus/reserved/benchmark-reserve-001/signatures/`
     - `corpus/raw/institutions/alio-rule-b028/`
   - 관측 읽기(선택): `s3:GetAccountPublicAccessBlock`, `cloudtrail:Describe/GetTrailStatus/GetEventSelectors`, `cloudwatch:ListMetrics/GetMetricData`.
3. 버킷 기본 암호화가 SSE-KMS라면 해당 키에 `kms:GenerateDataKey`·`kms:Decrypt`(kms:ViaService s3.ap-northeast-2)를 더해야 한다. 지금은 암호화 설정을 읽을 권한이 없어 SSE-S3인지 SSE-KMS인지 모른다.
4. `aws login` 방식이면 그 주체에 `SignInLocalDevelopmentAccess` 관리형 정책이 필요하다(signing-in-to-aws 스킬). 이번에는 로그인을 실행하지 않았다.

## 2. 구현 (기존 수집기·S3Store·버킷·계약 재사용, 새 인프라 없음)

**`scripts/collect_institutions.py`**
- **`s3_session_client(profile)`:**
  - `--profile`이 없으면 거부한다. 환경변수 자격증명(`AWS_ACCESS_KEY_ID` 등)이 있으면 거부한다.
  - sts로 호출자를 확인하고 `:root`면 거부한다. HTTPS 엔드포인트만 쓴다.
  - boto3는 이 함수 안에서만 import한다. 로컬·건조 실행 경로는 SDK를 불러오지 않는다.
- **`S3Store`:**
  - raw는 B-003처럼 한 번만 쓴다(If-None-Match, 다른 바이트면 `<stem>.<sha16>.<ext>`).
  - text도 한 번만 쓰도록 바꿨다(`put_new`).
  - 실행당 쓰기 바이트 상한(`max_disk_bytes`)을 넘으면 쓰기 전에 Stop한다.
  - `exists`를 추가했고, 응답 본문은 컨텍스트로 닫는다(SDK 스킬 S3 참고).
- **`reserved-migrate`:**
  - 로컬 저장소와 서명 파일을 예약 전용 prefix로 올린다. `put_new`라서 기존 객체를 바꾸지 않는다. 같은 바이트는 완료로 보고, 다른 바이트면 중단한다.
  - 모든 객체를 HeadObject(`ChecksumMode=ENABLED`)로 저장 SHA-256과 길이를 대조한다.
  - `--roundtrip` 키는 GET으로 되읽어 sha256을 대조한다.
  - 객체 manifest는 로컬(새 파일)과 S3(`_migration-manifest.jsonl`)에 남긴다.
  - 계획이 `--max-upload-bytes`(기본 10 GiB)를 넘으면 요청하기 전에 거부한다.
  - `--dry-run`은 오프라인으로 계획만 만든다.
- **`bucket-check`:** 읽기 전용 설정 조회. 권한이 없는 항목은 오류 코드로 기록한다.
- **`rules-batch`:**
  - 말뭉치 인벤토리의 ALIO 규정 계열에서 **예약 1,000계열을 뺀** 다음 배치를 받는다.
  - B-026의 한도 루프를 공유한다(`bounded_collect`로 일반화). 동시 1, 간격 1초 이상, robots 확인, 401·403·429·451·접근 제한·연속 네트워크 실패면 체크포인트 후 정지, 재시도 없음.
  - 첫 배치 응답은 1 GiB 이하(`min(--max-bytes, 1 GiB)`), S3 쓰기는 `--max-s3-bytes`(기본 1 GiB) 이하다.
  - 저장 위치는 `corpus/raw/institutions/alio-rule-b028/`이다. 모든 상태에 `training_hold: [licence_unconfirmed]`를 붙이고, `reservation: null`로 둔다.
- **`reserved_local`:** 공용 루프를 호출하는 얇은 래퍼가 됐다. B-026 동작과 기존 테스트는 그대로다.

**`tests/test_collect_institutions.py`(가짜 S3) 6개 추가**
- 업로드가 원본을 바꾸지 않는지, 같은 바이트로 다시 실행해 이어지는지, 다른 바이트에서 중단하고 저장본을 지키는지.
- 모든 객체 체크섬 검증과 되읽기.
- 상한을 넘는 계획은 요청 전에 거부되는지.
- S3 쓰기 상한에서 쓰기 전에 Stop하는지.
- 환경변수 자격증명과 프로필 누락이 SDK를 불러오기 전에 거부되는지. 루트 식별자가 거부되는지.
- 배치에서 예약 계열이 제외되는지.

## 3. 오프라인 검증과 실제 경로 스모크

- **이전 건조 실행(실제 로컬 파일):** 10,714 객체, 2,247,488,519 B(10 GiB 상한 이내). PUT 10,714회와 HEAD 10,714회가 필요하다(`MODEL-RESUME-B-028/migration-plan-summary.json`).

| 구분 | 객체 | 바이트 |
|---|---|---|
| raw | 4,825 | 1,366,009,154 |
| text | 5,028 | 218,592,393 |
| state | 852 | 7,255,690 |
| 상태·요약·예산 파일 | 3 | — |
| 서명 입력·manifest 6개 (148 봉인본, B-026, B-027 사본) | 6 | 648,373,135 |

- **다음 배치 선택(오프라인):** 비예약 ALIO 규정 6,990계열(예약 1,000 제외). 업무군을 번갈아 처리한다. 1 GiB면 B-026 실측(계열당 약 1.6 MB)으로 약 650계열이다 [추정]. SDK는 불러오지 않았다.
- **실제 경로 스모크:** `uv run --with boto3 --with 'botocore[crt]' … bucket-check --profile trid-aws-eval-006-operator`.
  - 비루트 세션과 HTTPS 클라이언트가 만들어졌다. 7개 설정 조회는 모두 `AccessDenied`로 기록됐다(1절과 같음).
  - `login_session` 프로필은 `botocore[crt]`가 있어야 해서 실행 명령에 넣었다(.venv에는 boto3가 없다).
- **전체 단위 테스트:** 커밋 전 게이트로 1회 실행했다.

## 4. 비용 (서울, AWS 공식 Price List에서 현재 단가 확인)

**단가** (`MODEL-RESUME-B-028/price-evidence.json`)
- 출처 공개 시점: S3 2026-10-15, 데이터 전송 2026-09-16, CloudTrail 2026-09-11.

| 항목 | 단가(USD) |
|---|---|
| S3 Standard 저장 (첫 50 TB) | 0.025 / GB-월 |
| PUT·COPY·POST·LIST | 0.0045 / 1,000건 |
| GET·HEAD 등 | 0.0035 / 10,000건 |
| 인터넷 반출 (월 10 TB까지, 전역 무료 한도 초과분) | 0.126 / GB |
| CloudTrail 데이터 이벤트 | 0.000001 / 이벤트 |

- 가격 페이지: <https://aws.amazon.com/s3/pricing/>. 데이터는 `pricing.us-east-1.amazonaws.com/offers/v1.0/aws/{AmazonS3,AWSDataTransfer,AWSCloudTrail}/current/ap-northeast-2/index.json`을 `scripts/logged_get.py`로 받았다.

**이번 계획 추정** (요청 수는 계획 기준, 배치는 B-026 계열당 객체 수로 추정)

| 항목 | 양 | 비용(USD) |
|---|---|---|
| 이전 PUT | 10,715건 | 0.048 |
| 이전 HEAD·GET 검증 | 약 10,720건 | 0.004 |
| 첫 배치 PUT | 약 8,200건 [추정] | 0.037 |
| 저장 | 이전 2.09 GiB + 배치 약 1.2 GiB [추정] | 월 약 0.082 |
| 반출 | 되읽기 몇 MB | 0.001 미만 |
| **합계(첫 달)** | | **약 0.17 USD ≈ 250원** (1,450원/USD 가정 [추정 환율]) |

- 이번 스토리지 작업 한도 1만원, 추가 예산 30만원 이내다.
- S3 10 GiB 상한을 꽉 채워도 저장비는 월 약 0.25 USD다.

**관측 권고** (현황을 볼 권한이 없어 확인 전, 활성화하지 않음)

| 항목 | 권고 | 비용 |
|---|---|---|
| 서버 접근 로그 | 별도 로그 버킷(SSE 암호화)에 남기기 | 로그 전달은 무료, 저장비만 든다 |
| CloudTrail S3 데이터 이벤트 | 위 세 prefix에만 이벤트 선택기로 기록. 트레일·로그는 KMS 암호화 | 이번 규모 약 3만 이벤트 ≈ 0.03 USD |
| CloudWatch | 무료 일일 저장 지표(BucketSizeBytes·NumberOfObjects)로 충분하다. 요청 지표는 유료라 제안만 한다 | — |
| 보안 기본값 | 버킷 정책의 `aws:SecureTransport` 거부, Block Public Access, ACL 비활성(BucketOwnerEnforced)이 이미 켜져 있는지 `bucket-check`로 먼저 확인 | — |

## 5. 보존·격리 상태 (변경 없음)

- **로컬 보존물:** 로컬 2 GiB 보존물(`data/benchmark-reserve-001/`)은 그대로다. 저장소 1,599,115,384 B와 서명 파일 6개다.
- **manifest:**
  - 봉인 148 `0ed3cc07…`, B-026 852 `584da062…`, B-027 재추출분 `c3310d7f…`는 그대로다.
  - 이전 계획에서는 원본 이름 그대로 `signatures/` 아래 사본으로만 올린다.
- **B-027 한계 기록:** 843 전체, 9 부분이다. 보호 파일 16건(배포용 HWP 13, DRM 3)은 보호를 풀지 않았고 그대로 남아 있다.
- **예약 원문 위치:** 예약 원문은 예약 전용 prefix로만 간다. 학습용 `corpus/clean`이나 `corpus/raw`(비예약)으로 올리지 않는다.
- **사용권:** 비예약 배치는 `licence_unconfirmed`라 전부 hold다. 사용권은 승인하지 않았다.

## 6. 차단 해소 후 실행 순서 (명령 준비 완료)

```
R="uv run -q --no-project --python 3.12 --with boto3 --with 'botocore[crt]' --with requests --with ko-pii==1.22.15 python"
$R scripts/collect_institutions.py bucket-check --profile <corpus-profile>
$R scripts/collect_institutions.py reserved-migrate --profile <corpus-profile> --local data/benchmark-reserve-001/local \
   --extra data/benchmark-reserve-001/{sign-input-local-852-b026.jsonl.gz,signatures-candidate-local-852-b026.jsonl,sign-input-local-reextract-b027.jsonl.gz,signatures-candidate-local-reextract-b027.jsonl,sign-input-partial-148.jsonl,signatures-partial-148.jsonl} \
   --manifest-out reports/coordination/MODEL-RESUME-B-028/migration-manifest.jsonl \
   --roundtrip corpus/reserved/benchmark-reserve-001/local-b026/budget.json <raw 1개> <text 1개> <state 1개>
$R scripts/collect_institutions.py rules-batch --profile <corpus-profile> --interval 1.0
```

- **검증 순서:** 객체 수·바이트·체크섬은 `reserved-migrate` 출력과 manifest로 확인한다. 되읽기 결과도 같은 출력에 있다. 배치는 `alio-rule-b028-acquisition-summary.json`(응답·쓰기 바이트, 정지 사유)으로 확인한다.

## 7. 산출물

- **코드:** `scripts/collect_institutions.py`.
- **테스트:** `tests/test_collect_institutions.py`.
- **문서:** `docs/CORPUS_CONTRACT.md` 3절(S3 전환 규칙).
- **보고서 폴더 `MODEL-RESUME-B-028/`:** `required-iam-policy.json`, `migration-plan-summary.json`, `price-evidence.json`.
- **README:** 바꿀 내용이 없어 고치지 않았다.
- **로그 메모:** 처음 단가 파일 크기를 확인할 때 `curl -I` 한 번을 `logged_get`을 거치지 않고 보냈다(가격 API HEAD 1건). 이후 요청은 모두 `logged_get`을 거쳤다.

SUPERVISION_REPORT_READY MODEL-RESUME-B-028
