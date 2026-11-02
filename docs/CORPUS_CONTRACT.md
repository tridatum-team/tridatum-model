# trid 말뭉치 계약: raw → versioned clean → tokens

소유: B(원문 수집·정제). 근거: `reports/coordination/B-001.md`, `B-001-review.md`, `A-001.md` 6절. 이 계약을 바꾸려면 새 `clean_version`을 만들고 총괄 검토를 받는다. 기존 버전은 덮어쓰지 않는다.

## 1. 단계와 소유

| 단계 | 소유 | 입력 → 출력 | 하는 일 | 하지 않는 일 |
|---|---|---|---|---|
| raw | B | 원 출처 → `corpus/raw/<source>/…` | 원본 바이트 보존, URL·revision·바이트 해시 기록 | 변환, 덮어쓰기 |
| clean | B | raw → `corpus/clean/<source>/<clean_version>/` | 출처 내부 정제, PII 치환, 사용권 판정·격리, 출처 내부 중복 제거, 개발용 분할 | 출처 간 중복 제거, 평가 오염 제거, 토큰화 |
| tokens | A | clean manifest → `corpus/tokens/<mix_version>/` | 출처 간 exact/near 중복 제거, 개발 평가 오염 제거, 분할 경계 재검증, 토큰화, 출처별 예산 | 원본 재수집, PII 정책 변경 |

평가 문서는 총괄 소유다. 최종 비공개 평가 원문·정답은 A·B에 전달하지 않는다. 오염 제거는 공개 개발 평가용 기존 필터(단어 8-gram 해시)를 재사용한다. 이 해시 목록은 암호화된 비밀 평가셋이 아니다.

## 2. S3 배치

버킷 `s3://trid-training-148761639313`(ap-northeast-2, 비공개, SSE AES256).

```
corpus/raw/<source>/<원본 경로>                         불변 원본
corpus/clean/<source>/<clean_version>/docs-*.jsonl.gz    학습 후보 행
corpus/clean/<source>/<clean_version>/quarantine-*.jsonl.gz  사용권 격리 행(학습 투입 금지)
corpus/reserved/<reservation>/…                          평가 예약 문서(원문·추출 텍스트·상태). 학습 금지
corpus/state/<source>/<clean_version>/…                  샤드/파일 manifest, 재개 상태, 완료 마커
corpus/state/<source>/<index 파일>                       수집 대상 목록(재개 가능)
corpus/code/<source>/<clean_version>/…                    SageMaker 실행 코드 묶음
corpus/tokens/<mix_version>/…                             A 소유
corpus/sm-output/                                         SageMaker 기본 출력
```

`<source>`: `fineweb2-ko`, `assembly-minutes`, `institutions`, `prism`, `korea-kr-boards`, `local-gov-press`.

버전 없는 `corpus/clean/<source>/docs-*.jsonl.gz`와 `corpus/state/<source>/*.done.json`, `shard-*.json`은 **clean v0**(파일럿)이다. NFC 변환만 적용했고 PII를 처리하지 않았다. 학습에 쓰지 않으며 v1 완료 판정에도 재사용하지 않는다.

## 3. 행 스키마 (clean-v1, clean-v2 공통)

gzip JSONL, 한 줄에 한 문서, UTF-8, NFC.

| 필드 | 필수 | 내용 |
|---|---|---|
| `id` | O | `<source>:<원본 id>`. 전체 출처에서 유일하고, 재실행해도 같다 |
| `group` | O | 중복 판정·분할 단위(문서 또는 문서 계열) |
| `kind` | O | 출처 안 세부 유형(`web`, `assembly_plenary`, `regulation` …) |
| `title`, `text` | O | 정제 후 텍스트(PII 치환 반영). 문단 구분은 `\n` |
| `source` | O | 원문 URL |
| `license` | O | 행 단위 사용권. 데이터셋 배포 라이선스와 원문 사용권을 구분해 적는다 |
| `attribution` | O | 사용권이 요구하는 출처 표시 |
| `date` | O | `YYYY-MM-DD` 또는 `null` |
| `sha256` | O | `text`(UTF-8)의 SHA-256 |
| `collected_at`, `extractor` | O | 수집 시각(UTC), 추출 방식 |
| `clean_version` | O | `clean-v1` 또는 `clean-v2` |
| `flags` | O | 목록. 예: `pii_redacted`, `lines_dropped`, `header_unparsed`, `quarantine:<사유>` |
| `split` | O | `train` / `valid` / `test`. 3절 분할 규칙 참고 |
| `provenance` | O | 확인된 값만 기록하고, 모르면 키를 생략한다. `institution`, `institution_id`, `institution_type`(중앙부처/공기업/준정부기관/기타공공기관/지방공기업/국회/web), `domain`(업무 분야, 근거가 있을 때만), `doc_type`, `doc_family`(개정본 묶음), `published_date`, `effective_date`, `revision`, `site`(웹 호스트) |
| `meta` | O | 출처별 원본 정보. `source_split`(업스트림 분할명), `pii_redactions`(라벨별 치환 수) |

분할 규칙:
- `split`은 `sha256("split:"+group)` 상위 32비트를 10,000으로 나눈 버킷으로 정한다. `test` 0.5%, `valid` 0.5%, 나머지는 `train`이다.
- 이 분할은 **개발용**이며 독립 최종 시험이 아니다.
- 기관·문서 계열·시점 단위 보류(최종 시험)는 총괄 목록으로 tokens 단계에서 적용한다. B는 이를 위해 `provenance`를 채운다.
- 업스트림 분할명은 `meta.source_split`에 보존한다. FineWeb-2의 HF `test` 분할을 개발 valid로 쓰는 경우에는 그 사실을 manifest에 적는다.

평가 예약 문서(BENCHMARK-SCALE-001, `reports/coordination/benchmark-reserve-*.jsonl`, 총괄 소유·불변):
- 예약 단위는 `doc_family_id`다. 그 계열의 모든 개정본·첨부·압축 안 파일이 학습 제외 대상이다.
- B는 예약 계열의 원문을 `corpus/reserved/<reservation>/`에만 쓴다. 구성은 `raw/<seq>/`(원본 바이트), `text/<seq>/`(추출 텍스트), `state/<seq>.json`(계열별 상태), `acquisition.jsonl`, `summary.json`, `sign-input.jsonl.gz`이다. 예약 계열은 `corpus/clean/**` 학습 후보 행으로 내보내지 않는다. 일반 수집에서 만나면 `flags: ["quarantine:benchmark_reserved"]`로 격리한다.
- 로컬 전용 확보(MODEL-RESUME-B-026)는 같은 구성을 `data/<reservation>/local/corpus/reserved/<reservation>/` 아래에 쓴다(`collect_institutions.py reserved --local …`). 여기서는 S3를 쓰지 않는다. `data/`는 git에서 제외되며 학습 경로가 아니다. 응답 바이트와 디스크 사용량에 상한이 있다(기본 각 2 GiB, `budget.json`에 누적). 차단·제한 응답을 받거나 연속 네트워크 실패가 나면 체크포인트를 남기고 멈춘다. 서명 입력은 `reserved-sign-input --local … --sign-out <새 파일>`로 만든다. 봉인된 서명 manifest(`signatures-partial-148.jsonl`)는 덮어쓰지 않고, 새 후보 manifest를 따로 만든다.
- 로컬 원문 재추출(MODEL-RESUME-B-027, `reserved-reextract --local`)은 네트워크 없이 보존된 원본 바이트만 다시 읽는다(sha256 대조). 내용 기준 형식 판정(DRM 래퍼, `.hwpx` 이름의 HWP, xlsx), 중첩 ZIP(3단계까지), 텍스트층 없는 PDF의 OCR(`kordoc --ocr`, 로컬 캐시 모델, `ocr: true` 표시)을 쓴다. 암호·배포용·DRM 보호가 걸린 HWP와 DRM 파일은 보호를 풀지 않고 `unextractable`과 사유로 남긴다. `ok`였던 기록은 바꾸지 않는다. 바뀐 기록에는 `reextract`(이전 형식·상태·오류)를 남긴다. 새 텍스트만 모은 서명 입력(`reserved-sign-input --only-reextracted`)으로 별도 후보 manifest를 만든다.
- 다른 출처(웹 등)에 같은 텍스트가 들어 있을 수 있다. 이 경우는 A가 tokens 단계에서 `scripts/check_reserved_overlap.py`(ID·URL·exact·MinHash·8-gram 포함률)로 제외한다. 입력은 B가 만드는 `sign-input.jsonl.gz`(`{family_id, text, doc_id, source_url, group}`)다.
- 상태값 `licence_status: unconfirmed`, 추출 실패, 다운로드 실패는 그대로 기록한다. 시험 적합으로 올리는 판단은 총괄이 한다.

## 4. 정제 처리 (B)

공통 순서:
1. NFC 정규화.
2. 출처별 본문 추출(HTML 태그와 뷰어 UI 제거, HWP/PDF 변환).
3. `trid.korean_quality.clean_lines`.
4. 문자 비율 검사(`FINEWEB2` 기준). 웹 출처에 적용하고, 공공 문서는 통계만 기록한다.
5. (clean-v2, 웹) 도박·성매매 광고 스팸 탈락(`spam_gambling_adult_ads`).
   - 광고 문구(예: "토토사이트 추천", "출장마사지")만 센다. 3회 이상이면서 1,000자당 2회 이상일 때 탈락시킨다.
   - '카지노', '먹튀', '조건만남', '온라인 카지노' 같은 단어만으로는 탈락시키지 않는다. 단속·피해 예방 정책 문서에도 나오기 때문이다.
6. 사용권 판정. 허용 근거가 없거나 유료·접근제한 원문은 `quarantine-*`에 사유와 함께 둔다.
7. (clean-v2) 평가 예약 기관 호스트(`reports/coordination/benchmark-reserve-*-hosts.json`)에서 온 행은 `quarantine:benchmark_reserved_host_pending_text_check`로 둔다. 예약 텍스트와 본문을 대조하고, 전 계열 확보가 끝날 때까지 해제하지 않는다.
8. `trid.korean_quality.redact`(ko-pii, 형식 검사)로 PII를 치환한다.
9. 출처 내부 완전 중복 제거(NFC, 공백 제거 문자열의 SHA-256, 먼저 나온 행 유지).
10. 분할 지정.

같은 함수를 A의 `prepare_fineweb2_ko.py`도 쓰므로 정제 결과가 일치한다. manifest에는 `korean_quality.py`의 SHA-256도 기록한다.

버전 차이:
- clean-v1(B-002 파일럿)에는 5·7단계가 없고, 유료 리포트 도메인 `report.kyobobook.co.kr`도 빠져 있다.
- clean-v1은 `redact`의 줄 끝 판정 버그가 고쳐지기 전 버전이다. 수정 전에는 '$'가 끝 줄바꿈 앞에서도 맞아 줄 단위 연락처가 남았다.
- clean-v2(B-004 파일럿)는 그 버그를 고쳤다. 그래도 다음은 남았다:
  - 구분 문자 두 개("02) 737-7378", "052- 290-3333")
  - 'ㅣ'가 붙은 번호
  - 두 번째 실행에서야 잡히는 연락처
- clean-v3에서 `redact`는 위 형식을 처리하고, 결과가 바뀌지 않을 때까지 최대 3회 반복한다.
- clean-v4(B-005) 변경:
  - 같은 `redact` 경로에 다음을 넣었다:
    - ko-pii가 놓치는 전화형 번호 보수적 치환. 050 안심번호, 해외·국제번호, 영문·'ㅣ'에 붙은 번호가 대상이다. 080 수신자 부담·060 정보이용료 ARS는 번호 계획상 사업자 번호라 남긴다.
    - 법인등록번호(문맥+검증식), EAN-13 바코드·상표 등 식별자 문맥의 13자리, URL 안 하이픈 없는 글번호(은행 문맥 없음)는 남긴다.
  - 웹 출처: 흔하지 않은 최상위 도메인(kr·com·net·org·co·io·me·tv·edu·gov 밖)은 `quarantine:uncertain_tld_mt_seo_risk`로 보류한다. 기계번역·SEO 위험 때문이며 품질 판정은 아니다.
- clean-v4 파일럿 이후(B-006) `redact` 변경. FineWeb은 재처리하지 않았고, 다음 버전에 반영한다:
  - 표의 연도·기간 열과 내선번호 나열은 카드번호로 보지 않는다.
  - 영숫자+하이픈 코드("F-B01B-T00120001-A180")는 붙은 문자로 본다.
  - 7번째 자리가 0·9인 13자리는 주민번호로 보지 않는다(주민 문맥이 있으면 예외로 치환).
  - URL 안 하이픈 없는 게시물·기사 ID는 해당 개인정보 문맥 단어가 없을 때만 남긴다.
- clean-v1~v3는 train-ready가 아니다. 버전마다 따로 보존한다.

알려진 한계:
- 도메인 차단 목록(유료 리포트 판매 사이트 등)은 알려진 사례만 격리한다. 전체 사용권 검증이 아니다.
- PII 치환은 위험 완화일 뿐 개인정보가 없다는 보장이 아니다. 탐지 수와 잔존 위험 표본 검토를 함께 기록한다. ko-pii는 080·050 국번(수신자 부담·안심번호)을 잡지 못한다.
- 기계번역 스팸에는 규칙이 없다.

## 5. manifest와 완료 마커

- **파일 단위 출처(FineWeb-2):** `corpus/state/<source>/<clean_version>/<파일>.manifest.json`에 다음을 기록한다.
  - raw 키·SHA-256, 원본 출처(`s3-raw` 또는 `hf:<revision>`), 코드 경로와 SHA-256, 작업 ID
  - 읽은 문서 수, 탈락·격리 사유별 수, PII 라벨별 수, 분할별 행 수
  - 샤드 목록(키·SHA-256·바이트·행·글자)
  - 모든 샤드의 크기와 S3 저장 SHA-256(`ChecksumSHA256`)을 다시 조회해 일치한 뒤에만 `<파일>.done`을 쓴다.
- **스트리밍 출처(국회 회의록, 기관 문서):** 샤드마다 `shard-NNNNN.json`에 키·SHA-256·포함 id·실패 id·분할·PII·처리량 지표를 기록한다. `aggregate`가 전 샤드 체크섬을 재검증하고, 수집 목록이 전부 처리됐을 때만 `_DONE`을 쓴다.
- A는 완료 마커가 있는 버전만 소비한다.

## 6. 재개와 중복 방지

- 파일 단위 작업은 `.done`이 있으면 건너뛴다. 중단된 파일은 같은 결정적 키로 처음부터 다시 쓴다.
- 스트리밍 작업은 `shard-*.json`의 id를 건너뛰고, 다음 샤드 번호는 기존 최대값 + 1로 정한다.
- 목록 탐색은 `(대수, 분류)` 등 완료 단위와 찾은 항목을 주기적으로 저장한다.
- 같은 출처를 여러 프로세스가 동시에 같은 버전으로 쓰지 않는다.

## 7. 요청 정책

- robots.txt와 게시된 이용 조건을 따른다. robots.txt가 없다고 무제한 수집을 허용한 것으로 보지 않는다.
- 같은 호스트에는 프로세스와 worker를 모두 합쳐 하나의 요청 시작 간격을 적용한다(기본 1초).
  - 429, Retry-After, 5xx, 지연 급증(최근 중앙값의 2배 초과) 시 간격을 늘리고, 정상일 때 천천히 기본값으로 돌아온다.
  - IP나 계정을 분산해 제한을 우회하지 않는다.
- 간격을 기본값보다 줄이려면 같은 호스트에서 소규모로 단계적으로 측정하고, 오류·지연 기록과 함께 총괄에 보고한다.
- 처리량 보고는 요청 수, 제한 대기, 응답 시간, 바이트, 정제, S3 시간을 분리해서 한다.

## 8. 실행 자원 (AWS-APPROVAL-001 범위)

| 항목 | 값 |
|---|---|
| SageMaker 실행 역할 | `arn:aws:iam::148761639313:role/trid-sagemaker-lora` (B가 생성, A의 `scripts/launch_sm_pretrain.py`도 사용) |
| 역할 S3 권한 | `s3:GetObject`, `s3:PutObject`, `s3:AbortMultipartUpload`는 `trid-training-148761639313/open-weight-lora/*`, `trid-training-148761639313/corpus/*`에만. `s3:ListBucket`은 같은 두 prefix에만. 그 밖의 prefix는 접근할 수 없다 |
| 기타 권한 | CloudWatch Logs(`/aws/sagemaker/TrainingJobs*`), `ecr:GetAuthorizationToken`, `pytorch-training` 저장소 이미지 받기(계정 763104351884), `cloudwatch:PutMetricData` |
| 컨테이너 | CPU `763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:2.8.0-cpu-py312-ubuntu22.04-sagemaker`, GPU `…:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker` |
| 코드 입력 | 채널 `code` = `s3://…/corpus/code/<source>/<clean_version>/`. 진입점은 `run.sh`이고, 작업 이름을 `JOB_NAME` 환경변수로 넘겨 manifest에 기록한다 |
| 출력 | 작업이 직접 `corpus/{raw,clean,state}/…`에 쓴다. SageMaker 기본 출력은 `corpus/sm-output/` |
| 종료 | 작업 완료 시 인스턴스가 자동 반납된다. `MaxRuntimeInSeconds`를 작업마다 지정한다 |

C 소유 GPU `i-01e894092261d7f9a`는 이 계약 범위 밖이다. 자격증명은 코드·로그·문서에 넣지 않는다.
