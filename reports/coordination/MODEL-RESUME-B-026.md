# MODEL-RESUME-B-026: 예약 852계열 원문 로컬 확보 재개, 새 후보 서명, v2 후보 대조

- **근거:** 사용자 명시 승인. ALIO 직접 수집 재개, 로컬 저장 최대 2GB(이번 수집은 기존 200MB 한도 대신 2GB).
- **하지 않은 것:**
  - GPU·유료 AWS 자원 생성 0, S3 호출 0, 운영 미니 부하 0.
  - 우회 IP·계정·프록시 0, 무한 재시도 0.
  - 질문·정답 열람·출력 0. 기존 148 원문 재다운로드 0. 봉인 manifest 변경 0. training hold 해제 0.
- **B-025 확인:** C-025는 B-025의 고친 4건을 모두 pass로 판정했고, 나머지 29건은 v1과 바이트까지 같다고 확인했다(`MODEL-RESUME-C-025.md` 29행, 5절). 33건 모두 training hold다.

## 1. 결론

| 항목 | 실측 |
|---|---|
| 확보 | **852/852계열**. `text_acquired` 829, `partial` 23(일부 파일 추출 실패, 3절) |
| 요청 | 4,826건(robots.txt 1 + 상세·파일). HTTP 200 4,826. 403·429·5xx·네트워크 실패 0. 중단 조건 발동 0 |
| 받은 바이트 | **1,366,009,287 B(약 1.27 GiB)**. 한도 2 GiB의 63.6% |
| 디스크 | 로컬 저장소 1,591,280,409 B(한도 2 GiB) |
| 시간 | 2026-11-01 16:37:18Z → 18:08:38Z, 91분. 동시 요청 1 |
| 요청 간격 | 앞 요청 끝 → 다음 요청 시작 최소 1.00004 s(요청 로그 4,826줄로 확인) |
| 파일 | 3,973개(HWP 2,321, PDF 1,417, HWPX 128, ZIP 101, 기타 6). 다운로드 실패 0. 추출 ok 3,934 |
| ZIP 내부 | 1,015개, 추출 ok 980 |
| 새 서명 | `data/benchmark-reserve-001/signatures-candidate-local-852-b026.jsonl` 4,825행·852계열, sha `584da062…`. 봉인 148 manifest(`0ed3cc07…`)는 그대로이고, 겹치는 계열은 0 |
| v2 후보 대조 | 33행 중 852 manifest **일치 0, 부분 검토 0**. 148 manifest 결과(0/0)와 합치면, 서명된 1,000계열에 대해 일치·부분 검토가 0이다 |
| training hold | 유지. C 검토 전에는 해제하지 않는다 |

**B-003 추정과 실측 비교**

| 항목 | 추정 | 실측 |
|---|---|---|
| 요청 | 약 4,700 | 4,826 |
| 바이트 | 0.54~1.44 GB | 1.37 GB |
| 시간 | 약 80분 | 91분 |

## 2. 로컬 전용 모드 구현

**`scripts/collect_institutions.py`**
- B-003 파이프라인(상세 JSON → 모든 개정본·첨부 → 내용 기반 형식 판정 → 추출 → 계열 상태)을 그대로 쓴다.
- 저장 계층만 `S3Store`·`LocalStore`로 나눴다. 기존 S3 경로의 동작과 키는 그대로다.

**`--local DIR`**

| 항목 | 동작 |
|---|---|
| AWS | boto3를 import하지 않고 S3 client도 만들지 않는다(테스트에서 하위 프로세스의 `sys.modules`로 확인) |
| 요청 | `--workers 1`, `--interval >= 1.0`이 아니면 실행을 거부한다. 첫 요청은 robots.txt이고, 상세·파일 경로가 허용될 때만 진행한다 |
| 중단 | 401·403·429·451 또는 접근 제한 페이지면 즉시 멈춘다. 연속 네트워크 실패 3회에서 멈춘다. 어느 경우든 체크포인트를 쓰고 멈춘다. 재시도 없음(`Client(retry=False)`) |
| 전송 한도 | 본문을 스트리밍으로 읽어 남은 한도(64 KiB 여유 포함)에서 끊는다(`BudgetExceeded`). 누계는 `budget.json`에 남아 다음 실행까지 이어진다 |
| 디스크 한도 | 쓰기 전에 검사한다. 시작 전 여유 공간이 (남은 한도 + 1 GiB)보다 작으면 실행을 거부한다 |
| 경로 안전 | 고정 문자 집합 키만 받는다. `..`·절대경로·저장소 밖 경로는 거부한다. 확장자는 정제한다 |
| 원자적 쓰기 | 같은 폴더의 임시 파일 + fsync 뒤 `os.link`로 쓴다. 원본은 덮어쓰지 않고, 같은 바이트면 그대로, 다른 바이트면 `<stem>.<sha16>.<ext>`로 나란히 둔다. 상태·요약은 `os.replace`로 쓴다. 중단된 계열은 상태가 없어 다음 실행에서 다시 받는다 |
| 재개·중복 | 이미 확보된 148계열(`benchmark-reserve-001-acquisition.jsonl`)과 로컬 상태가 있는 계열은 건너뛴다 |
| 출처 기록 | URL·sha256·HTTP·바이트·형식·추출기·시행 문구·사용권 근거는 기존 상태 형식과 같다 |
| 상태 파일 | `reports/coordination/benchmark-reserve-001-local-acquisition{.jsonl,-summary.json}`. 기존 148 파일은 덮어쓰지 않는다 |

**`scripts/collect_assembly_minutes.py` `Client`**
- `retry=False`(1회 시도, 모든 응답을 반환, 네트워크 오류면 `NetworkError`)와 `max_bytes`(스트리밍 상한)를 추가했다.
- 기본값은 기존 동작 그대로다.

**검증(실제 실행)**
- `tests/test_collect_institutions.py`에 7개를 더했다.
  - 키 탈출 거부, 원본 비교체·임시 파일 없음, 디스크 한도 사전 거부.
  - 403에서 체크포인트 후 정지(이미 확보한 계열은 요청하지 않고, 중단 계열은 대기로 남음), 연속 네트워크 실패 정지.
  - 바이트 한도의 실행 간 누적, 실제 `Client`의 본문 절단·429 비재시도·네트워크 오류 1회.
  - 로컬 모드의 AWS SDK 미import.
- 오프라인 CLI 경계: `--interval 0.5`, `--workers 2`, `--local`과 다른 action, `--sign-out` 누락, 여유 공간 부족은 모두 네트워크 요청 전에 거부됐다.
- 전체 단위 테스트는 커밋 전 게이트로 실행했다.

## 3. 커버리지와 남은 결손

- **partial 23계열:** 업무군별로 공기업 운영·서비스 6, 보훈·복지 4, 보건·안전 4, 고용·인사 3, 주거·부동산 2, 계약·조달 2, 교육·연구 1, 회계·예산 1.
- **추출 실패 원인**

| 원인 | 건수 |
|---|---|
| ZIP 안 xlsx 16·zip 14 등 추출기 없는 내부 파일 | 내부 실패 32 |
| rhwp가 처리하지 못한 HWP | 11 |
| 추출기 없는 형식(`unknown:pdf` 3, `ole:hwpx` 3) | 6 |
| 빈 텍스트 | 10 |
| 부분 추출 ZIP | 10 |

- 원본 바이트는 모두 보존돼 있다. 추출기를 고치면 `--retry-status partial`로 다시 처리할 수 있다.
- **서명 범위:** 서명은 추출에 성공한 텍스트로 만들었다(4,825행). 위 실패 파일은 서명에 들어가지 않았다. 해당 23계열은 서명이 불완전하다.
- **시점 차이:** 이번 원문은 2026-11-01 기준 ALIO 목록이다. 예약 메타데이터(10-27~29 카탈로그)와 개정본 구성이 다를 수 있다. 계열 단위로는 같은 상세 페이지의 전체 파일을 받았다.
- **사용권:** 852계열 모두 `unconfirmed`다(ALIO 정책 근거만 있음). 공공누리 표시가 있는 파일은 1개다.

## 4. 격리와 보존

- **저장 위치:** 예약 원문은 `data/benchmark-reserve-001/local/corpus/reserved/benchmark-reserve-001/`(raw·text·state)에만 있다. `data/`는 git 제외이며 학습 경로가 아니다.
- **서명 파일:**
  - 서명 입력 `sign-input-local-852-b026.jsonl.gz`(sha `c832c9a0…`, 4,825행).
  - 서명 결과 `signatures-candidate-local-852-b026.jsonl`.
  - 둘 다 새 파일이다. `sign` 실행 전에 출력이 없는지 확인했다.
  - 서명 단계에서 만든 평문 임시 파일(/tmp)은 지웠다.
- **보고서 파일:**
  - 커밋: `benchmark-reserve-001-local-acquisition-summary.json`(집계만).
  - 커밋하지 않음: 계열별 상태 `benchmark-reserve-001-local-acquisition.jsonl`(7.1MB). 파일명과 원문 시행 문구 줄이 들어 있어서다. B-003의 148 파일은 같은 형식으로 이미 커밋돼 있다.
- **요청 로그:** `logs/requests-www.alio.go.kr.jsonl`(git 제외)에 4,826줄이 있다.
- **봉인 manifest:** `signatures-partial-148.jsonl` sha `0ed3cc07…`는 실행 뒤에도 같다.

## 5. 인계

| 대상 | 내용 |
|---|---|
| A | 학습 후보 대조에 두 manifest를 함께 쓴다: 봉인 148 + 후보 852(`signatures-candidate-local-852-b026.jsonl`, sha `584da062…`). 후보 manifest는 감독 승인 전 "후보" 지위다 |
| C | v2 33건의 852 대조 결과는 `data/work-case-candidates-v2-reserve-check-b026/overlap-852.json`에 있다(일치 0, 부분 0). training hold 해제 여부는 C가 판단한다. 23계열 불완전 서명과 시점 차이(3절)를 판단에 넣어야 한다 |
| 감독 | 852 확보로 "852 미서명" 블로커는 서명 범위가 생겼다. 남은 것은 23계열 추출 결손, 사용권 unconfirmed, 업무군 검토(pending)다. 예약 메타데이터의 `pending_checks`는 바꾸지 않았다 |

## 6. 재현

```
.venv/bin/python scripts/collect_institutions.py reserved --local data/benchmark-reserve-001/local --interval 1.0
.venv/bin/python scripts/collect_institutions.py reserved-sign-input --local data/benchmark-reserve-001/local \
    --sign-out data/benchmark-reserve-001/sign-input-local-852-b026.jsonl.gz
# 평문으로 풀어서(임시) sign
.venv/bin/python scripts/check_reserved_overlap.py sign --input <plain jsonl> \
    --output data/benchmark-reserve-001/signatures-candidate-local-852-b026.jsonl
.venv/bin/python scripts/check_reserved_overlap.py check --manifest data/benchmark-reserve-001/signatures-candidate-local-852-b026.jsonl \
    --corpus data/work-case-candidates-v2/overlap-input.jsonl --report data/work-case-candidates-v2-reserve-check-b026/overlap-852.json
```

- 같은 명령을 다시 실행하면 852계열이 모두 상태가 있으므로 robots.txt 1건 외에는 요청하지 않는다.
- 요약의 `families_pending: 148`은 로컬 상태가 없는 계열 수다. 이 148계열은 B-003에서 이미 확보됐다(`families_acquired_elsewhere: 148`).
- 요약 지표 `s3_s`는 이름만 남은 것이다. 로컬 모드에서는 로컬 저장 시간이다.

**산출물**
- 코드: `scripts/collect_institutions.py`, `scripts/collect_assembly_minutes.py`.
- 테스트: `tests/test_collect_institutions.py`.
- 문서: `docs/CORPUS_CONTRACT.md` 3절.
- 보고서: `reports/coordination/benchmark-reserve-001-local-acquisition-summary.json`, 이 보고서.

SUPERVISION_REPORT_READY MODEL-RESUME-B-026
