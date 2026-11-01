# MODEL-RESUME-B-025: C-024 독립 검토 반영(4건 원문 재확인·v2 후보)과 852 미서명 가족 메타데이터 집계

- **입력:**
  - C의 공개 보고 `MODEL-RESUME-C-024.md`, `MODEL-RESUME-C-024/b024-semantic-review.jsonl`.
  - 검토 결과 33건 중 pass 29, partial 1, minor 3.
- **원칙:** C의 지적 문구를 그대로 옮기지 않았다. 해당 조문을 원문(v5 색인 본문)에서 다시 읽고 고쳤다.
- **하지 않은 것:**
  - 비공개 문항·정답 열람 0. 새 승인 없는 보류 해제 0. 신규 다운로드 0. ALIO 요청 0.
  - AWS·운영 미니 부하 0. 봉인된 예약 manifest와 v1 후보 덮어쓰기 0.
- **상태:** 모든 후보는 training hold 그대로다. 학습 승인이 아니다.

## 1. 결론

| 항목 | 결과 |
|---|---|
| 새 후보 | `data/work-case-candidates-v2/` 33건, 보류 0 (candidates sha `cf7dc105…`, manifest `f3c40090…`) |
| 변경 | 4건만 바뀌었다(content sha 비교). 29건은 v1과 같다 |
| v1 | 그대로다(candidates sha `b7ce8b05…`, 재계산 확인) |
| 148 서명 가족 겹침 | 일치 0, 부분 검토 0 (33행) |
| A 학습 로더 | 33행 로드, 계약 위반 0, 인코딩 33/33(Qwen3-1.7B 토크나이저). 고친 4건의 답 토큰은 236 / 256 / 253 / 248(한도 256) |
| 852 미서명 가족 | 원인·해결 경로·필요 원천·권리·다운로드량을 메타데이터만으로 집계했다(4절). 이미 가진 원문으로 서명할 수 있는 가족은 **0**이다. 그래서 서명 작업은 실행하지 않았고, 계획만 세웠다 |

## 2. 4건 수정 (원문 재확인 결과)

| 사례 | C 지적 | B가 원문에서 확인한 것 | 고친 내용 |
|---|---|---|---|
| `naqs-daily-travel-65km-official-car` | 최종 금액 없음 | 별표 "출장거리 50㎞ 이상～80km미만 15,000원", 제3조② "공용차량…일액여비의 2분의 1", 별표 비고 "출발부터 귀청까지…전체 출장거리" | 결론을 **7,500원**으로 명시하고 계산식 `15,000원 × 2분의 1`을 붙였다. 관할 시·군 안 품질관리업무 출장이라는 적용 조건(제3조③ 반대)을 조건에 추가 |
| `teacher-leave-bonus-days` | 질문 사실을 적용한 결론 없음 | 제5조⑤ 본문: 연도 중 결근·휴직(공무상 휴직 제외)·정직·강등·직위해제 없음, 각각 1일(총 2일 이내). 1호: 병가 1일 미만(공무상 병가만 쓴 경우 포함). 2호: 연가실시 3일 미만 | 결론을 "제외 사유가 없으면 1호·2호 각 1일, 총 2일 가산"으로 고쳤다. 병가 미사용 → 1호, 연가 2일 → 2호를 적용했다. 남은 확인 사실(작년 결근·휴직 등)은 따로 두었다 |
| `surplus-sale-private-contract` | 4호와 그 단서 누락 | 제22조③ 단서 "제4호의 경우는…매각내용을 공고하여 다수가 참여할 수 있도록 한다", 4호, 그리고 C가 지적하지 않은 ①단서 "자산처분시스템을 이용하는 것이 곤란할 경우에는 다른 방법으로 매각할 수 있다" | 1~4호를 모두 나열하고 4호의 공고 의무를 넣었다. ①단서도 추가했다. 4호 세부 요건은 국가계약법 시행령에 있어 근거에 없다고 명시 |
| `event-service-weekend-work-cost` | 주말 = 휴일 전제 | 지침에 '휴일' 정의 조문이 없다. 다른 용례는 제15조④ "공휴일 및 토요일은 제외한다"뿐인데, 이는 기간 계산에서 공휴일과 토요일을 따로 적은 것이다 | "주말 작업이 '휴일 또는 야간작업'에 해당하는 경우"를 전제로 바꾸고, 해당 여부는 근거로 판단할 수 없다고 적었다. behavior를 `answer_with_condition`으로, missing을 "주말 작업일이 휴일에 해당하는지"로 바꿈 |

- **기록:** 각 행의 `candidate.revision`에 v1 content sha, C 검토 파일, B의 원문 확인 내용을 남겼다.

## 3. 빌더 변경과 검증

**`scripts/build_work_case_candidates.py`**
- `structure.calculations`를 추가했다. 산술식에 근거 숫자만 쓰고 결과가 맞을 때에만, 원문에 없는 계산 결과(7,500)를 목표 문장에 쓸 수 있다.
- 산술식이 아니거나, 피연산자 근거가 없거나, 결과가 다르면 보류한다.
- `candidate.revision`을 그대로 옮긴다.
- docstring을 고쳤다.

**검증(실제 실행)**

| 검증 | 결과 |
|---|---|
| 정상 빌드 | 33/33 통과 |
| 일부러 틀린 계산(결과 7000) | `calculation_result_mismatch`·`ungrounded_number:7500`으로 보류 |
| 계산 필드 없이 7,500원을 쓴 경우 | `ungrounded_number:7500`으로 보류 |
| 첫 v2 건조 실행 | 답 토큰 초과 2건(260, 267)이 잡혔다. 문장을 줄여 236, 253으로 맞춤 |
| A 로더(`train_open_weight_lora.load_rows(adapter=case)` + `case_contract_violation` + `encode`) | 33/33 |
| 전체 단위 테스트 | 커밋 전 게이트로 실행 |

- **음성 검사 위치:** 틀린 계산·계산 누락 검사는 `/tmp` 임시 스펙으로 돌렸다. 영구 테스트로 남기지 않았다.

## 4. 852 미서명 가족 (메타데이터만, `MODEL-RESUME-B-025/unsigned-852-metadata.json`)

**읽은 것과 읽지 않은 것**
- 읽은 것: 공개 예약 파일(`benchmark-reserve-001.jsonl`), 그 manifest·확보 요약·상태, 서명 manifest(가족 id만), B-003.
- 예약 원문 텍스트·평가 문항·정답은 열지도 출력하지도 않았다.

**왜 미서명인가**
1. 서명은 추출 원문으로만 만든다. `check_reserved_overlap.py sign`이 NFC·공백 제거 원문의 sha256, 단어 5-shingle MinHash 64개, 단어 8-gram 집합을 만든다.
   - 예약 파일은 메타데이터라 서명 입력으로 쓰면 안 된다(manifest 한계 문구).
2. 원문은 148가족만 확보됐다.
   - B-003 확보(ALIO, `collect_institutions.py reserved`)가 2026-10-30 04:15 KST부터 ALIO 연결 실패로 멈췄고, 04:25 이후 재개하지 않았다.
   - 852가족은 모두 `text_status: catalogued`, `text_path: null`이다.
3. 이후 지시로 B의 ALIO 접근이 막혀 있다. 확보 스크립트는 S3(`corpus/reserved/…`)에 쓰는데, 이번 라운드는 AWS를 쓰지 않는다.

**이미 확보한 원문으로 해결 가능한 부분: 0가족**
- 로컬 서명 입력(867행)은 서명된 148가족과 정확히 같다.
- 852가족을 기관명 + 정규화 제목으로 로컬 원천 전체에 맞춰 봤다(v5 색인 meta, `runs/search-snapshots/*`, `data/public-data` guides-v11·inst-rules-reb-*). 제목 일치는 0이다.
- 로컬 문서가 있는 기관 가운데 미서명 가족을 가진 기관은 10곳, 가족은 26개다. 해당 가족 원문 자체는 없다.
  - 기관: 주택도시보증공사, 한국부동산원, 국가철도공단, 한국동서발전, 한국국제협력단, 한국산업기술기획평가원, 한국환경공단, 한국남부발전, 한국토지주택공사, 수도권매립지관리공사.
  - **이 기관들의 자체 규정을 검색·학습에 쓸 때 미검사 위험이 가장 크다.** 예: 한국부동산원은 미서명 4가족, LH와 한국남부발전은 각각 미서명 '여비규정'이 있다.

**B-025 후보와의 관계**
- 후보의 발행 주체 8곳(부처·청과 법령)은 예약 기관 354곳에 없다.
- 미서명 852가족 제목 중 후보 분야 키워드: 계약·용역·입찰 97, 여비·출장 32, 휴가·복무 24, 예산·성과금 19, 물품·불용·자산 13, 징계 2.
- 이는 제목 근접일 뿐이다. 본문 겹침은 852 원문 없이는 배제할 수 없다.

**852 구성**

| 기준 | 분포 |
|---|---|
| 업무군 | 8개 업무군 각 106~107 |
| 기관 유형 | 기타공공기관 620, 준정부기관(위탁집행형) 126, 공기업(준시장형) 42, 공기업(시장형) 35, 준정부기관(기금관리형) 29 |
| 규정 분류 | 기타 579, 인사·복무·징계 168, 보수 52, 직제 38, 정관 15 |
| 기관 수 | 354 |

**해결에 필요한 것 (계획만, 실행 안 함)**

| 항목 | 내용 |
|---|---|
| 원천 | ALIO 상세·규정 파일(robots Allow: /, B-003) |
| 권리 | ALIO 저작권정책(공공데이터법 자유이용, 저작권법 제24조의2). 파일별 공공누리 미확인은 148가족과 같다 |
| 양 [추정] | 148가족 기준 가족당 파일 4.86개, 파일 바이트 평균 1.69MB·중앙값 0.63MB·p90 3.73MB, B-003 기준 가족당 약 5.5요청 → 약 4,700요청, **약 0.54~1.44GB**, 1 req/s로 약 80분 |
| 예산 | 이 작업의 200MB 한도를 넘는다 |
| 필요한 승인 | ① B의 ALIO 접근(먼저 단일 요청으로 B-003 연결 실패 상태 재확인) ② 저장 위치: S3(AWS 승인) 또는 로컬 전용 모드(코드 변경) ③ 200MB를 넘는 다운로드 예산 |
| 산출 계획 | 새로 추출한 원문은 **새 후보 manifest**(예: `data/benchmark-reserve-001/signatures-candidate-<N>.jsonl`)로 서명한다. 봉인된 `signatures-partial-148.jsonl`과 예약 파일은 덮어쓰지 않는다. 모든 학습 후보를 두 manifest에 대해 다시 검사한다 |

## 5. C-024 판정 반영

| 대상 | 판정 | B 조치 |
|---|---|---|
| DG-008 applies-to-backfill 4행 | metadata 범위로만 허용, 원 구간 재적재 금지 | 그대로 두었다. 색인·구간에 아무것도 넣지 않음 |
| DC-007 term-map 6행 | 미소비 상태 유지 허용 | 그대로 |
| REB 행동강령, law-byl 보류 10개 | 보류 유지 | 그대로 |
| v2 후보 | training hold | 852 미서명 미검사를 이유로 유지 |

## 6. 인계

**A**
- `data/work-case-candidates-v2/`(git 제외), manifest 사본 `MODEL-RESUME-B-025/work-case-candidates-v2.manifest.json`.
- `split: candidate`, hold 그대로다. v1을 쓰던 곳이 있다면 v2로 바꿀지는 A가 정한다.
- 재현 명령:

  ```
  build_work_case_candidates.py --spec reports/coordination/MODEL-RESUME-B-025/case-spec.json \
      --screen reports/coordination/MODEL-RESUME-B-024/screen-and-protect.json \
      --tokenizer runs/open-weight-base/qwen3-1.7b-hf-rev70d244cc --out data/work-case-candidates-v2
  ```

  spec sha `c81182b2…`, 빌더 code sha `b4abddfc…`.

**감독**
- 852를 서명하려면 4절의 승인 세 가지가 필요하다. 승인 전에는 B가 ALIO에 요청하지 않는다.

**산출물**
- 코드: `scripts/build_work_case_candidates.py`.
- 문서: `docs/RESEARCH_LOG.md`.
- `reports/coordination/MODEL-RESUME-B-025/`: `case-spec.json`, `unsigned-852-metadata.json`, `work-case-candidates-v2.manifest.json`.

SUPERVISION_REPORT_READY MODEL-RESUME-B-025
