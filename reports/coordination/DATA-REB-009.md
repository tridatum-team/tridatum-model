# DATA-REB-009: 한국부동산원 신규 내규 2건 후보 패키지(inst-rule-reb-003)와 DATA-GAPS-008 감독 결정 반영

- **범위:**
  - 2026-11-01 감독 결정을 반영했다.
  - DATA-CONTINUE-007에서 받은 한국부동산원 직원 복무규정·임직원 행동강령을 기존 기관 artifact 규격(inst-rule-reb-002와 같은 사슬)으로 묶었다.
- **하지 않은 것:**
  - 모델·학습·색인 적재·AWS 0. 웹 요청 0(DC-007 원본 재사용).
  - 보류·평가 노출 제외 해제 0. 시행연도 추정 교정 0.
- **순서:** SITE-MINI-AUDIT-010(긴급, `7d66dc4`)을 먼저 끝내고 이 작업을 이어 했다.

## 1. 결론

| 항목 | 결과 |
|---|---|
| 직원 복무규정 (nttSn 116253) | 예약 148 서명 가족과 겹침 없음(직접 일치 0, 부분 공유 0). `inst-rule-reb-003` 스냅샷 레코드 1건 |
| 임직원 행동강령 (nttSn 98560) | **부분 겹침 → 보류(`reserved_overlap`). 스냅샷에서 뺐다.** `institutions:alio-rule:20256`과 8-gram 394/5108 공유. 구간 검사 144개 중 비법령 구간 120 → `separate_from_eval_index` |
| 숫자 답 | 두 문서 모두 **자동 숫자 답으로 승격하지 않는다.** 검토 행과 structured_facts가 0이고, 별표·서식·표는 검토하지 않았다 |
| 교육비 목적 절 파생 metadata | C 판정 전에는 색인·검색 메타데이터로 승격하지 않는다(A 인계 3절). DC-007의 2·3절 근거 scope는 그대로 유지할 수 있다 |

## 2. 감독 결정 수령 (DATA-GAPS-008)

- **보류·평가 노출 제외 유지:**
  - law-byl 10개 별표는 일괄 해제하지 않는다.
  - 교육비특별회계 집행기준의 평가 노출 구간 20개는 색인에 다시 넣지 않는다.
  - B는 아무것도 바꾸지 않았다.
- **C 읽기 전용 검토 대상** (어느 고정 평가·학습·검색 범위에 어떤 제외 규칙이 걸리는지)

| # | 제외 규칙 | 위치 | 걸린 대상 |
|---|---|---|---|
| 1 | 148 서명 가족 partial_review를 보류로 처리 | `collect_statute_tables.py:226`, `extract_public_guides.py:213`, `curate_admrul_attachments.py:41` | law-byl 10개 별표, 이번 행동강령 |
| 2 | search-corpus-v2 `evaluation_exposure` 구간 제외 | `data/search-corpus-v2/dropped.jsonl` | 교육비 집행기준 20개 구간 등 |
| 3 | 852 미서명 가족 | 검사 안 함 | 이번 두 문서 포함 전부 |

- **C 확인 요청: 이미 커밋된 행동강령 짧은 인용**
  - 인용 위치:
    - DC-007 `source-ledger.jsonl` L09
    - DC-007 `term-map.jsonl` 6행(A가 sha 고정, `scripts/shared_demo.py:589`)
    - DATA-GAPS-008 `source-ledger.jsonl` G05
  - 내용: 별표3 금액 문장, "일률적으로 제공하는 교통, 숙박, 음식물 등의 금품등".
  - 마지막 문구는 청탁금지법과 같은 법령 문구일 가능성이 있다 [추정: 구간 검사의 법령 구간 24개에 포함되는지는 구간별로 대조하지 않음].
  - 이 인용들이 alio-rule:20256 가족 노출에 해당하는지는 C가 판정할 일이다. B는 고정 파일을 고치지 않았다.

## 3. A 조건부 인계

**1) inst-rule-reb-003** (`runs/search-snapshots/inst-rule-reb-003`, gitignored)

| 항목 | 값 |
|---|---|
| manifest sha | `d610e606efdbd36f40e1daff4e001d4eefdcc8fee408dd29303f75458bce74e6` |
| records sha | `6c80b214967828f00efd0fc9a30cb5a12b7db2926f8fb5caf49f45782c8573f1` |
| 레코드 | 1건(복무규정) |
| 문서 id | `reb.or.kr:1244:116253:file:021728bfe39ccd71` |
| 분리 설정 | `holds [evaluation, training]`, `index institution_rules_separate` |
| 사본 | `DATA-REB-009/inst-rule-reb-003.manifest.json` |

- **A 수집기로 확인(B 실행, 메모리 안에서만):**
  - `build_workspace_index.api_snapshot(..., separate=True)`: 문서 1, 구간 217, facts 0, kind '기관 내규(게시본)'.
  - date 2026-08-20, source_notes 4, table_check `unverified`.
  - `separate=False`는 거부됐다(공용·평가 색인에 들어가지 않음).
  - 색인 파일은 만들지 않았다.
- **조건:** 적재는 A가 분리 색인으로만 한다.
  - 표시 날짜는 "마지막 부칙<2026.8.10> 시행일, 현행 표시 안 함" 그대로 쓴다.
  - 숫자는 본문 인용(원문 일치·정답 검증 아님)으로만 쓰고, 표 행·structured_facts 경로로 승격하지 않는다.
  - 레코드 `source_notes`의 `review_scope`·`numeric_answer`·`eval_overlap`·`posting_status`를 표시하거나 따른다.

**2) 행동강령은 적재하지 않는다.**
- 보류 상태다(`data/public-data/inst-rules-reb-002/held.jsonl`).
- 해제는 C 또는 감독의 결정 사항이다.

**3) 교육비 목적 절 파생 metadata 승격 금지**
- 금지 대상: `reports/coordination/DATA-GAPS-008/applies-to-backfill.jsonl`(sha `4f45cbf3…`) **4행**, `law-go-kr:admrul-file:143116119`.
  - 근거가 평가 노출 구간("1. 목적")의 원 PDF 인용이다.
  - C 판정 전에는 색인이나 검색 메타데이터로 쓰지 않는다.
- 같은 파일 1~3행(선관위, 행안부 복식부기, 방사청 국제계약지원단)은 v5 본문 인용이라 이 제한과 관계없다.
  - 다만 이 파일을 sha로 고정하면 4행도 함께 들어온다. 4행을 걸러 쓰는 방식은 A가 정한다.
- DC-007 `applies-to-backfill.jsonl`(sha `f649524b…`)의 교육비 항목은 2·3절 근거라 지금처럼 유지할 수 있다.

## 4. 패키지 내용 (`DATA-REB-009/inst-rule-reb-003.package.json`)

| 항목 | 복무규정 | 행동강령 |
|---|---|---|
| 원문 | 게시판 nttSn 116253, 2026-10-31T07:49:33Z 수신(요청 로그 200) | nttSn 98560, 07:49:36Z |
| raw sha256 | `60b3ce78…26a73945e4`, 32쪽 | `f8eafee9…5bae58d0d766`, 44쪽 |
| 텍스트 sha | `021728bf…`, 28,775자, kordoc | `eb6693e9…`, 37,333자, kordoc |
| 시점 | 부칙<2026.8.10> "2026년 8월 20일부터 시행". 개정 이력 마지막 줄 2026.08.10과 맞음 | 부칙<2024.8.27> 시행. 개정 이력 마지막 줄과 맞음. 2026-10-31 목록에 이후 게시본 없음 |
| 적용 대상 | 직원(제2조), C0150 | 임원·직원(비정규직 포함)(제3조) |
| 검증된 조건·단위 | 제28조 휴가 종류. 제29조② 1호 연차 15일(전년도 80% 이상 근무). 4호 3년 이상 계속근무 시 매 2년 1일 가산 | 별표3 음식물 5만원(받는 쪽 상한, 출장 식비 아님). 경조사비 5만원 |
| 검토 수준 | 독립 판독 2개(pdftotext·kordoc) 조문 제목 78=78, 6·32쪽 그림(AI) | 42=42, 22쪽 그림(AI) |
| 이용허락 | 한국부동산원 저작권정책(저작권법 제24조의2 자유이용) | 같음 |
| PII | ko-pii 치환 0, PII 힌트 0 | 같음 |
| 평가 겹침 | 없음 | 부분 공유, 보류 |

- **추출본 대조:** 파이프라인 추출 텍스트의 조문 머리 목록이 DATA-GAPS-008에서 검토한 kordoc 판독과 같다(108=108, 49=49, 부칙 조문 포함). 핵심 문장도 모두 들어 있다.
- **원본 위치:** 원본 PDF는 `data/source-checks/dc007/raw`에서 표준 위치 `data/public-data/raw/reb.or.kr-1244/<sha16>.pdf`로 복사했다(sha 동일, 덮어쓰기 없음).

## 5. 실행 기록

```
uv run -q --no-project --python 3.12 --with 'ko-pii==1.22.15' --with kiwipiepy python scripts/extract_public_guides.py \
    --version inst-rules-reb-002-src --state-files files-state-slice-reb009.jsonl
.venv/bin/python scripts/check_reserved_overlap.py check --manifest data/benchmark-reserve-001/signatures-partial-148.jsonl \
    --corpus data/public-data/inst-rules-reb-002-src/overlap-input.jsonl --report data/public-data/inst-rules-reb-002-src/overlap-148.json
.venv/bin/python scripts/verify_overlap_spans.py --overlap-report data/public-data/inst-rules-reb-002-src/overlap-148.json \
    --corpus data/public-data/inst-rules-reb-002-src/overlap-input.jsonl --out data/public-data/inst-rules-reb-002-src/overlap-span-check.jsonl
uv run ... scripts/extract_public_guides.py --version inst-rules-reb-002 --resplit inst-rules-reb-002-src   # consumer check ok
.venv/bin/python scripts/build_institution_rule_snapshot.py --guides inst-rules-reb-002 \
    --doc reb.or.kr:1244:116253:file:60b3ce78755ba7db --snapshot-id inst-rule-reb-003 \
    --out runs/search-snapshots/inst-rule-reb-003 --producer B/DATA-REB-009
```

- **명령 기록 메모:** 구간 검사의 처음 출력 경로는 `reports/coordination/DATA-REB-009/`였다. 겹치는 텍스트 발췌가 들어 있어 gitignored 위치로 옮겼다. 위 명령은 옮긴 뒤의 경로로 적었다.
- **산출물 sha:**
  - 커밋한 것: 요약 `reb-overlap-span-check.summary.json`.
  - gitignored:
    - 구간 검사 원본 `overlap-span-check.jsonl`(`57438eb5…`)
    - src manifest `89152ff5…`
    - clean manifest `2be51579…`
    - overlap report `d3723dff…`
    - state slice `15cf3c10…`
- **코드 변경:** 0. 기존 스크립트만 사용했다.
- **테스트:** 커밋 전에 전체 단위 테스트를 게이트로 실행했다.

## 6. 미해결

- **행동강령 보류 해제:** 행동강령의 alio-rule:20256 부분 겹침과 이미 커밋된 짧은 인용의 노출 판정은 C 또는 감독의 결정 사항이다.
- **852 미서명 가족:** 검사하지 않았다.
- **복무규정 미검토 부분:** 별표·서식·표 3행과 나머지 조문은 원문과 대조하지 않았다.
- **현행 여부:** 두 문서 모두 게시 뒤 개정 여부를 확인하지 않았다.
- **초과근무 hwpx 연도 불일치(DC-007):** 이번 범위가 아니며 그대로 남아 있다.

SUPERVISION_REPORT_READY DATA-REB-009
