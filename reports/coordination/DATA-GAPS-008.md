# DATA-GAPS-008: DATA-CONTINUE-007 결손 보강과 RETRIEVAL-ROOT-008용 적용 조건·범위 근거

- **범위:** DATA-CONTINUE-007과 A의 RETRIEVAL-ROOT-008에서 나온 공백만 메웠다. 새 전국 수집 0. 웹 요청 0(기존 원본만 사용). ALIO 접속 0.
- **하지 않은 것:**
  - 색인·학습셋·비공개 평가·gold 수정 0. 서비스 코드 변경 0.
  - 보류(held)나 평가 노출 제외를 푼 것 0.
  - AWS·GPU·학습 0.
- **산출물:** `reports/coordination/DATA-GAPS-008/`
  - `source-ledger.jsonl`: 5건, 근거 인용 17개
  - `rule-conditions.jsonl`: 3건
  - `applies-to-backfill.jsonl`: 4건
  - `summary.json`, `source-checks-dg008.manifest.json`
  - 모든 인용은 `scripts/check_topic_sources.py`가 원문에서 위치(char span)를 찾았다. 하나라도 못 찾으면 빌드가 멈춘다.

## 1. 결론

| 항목 | 결과 | 상태 |
|---|---|---|
| DC-007 계약 파일 추적 여부 | `57810aa`부터 모두 git 추적. A가 고정한 sha와 일치 | 재작업 없음 |
| 별표 4 국외 여비 지급표가 v5 문서로 없는 원인 | 원인 추적 완료: 수집 단계의 보수적 보류(`reserved_overlap`). 21행은 이미 A의 TABLE-LINK 입력에 있음 | 문서 단위 보류 해제는 감독 결정 필요 |
| 교육비특별회계 "1. 목적" | 원 PDF에서 복원. **DC-007의 "추출 누락"은 틀렸다.** 실제 원인은 search-corpus-v2의 평가 노출 구간 제외 | 복원 텍스트는 산출물로만 둠 |
| 한국부동산원 복무규정·행동강령 | 독립 판독기(kordoc) 대조 일치, 핵심 쪽 그림 확인 | 검증됨, 색인 미적재 |
| 별표 8 vs 별표 2 적용 조건 | 제16조①·제29조②·제18조① 근거로 3건 | A 전달 |
| 선관위·행안부 복식부기·방사청 국제계약지원단 적용 범위 | 공식 조항으로 3건, 교육비 목적 근거 1건 추가 | A 전달 |

## 2. DATA-CONTINUE-007 파일 추적 확인

- A의 ROOT008 메모("untracked")는 시점이 지난 내용이다. 모든 파일이 `57810aa`(2026-10-31 17:00)부터 추적 중이고, ROOT008(`d4b0291`, 11-01 00:43)보다 앞선다.
- A의 `scripts/shared_demo.py:589-591` 고정값이 커밋된 파일과 같다.
  - `term-map.jsonl` `531b654b373f5208273fcba2fcdaeebfca091c93c3631d700e12dd0875303897`
  - `applies-to-backfill.jsonl` `f649524b3acb243b41b993b2b16250c5aee120c38786fa66f89b5e0c07288de4`
- `form-passages.jsonl`은 A가 sha로 고정하지 않는다. 커밋된 sha는 `1ceb8eaaced561944bc1143b1134f47ac623d55b64e2770a15f585ba0390137d`.
- 이번 스크립트 변경 후 DC-007 입력으로 다시 실행했다(`/tmp` 출력). 기존 출력 6개의 sha가 모두 같았다. 새 파일 `rule-conditions.jsonl`만 추가된다.

## 3. 공무원 여비 규정 [별표 4] (`law-go-kr:law-byl:169708653`)

**원인 사슬** (`source-ledger.jsonl` G01, 인용 6개 위치 확인)

1. `data/law-byl/v2/overlap-148.json`: 이 별표는 예약 평가 가족과의 `matches`에는 없다. `partial_review`에만 올라 있다.
   - 공유 8-gram: `institutions:alio-rule:54431` 105/3664, `15386` 124/3329, `36465` 207/3319.
   - 보고서 계약 문구: "not counted as matches or excluded automatically".
2. `scripts/collect_statute_tables.py:226` `status()`가 `partial_review`도 겹침 집합에 넣는다. 그래서 `held: reserved_overlap`이 됐다. 보고서 계약보다 보수적인 구현이다.
3. `scripts/build_search_corpus.py:122`는 `candidate_*` 상태만 싣는다. 그래서 이 별표는 `dropped.jsonl` 기록 없이 search-corpus-v2와 v5에서 빠졌다.
4. 값 자체는 문제없다. `values_check: agree`(숫자 110개), 2,643자.

**같은 원인으로 보류된 별표: 10개**

- 여비 2개: 공무원 여비 규정 별표 4, 법원공무원 여비 규칙 별표 4(`155902189`, G02).
- 국가계약법 시행규칙 별표 2·3·4.
- 국유재산법 시행규칙 서식 4개. 이 중 `160401251`은 `values_differ`도 함께 걸려 있다.
- 공무원수당 등에 관한 규정 별표 6.

**복구 산출물**

- **행 단위:** 이미 A의 입력에 있다.
  - `data/law-byl/v2-structure-003-rows-001/queryable-rows.jsonl`에 21행이 있고, 모두 `ai_visual_review_match`다. B-030 렌더 대조는 21/21 pass다.
  - `reports/coordination/B-030/byl-index-links.jsonl`이 이 21행을 본문 `law-go-kr:law:290313`의 제16조 구간에 연결한다(`byl_doc_in_index: null`).
  - 그래서 별표 4 숫자 답변은 문서 보류를 풀지 않아도 TABLE-LINK로 가능하다.
- **문서 단위:** `data/source-checks/dg008/candidates/law-byl-169708653.candidate.json`(gitignored, sha는 manifest).
  - v2 추출 원문 그대로이며, 텍스트 sha `d6bf9310…`는 `kordoc_text_sha256`과 같다.
  - 보류 상태와 적재 조건("감독 결정 전 적재 금지")을 함께 적었다.
- **결정 필요(감독 또는 C):** partial_review 수준 겹침(전체 8-gram의 3~6%)을 평가 독립성 위반으로 볼지 판단해야 한다. 보지 않는다면 수집 단계의 보수적 처리를 고쳐 10개 별표를 함께 다시 판정한다. B는 판단하지 않았다.

## 4. 교육비특별회계 세출예산 집행기준 "1. 목적" (`law-go-kr:admrul-file:143116119`)

**정정:** DC-007 보고서 75·97행, A 통지 69행, contract-input 120행의 "추출 누락"은 틀렸다. 고정된 DC-007 파일은 덮어쓰지 않고, 이 보고서로 정정한다.

- 추출본 `data/law-attachments/v6/candidate_quality_ok.jsonl`에는 "### 1. 목적" 절이 있다.
- search-corpus-v2에서 `#1219`(1219~1518자) 구간이 `evaluation_exposure`(eval_window_hits 2)로 빠졌다. 이 문서는 모두 20개 구간이 같은 사유로 빠졌다.

**원문 복원** (G03)

- 원본: `data/law-attachments/raw/2100000245194/143116119.pdf`, sha256 `3800a9711109d533233c21239c75cf165da518e9e747645c7be1ba585d91aeda`.
- 위치: PDF 9쪽(인쇄면 5쪽).
- 확인: pdftotext 텍스트와 페이지 그림(`render/edu-p-009.png`)으로 확인했다.
- 복원 텍스트:

  > 1. 목적
  > ○ 본 기준은 「지방회계법 시행령」 제64조를 근거로, 세출예산 집행에 관한 기본원칙과 세부적인 기준을 정함으로써 건전하고 투명한 재정운용과 예산집행의 효율성을 제고하고자 한다.
  > ○ 재정운용의 자율성은 최대한 보장하되 공통된 기준을 제시하여 예산집행의 형평성을 도모한다.
  > ○ 교육부장관은 이 예규에 대하여 「훈령ㆍ예규 등의 발령 및 관리에 관한 규정」에 따라 2024년 8월 1일 기준으로 매3년이 되는 시점(매 3년째의 7월 31일까지를 말한다.)마다 그 타당성을 검토하여 개선 등의 조치를 하여야 한다.

- **색인 재적재 안 함:** 평가 창과 겹치는 구간이기 때문이다. 적용 범위 근거로만 `applies-to-backfill.jsonl`에 인용했다. 인용 출처는 원 PDF 텍스트 파일이다.

## 5. 한국부동산원 직원 복무규정·임직원 행동강령: 독립 판독 대조 (G04, G05)

- **판독기:** pdftotext(DC-007)와 kordoc 4.24.23(npx, 이번 실행). 출력은 `data/source-checks/dg008/text/`.

| 문서 | 조문 제목 집합 | 핵심 문장(두 판독기 모두 존재) | 그림 확인 쪽 |
|---|---|---|---|
| 복무규정 (bbsId 1244, nttSn 116253) | 78 = 78, 차이 0 | 제28조 휴가 종류; 제29조② "전년도중 80% 이상 근무한 자 : 15일"; 3년 이상 매 2년 1일 가산; 부칙 "2026년 8월 20일부터 시행" | 6쪽(제27~29조), 32쪽(부칙<2026.8.10>) |
| 행동강령 (nttSn 98560) | 42 = 42, 차이 0 | 별표3 음식물 5만원; 경조사비 5만원; 부칙 "2024년 8월 27일부터 시행" | 22쪽(별표3: 음식물 5만원, 경조사비 5만원·화환 10만원, 선물 5만원·농수산물 15만원(30만원)) |

- **숫자 다중집합:** 두 판독기의 숫자 묶음은 다르다. kordoc이 서식 표의 칸을 따로 끊기 때문이다. 핵심 금액·일수 문장은 같으므로 결함으로 세지 않았다.
- **검토 범위:** 그림 확인은 B(AI)가 위 쪽만 한 것이다. 전문 대조는 하지 않았다.

## 6. A 전달: RETRIEVAL-ROOT-008 남은 한계용

### 6.1 별표 8 vs 별표 2 (`rule-conditions.jsonl`)

| id | 적용 조건 | 근거 표 | 근거 조항(원문 위치 확인) |
|---|---|---|---|
| R01 | 일반 국내 출장의 일비·숙박비·식비 | 별표 2 (`169708641`) | 제16조① "국내 여행자의 일비(日費)ㆍ숙박비 및 식비는 별표 2에 따라 지급하고" |
| R02 | 상시 출장이 필요한 공무원만. 지급 대상·지급액은 소속 장관이 정함. 월정 여비 방식 | 별표 8 (`169708689`) | 제29조②. 별표 8 제목 "(제29조제2항 관련)", "출장일수가 월 15일 이상일 때에는 전액을 지급하고" |
| R03 | 근무지 내 국내 출장 | 제18조 정액(별표 아님) | 제18조① "4시간 이상인 공무원에게는 2만원" |

- 일반 "출장비 지급 기준" 질문의 근거 표는 R01이다. 별표 8은 질문에 상시 출장·특정 직무가 있을 때만 맞는다.
- 반영 방법(조건 매칭, 순위 조정 등)은 A가 정한다.

### 6.2 적용 범위 (`applies-to-backfill.jsonl`, DC-007과 같은 스키마·enum)

| 문서 | 현재 v5 `applies_to` | 제안 | 근거 |
|---|---|---|---|
| 선관위 공무국외출장 규정 `admrul:1106031000` | `scope: unknown` | `institution`: 선관위 소속 공무원 + 선관위 예산으로 국외출장 가는 사람 | 제1조, 제2조②. 제12조는 비용을 「공무원여비규정」에 맡김(금액 기준 문서 아님) |
| 행안부 지방자치단체 복식부기·재무회계 운영규정 `admrul-file:122610539` | 문서 종류 설명 문자열 | `local_government`: 지방자치단체 재무회계 실무 | 규정 의의 "가.", 지방회계기준 적용대상. 여비 지급 기준 아님 |
| 방위사업청 국제계약지원단 운영규정 `admrul:2100000284496` | null | `institution`: 국제계약지원단·국제계약지원관 업무를 조정·감독하는 방위사업청 관계 부서 | 제2조(적용범위), 제1조 |
| 교육비특별회계 세출예산 집행기준 `admrul-file:143116119` | (DC-007 보강과 같음) | `local_government` 유지, 목적 절 근거 추가 | 원 PDF 9쪽 목적 절 |

- **반영 시 주의:** A의 `scripts/shared_demo.py`는 DC-007 파일을 sha로 고정한다. 이 파일들은 별도 경로의 새 파일이므로, 쓰려면 A가 경로와 sha를 추가해야 한다. 파일 sha는 7절에 있다.

## 7. 미해결

- **별표 4 문서 단위 보류 해제(10개 별표 공통):** 감독 또는 C의 평가 독립성 결정이 필요하다. 행 단위 답변은 지금도 가능하다.
- **교육비 목적 절 색인 재적재:** 평가 노출 판단이 필요하다. 같은 문서의 나머지 19개 제외 구간도 같다.
- **한국부동산원 초과근무 hwpx의 연도 불일치(DC-007):** 그대로 남아 있다. 날짜를 추측해 바로잡지 않았다.
- **현재성:** 모든 문서의 현재성은 수집 시점 기준이다. 어느 문서도 현행이라고 표시하지 않았다.

## 8. 실행 증거

- **스크립트 변경** `scripts/check_topic_sources.py`
  - 새 contract 키 `rule_conditions`와 출력 `rule-conditions.jsonl`.
  - `--skip-diagnostics`: 색인 전체 진단 파일을 다시 만들지 않는 옵션.
  - DC-007 재실행 출력은 기존과 같다(2절).
- **실행 명령:** `python3 scripts/check_topic_sources.py --index runs/workspace-index/v5 --inst-index runs/workspace-index/inst-reb-002 --ledger reports/coordination/DATA-GAPS-008/ledger-input.json --contract reports/coordination/DATA-GAPS-008/contract-input.json --out reports/coordination/DATA-GAPS-008 --skip-diagnostics`
  - 입력: code sha `257654d7…`, v5 manifest `1c9540ca…`, inst manifest `f2239545…`.
  - 출력 sha:
    - `source-ledger.jsonl` `b9aa72e74bde832b37bd712120e3de6b141df833424bc3f64d0fdb99e938710b`
    - `applies-to-backfill.jsonl` `4f45cbf3c63d279f626ee769ecbdf63b8f7c4009c1b96b1913d8407378ce5827`
    - `rule-conditions.jsonl` `294e8bc79f137197623e6fe9a228810a4a9248d163d9f6b233b8d594a872cd95`
    - `term-map.jsonl`은 빈 파일이다(이번 입력에 용어 대응 없음).
- **로컬 검증 파일 8개** (`data/source-checks/dg008/`, gitignored): kordoc 출력 2, pdftotext 쪽 1, 페이지 그림 4, 별표 4 후보 1. sha는 `source-checks-dg008.manifest.json`에 있다.
- **다운로드 예산:** 이번 작업 0바이트. 누적 약 1.7MB.
- **테스트:** 커밋 전에 전체 단위 테스트를 게이트로 실행했다.

SUPERVISION_REPORT_READY DATA-GAPS-008
