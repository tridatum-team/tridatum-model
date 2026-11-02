# MODEL-RESUME-B-027: 보존 원문으로 partial 23계열 추출 결손 해소(오프라인)

- **범위:** B-026이 보존한 원본 바이트만 다시 읽었다.
- **하지 않은 것:**
  - 추가 다운로드·네트워크 요청 0. GPU·AWS·맥미니 부하 0.
  - 예약 원문을 학습 입력으로 복사 0. 사용권 승인 0. training hold 해제 0.
  - 봉인 148 manifest와 B-026 후보 manifest 변경 0(sha 재확인).
  - A 서빙·AWS 코드 수정 0. 수정한 코드는 수집기(`collect_institutions.py`)와 그 테스트뿐이다.
  - 다른 역할의 작업 중 파일(`RETRIEVAL-ROOT-008.md`, `shared_demo.py` 등)은 건드리지 않았다.
- **C 요청:** C-028에 B-026 독립 검토를 요청해 두었다. C 판정 전에는 hold를 유지한다.

## 1. 결론

| 항목 | B-026 | B-027 |
|---|---|---|
| 계열 상태 | text_acquired 829, partial 23 | **text_acquired 843, partial 9** |
| 추출되지 않은 파일·내부 파일(ZIP 집계 제외, 개별 문서 단위) | 62건(xlsx 16, 중첩 ZIP 14, 배포용 HWP 13, 빈 텍스트 PDF 13, `ole:hwpx` 3, DRM 3) + 부분·실패 ZIP 집계 12 | **16건, 모두 보호 파일**: 배포용 HWP 13, DRM PDF 3. 사유를 기록했다. ZIP 집계는 partial 1 |
| 새로 추출한 텍스트 | — | 203문서, 3,366,684자(중첩 ZIP 안 171문서 포함) |
| 새 후보 manifest | — | `data/benchmark-reserve-001/signatures-candidate-local-reextract-b027.jsonl` 203행·17계열, sha `c3310d7f…` |
| v2 33건 대조(새 manifest) | — | **일치 0, 부분 검토 0** (`data/work-case-candidates-v2-reserve-check-b027/overlap-reextract.json`, sha `0ca19d56…`) |
| 봉인·기존 manifest | 148 `0ed3cc07…`, 852 `584da062…` | 둘 다 그대로 |

- **결손 감소:** 계열 23 → 9. 해소된 결손은 형식 판별 오류와 추출기 부재 46건(xlsx 16, 중첩 ZIP 14, 텍스트층 없는 PDF 13, `.hwpx` 이름의 HWP 3)이다. ZIP 집계 12건 중 11건이 ok가 됐다.
- **남은 결손:** 남은 16건은 보호를 풀어야만 읽을 수 있어 해소하지 않았다(4절).

## 2. 진단 (현존 상태·로그·원본 바이트 기준, 알려진 실패를 다시 실행하지 않음)

| B-026 분류 | 건수 | 바이트·도구 진단 | 실제 형식·원인 |
|---|---|---|---|
| ZIP 안 `zip:xlsx` | 16 | 이름 확장자 xlsx, PK | xlsx인데 추출기가 없었음 |
| ZIP 안 `zip:zip` | 14(+상위 ZIP 2 `failed`) | PK, ZIP 안의 ZIP | 중첩 ZIP을 열지 않았음 |
| 부분 ZIP | 10 | 위 내부 결손의 집계 | 내부 결손이 원인 |
| `hwp` rhwp 실패 | 11 + ZIP 안 2 | HWP 5 FileHeader 속성 `0x5` = compressed + **distribution(배포용)** | 본문(ViewText)이 배포용 잠금으로 암호화돼 있음 |
| `ole:hwpx` | 3 | OLE 서명 + `HWP Document File` | `.hwpx` 이름으로 올라온 HWP 5(보호 플래그 없음) |
| `unknown:pdf` | 3 | 앞부분 `\x9b DRMONE  This Document is encrypted` | DRM 래퍼 문서(PDF 아님) |
| 빈 텍스트 PDF | 10 + ZIP 안 3 | pdfinfo 암호화 no, pdffonts 0~1, pdfimages 0~4,108 | 스캔·이미지 PDF 또는 글자를 윤곽선으로 그린 PDF(텍스트층 없음) |

## 3. 보완 (기존 수집 파이프 재사용, 형식 판별과 지원 형식만)

**`scripts/collect_institutions.py`**
- **`kind_of`(내용 기준):**
  - DRM 래퍼는 `drm:drmone`.
  - OLE 안에 HWP 서명이 있으면 이름과 상관없이 `hwp`.
  - PK + `.xlsx`는 `xlsx`.
- **`extract`:**
  - HWP는 FileHeader 플래그를 먼저 읽는다. 암호·배포용·DRM·인증서 암호화면 `Unextractable('hwp_<flag>_protected')`로 기록하고, 보호는 풀지 않는다.
  - DRM은 `Unextractable('drm_drmone_encrypted')`로 기록한다.
  - xlsx는 표준 라이브러리로 시트 XML을 읽는다(공유 문자열 포함, 행마다 탭 구분).
- **`text_record`:**
  - 텍스트층 없는 PDF는 `kordoc 4.24.23 --ocr`로 OCR한다. B-015와 같은 도구이고, 내장 PP-OCRv5 korean 모델은 `~/.cache/kordoc`의 로컬 캐시를 쓴다.
  - OCR 결과는 `ocr: true`와 추출기명으로 표시해, 원본 텍스트층 추출과 구분된다.
  - 실패 상태는 셋으로 나눠 기록한다: `unextractable`(사유), `failed`(오류), `empty_text`.
- **`zip_members`:** 중첩 ZIP을 3단계까지 연다(member stem `<no>-m001-m002`).
- **`reserved-reextract --local`:**
  - 오프라인으로 실행하며 HTTP client를 만들지 않는다.
  - 원본 sha256을 대조한 뒤 `ok`가 아닌 기록만 다시 추출한다. `ok` 기록은 바꾸지 않는다.
  - 바뀐 기록에는 `reextract`(by, 시각, 이전 kind·status·error)를 남긴다.
  - 결과 상태 파일은 `benchmark-reserve-001-local-b027-*`이다. B-026 상태 파일은 덮어쓰지 않는다.
- **`reserved-sign-input --only-reextracted`:** 재추출로 새로 생긴 텍스트(중첩 내부 포함)만 서명 입력으로 만든다.

**테스트(`tests/test_collect_institutions.py`) 4개 추가**
- 내용 기준 형식 판별.
- 배포용 HWP와 DRM이 사유와 함께 `unextractable`로 기록되고 텍스트가 없는지.
- xlsx 행 추출과 중첩 ZIP.
- 재추출이 저장 바이트만 쓰고, `ok` 기록을 그대로 두며, 서명 입력에는 변경분만 담는지.

**실제 원문 재추출:** 1회 실행, 276초.

## 4. 해소 결과와 남은 결손

**결손별 결과**

| 결손 | 결과 |
|---|---|
| xlsx 16 | 모두 ok |
| 중첩 ZIP 16(내부 14 + 상위 2) | 모두 ok. 그 안의 171문서에 텍스트가 생김 |
| 부분 ZIP 10 | 9개 ok. 1개(24653/239904)는 안에 배포용 HWP 2개가 있어 partial로 남음 |
| `.hwpx` 이름의 HWP 3 | 모두 ok(rhwp) |
| 텍스트층 없는 PDF 13 | 모두 OCR로 ok |

- **OCR 품질:** OCR 결과 13건의 한글 비율은 0.34~0.84이다. 문서마다 '제N조' 표지가 17~182개 잡혔다.
- **OCR 검토 범위:** 쪽 그림과 글자 단위로 대조하지는 않았다. B-015에서 같은 도구가 원문자 항 번호를 숫자로 읽은 사례가 있다. 그래서 이 텍스트는 겹침 검사용 서명 입력으로만 쓰고, 정확한 원문으로 쓰지 않는다.

**남은 결손 16건, 9계열**

| 원인 | 건수 | 계열(seq) | 이유 | 다음 조치 |
|---|---|---|---|---|
| 배포용(읽기 전용) HWP | 13 (파일 11 + ZIP 안 2) | 16498, 24653, 24670, 28238, 28297, 31539, 37698, 51931 | 본문 스트림이 배포용 잠금으로 암호화돼 있다. 설치된 `rhwp convert`로 잠금을 풀 수는 있다. 그러나 기관이 건 복제 제한(기술적 보호조치)을 해제하는 일이라 지원 형식 추출 범위 밖으로 보고 하지 않았다 | 잠금 해제 허용 여부(권리·정책)를 총괄이 결정해야 한다. 허용되면 같은 저장 바이트로 오프라인 처리가 가능하다 |
| DRM 래퍼 PDF | 3 | 22201 | `DRMONE` 암호화. 열 수 있는 추출기가 없다 | 원기관 비DRM 원문이 있는지 확인(새 수집이 필요해 이번 범위 밖) |

- **서명 영향:** 이 9계열의 서명은 해당 파일 없이 만들어져 불완전하다. 다른 파일·개정본의 텍스트는 서명돼 있다.

## 5. 용량 (총 로컬 2 GiB 승인 한도 유지)

| 항목 | 바이트 |
|---|---|
| 로컬 저장소(원본·텍스트·상태) | 1,599,115,384 (+7,834,975) |
| B-026 서명 입력 + 후보 manifest | 45,096,159 + 440,933,682 |
| B-027 서명 입력 + 후보 manifest | 1,915,352 + 15,525,400 |
| **합계** | **2,102,585,977 / 2,147,483,648** (남은 44.9 MB) |

- **여유 계산:** 실행 전에 남은 여유(약 70 MB)를 계산했고, 저장소 상한을 `--max-disk-bytes 1631280409`로 낮춰 실행했다.
- **임시 파일:** OCR·추출 임시 파일은 임시 폴더에서 바로 지워졌다. 서명용 평문 임시 파일도 지웠다.

## 6. 격리·보존

- **예약 원문 위치:** `data/benchmark-reserve-001/local/`(git 제외)에만 있다.
- **새 서명 파일:** `sign-input-local-reextract-b027.jsonl.gz`(sha `4f53f0d4…`)와 `signatures-candidate-local-reextract-b027.jsonl`은 새 파일이다(출력이 없는지 확인한 뒤 작성).
- **커밋한 상태 파일:** 집계 `benchmark-reserve-001-local-b027-acquisition-summary.json`만 커밋했다. 계열별 jsonl(7.3MB, 파일명·시행 문구 포함)은 B-026과 같이 커밋하지 않는다.
- **manifest 사용법:**
  - 학습 후보 대조에는 봉인 148, 후보 852(B-026), 후보 재추출분(B-027) 세 manifest를 함께 쓴다.
  - 두 후보 manifest는 감독 승인 전까지 후보 지위다.

## 7. 재현

```
.venv/bin/python scripts/collect_institutions.py reserved-reextract --local data/benchmark-reserve-001/local \
    --max-disk-bytes 1631280409 --report benchmark-reserve-001-local-b027 --by MODEL-RESUME-B-027
.venv/bin/python scripts/collect_institutions.py reserved-sign-input --local data/benchmark-reserve-001/local \
    --only-reextracted --sign-out data/benchmark-reserve-001/sign-input-local-reextract-b027.jsonl.gz
# 평문으로 풀어서(임시) sign
.venv/bin/python scripts/check_reserved_overlap.py sign --input <plain> \
    --output data/benchmark-reserve-001/signatures-candidate-local-reextract-b027.jsonl
.venv/bin/python scripts/check_reserved_overlap.py check \
    --manifest data/benchmark-reserve-001/signatures-candidate-local-reextract-b027.jsonl \
    --corpus data/work-case-candidates-v2/overlap-input.jsonl \
    --report data/work-case-candidates-v2-reserve-check-b027/overlap-reextract.json
```

- **다시 실행할 때:** 같은 재추출을 다시 실행하면 남은 16건만 다시 읽는다. 결과는 같은 사유로 남는다.
- **집계 기준 차이:** 요약의 `text_chars`(94,936,926)는 파일과 1단계 ZIP 내부만 센다. 중첩 내부 텍스트는 위 3,366,684자에 들어 있다.

**산출물**
- 코드: `scripts/collect_institutions.py`.
- 테스트: `tests/test_collect_institutions.py`.
- 문서: `docs/CORPUS_CONTRACT.md` 3절.
- 보고서: `reports/coordination/benchmark-reserve-001-local-b027-acquisition-summary.json`, 이 보고서.
- 전체 단위 테스트는 커밋 전 게이트로 1회 실행했다.

SUPERVISION_REPORT_READY MODEL-RESUME-B-027
