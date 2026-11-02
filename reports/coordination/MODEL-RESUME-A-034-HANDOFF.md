# MODEL-RESUME-A-034-HANDOFF (A): 배포·AWS 현재 상태와 일반 대화 병목 (짧은 인계)

작성: A, 2026-11-02 05:48 UTC.

## 1. A032 배포: 대기 아님, **이미 반영됨**
- 맥미니 운영이 **`2b8ba92`(branch `model-resume-030-followup`)**로 바뀌었다.
  - 서비스 재시작 2026-11-02 05:37:19Z, ready 05:37:30Z(약 12초)
  - `DEPLOYED_FROM_COMMIT` = `2b8ba92 (branch model-resume-030-followup)`, 이전 값은 `6a5eb37`
- **바꾼 파일:**
  - `scripts/shared_demo.py`, `scripts/share_demo_control.py`
  - 고정 계약 3개: DATA-CONTINUE-007 term-map·backfill, DATA-GAPS-008 backfill
- **바꾸지 않은 것:**
  - 버전 목록(원본 Qwen3-1.7B + reader 2개), CPU 환경변수, 터널, 보안, 색인, 모델 파일
  - 다른 서비스. 연구 GGUF는 복원하지 않았다.
- **롤백 경로:**
  - `~/Services/trid-site/rollback-6a5eb37/`(이전 두 스크립트와 DEPLOYED_FROM_COMMIT, 추가 파일 목록)
  - 기존 `rollback-9256df4/` 보존
- **배포 후 확인 (1회, 맥미니 원점에 포트 포워드로 실제 경로):**
  - healthz 200, 버전 2개
  - 기관 질문 200(quoted)
  - 규정 명시 전환 200(공무원 여비 규정)
  - 후속 "그럼 숙박비는?": 국내 승계, 별표 2 행
  - 새 대화 뒤에는 승계 없음
  - 삭제된 ID는 422
  - 공개 `https://test.tridatum.co/` 200, 옵션 2개
- **메모리:** 압력 1, 여유 55%. 백엔드 1.78 GB, llama-server 3.31 GB RSS(재시작 직후)
- **유일한 지연 원인:** 04:50–05:37Z 동안 맥미니 메모리 압력이 2로 계속 유지됐다. 이 압력은 내 변경 전부터 있었다. 파일을 먼저 준비(staging)해 두고 상태를 지켜봤다.
  - 압축 메모리의 대부분은 오래 돈 운영 TRID 프로세스와 Docker가 차지하고 있었다.
  - 재시작하면 오히려 줄 것으로 판단해 05:37에 반영했다. 실제로 재시작 뒤 압력 1로 돌아왔다(05:38–05:46 표본 모두 1).
- **남은 일:** README·SHARING을 배포 실제와 맞추는 것과 `MODEL-RESUME-A-032.md` 보고다. 바로 이어서 한다.
- **주의:** 이 배포는 검색·라우팅 수정이다. **일반 대화 품질은 바꾸지 않는다.** 모델과 prompt는 그대로다.

## 2. AWS-EVAL-006 (A031/A033) 현재 상태
- **지금 실행 중인 자원: 0.** 기동한 4대가 모두 terminated다.
  - watchdog `i-0e6945ad10818d1f2`, `i-048546d89da7a64b5`
  - smoke 1차 `i-0a924d4ca69ca3f5d`, smoke 2차 `i-008eccd96cec002ff`
  - GPU 추론은 0이다. smoke 두 번 모두 빌드 뒤 Python setup에서 실패했다.
- **장부:** 40 / 230분
- **남은 일정 2개:** 05:59:01Z, 06:18:23Z(종료된 인스턴스 대상, 실행 뒤 자동 삭제). 삭제 확인은 대기 중이다.
- **수정 파일:**
  - `scripts/aws_eval/bootstrap.sh`: `9960807`(python3-venv), `b57887e`(CPython 3.12.13 + 고정 uv 0.11.28)
  - `scripts/aws_eval/aws_eval_006.py`: `eca7e0f`(watchdog backstop, `watchdog-status`)
- **A033 미착수 (인계 가능 지점, 총괄 지시 그대로):**
  1. bootstrap에서 Python·패키지 검증을 CUDA 빌드보다 먼저 실행하도록 순서를 바꾼다.
  2. t3.micro 1대(같은 DLAMI, 15분 backstop)로 Python 3.12.13 + uv + import + 실제 토크나이저 준비를 검증한다.
  3. 그 뒤에만 3차 smoke 1회를 기동한다.
- **C 인계 파일:** `MODEL-RESUME-A-030-SMOKE.md`. 지금 유효한 run은 없다.
- 비공개 매핑, 모델명, 패키지 세부는 어디에도 적지 않았다.

## 3. 사용자가 지적한 일반 대화 병목 (사실 정리, 재실행 안 함)
- 부모가 관측한 대로, 기본 버전(원본 Qwen3-1.7B)은 일반 대화 경로에서 업무 메일 다듬기 요청의 뜻을 바꿨다.
  - "참석 가능한지 알려 달라"는 확인 요청을 "참석 가능합니다"라는 확정으로 바꿨다.
  - 질문도 새로 덧붙였다.
- **이것은 모델 능력 문제다.**
  - A024–A032의 검색·라우팅·조건 수정은 이 경로와 무관하다.
  - C-030은 프롬프트 3종과 디코딩 3종이 모두 실패했다는 근거를 남겼다. 문구 튜닝을 반복하지 않는다.
- **개선은 기반 모델 선택과 실제 서빙 검증으로만 가능하다. 근거와 제약:**
  - 맥미니 내부 검토에서 대안 3종은 모두 미합격이었다. Qwen2.5-7B는 요청 없이 중국어로 전환했고, gemma3:4b는 새 세션 기억을 날조했다(BASELINE-COMPARE-002/003).
  - 원본 대비 비교 E1(Gemma 4 E4B QAT, 비공개 80 블라인드)은 아직 실행되지 않았다. 지금 GPU smoke 단계에서 막혀 있다.
  - 맥미니는 RAM 16 GB, swap 4 GB를 이미 쓰고 있다. 4B급 이상 모델을 같이 올리기에는 메모리 여유가 작다(A-029 2절). 모델 교체에는 맥미니 메모리 실측이 함께 필요하다.
- 수집·인프라·배포를 완료했다고 해서 대화 개선으로 보고하지 않는다.

SUPERVISION_REPORT_READY MODEL-RESUME-A-034-HANDOFF
