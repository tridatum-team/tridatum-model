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
- **메모리:** 05:46Z 압력 1, 여유 55%, backend 약 1.78GB / llama-server 약 3.31GB RSS. **05:53Z 메타 확인에서는 압력 2로 돌아왔다.** 공개 페이지는 응답했으며 지속 안정으로 주장하지 않는다.
- **유일한 지연 원인:** 04:50–05:37Z 동안 맥미니 메모리 압력이 2로 계속 유지됐다. 이 압력은 내 변경 전부터 있었다. 파일을 먼저 준비(staging)해 두고 상태를 지켜봤다.
  - 압축 메모리의 대부분은 오래 돈 운영 TRID 프로세스와 Docker가 차지하고 있었다.
  - 재시작하면 오히려 줄 것으로 판단해 05:37에 반영했다. 실제로 재시작 뒤 압력 1로 돌아왔다(05:38–05:46 표본 모두 1).
- **문서 완료:** README·SHARING 배포 상태와 `MODEL-RESUME-A-032.md`를 정리했다. 추가 배포·재시작은 하지 않았다.
- **주의:** 이 배포는 검색·라우팅 수정이다. **일반 대화 품질은 바꾸지 않는다.** 모델과 prompt는 그대로다.

## 2. AWS-EVAL-006 (A031/A033) 현재 상태
- **지금 실행 중인 자원: 0.** 기동한 4대가 모두 terminated다.
  - watchdog `i-0e6945ad10818d1f2`, `i-048546d89da7a64b5`
  - smoke 1차 `i-0a924d4ca69ca3f5d`, smoke 2차 `i-008eccd96cec002ff`
  - GPU 추론은 0이다. smoke 두 번 모두 빌드 뒤 Python setup에서 실패했다.
- **장부:** 40 / 230분
- **종료 일정:** 1차 일정(05:59:01Z)은 06:00Z 이후 GetSchedule에서 ResourceNotFoundException 확인. 2차 일정(06:18:23Z)은 아직 예정 시각 전이며 자동 삭제 확인 미완료다.
- **수정 파일:**
  - `scripts/aws_eval/bootstrap.sh`: `9960807`(python3-venv), `b57887e`(CPython 3.12.13 + 고정 uv 0.11.28)
  - `scripts/aws_eval/aws_eval_006.py`: `eca7e0f`(watchdog backstop, `watchdog-status`)
- **A033 중지 / 인계 가능:** 최신 진단 우선 지시에 따라 fail-fast 순서 변경, CPU 환경 검증, 3차 GPU smoke 모두 미착수·보류다. 자동 후속 기동은 하지 않는다.
- **C 인계 파일:** `MODEL-RESUME-A-030-SMOKE.md`. 지금 유효한 run은 없다.
- 비공개 매핑, 모델명, 패키지 세부는 어디에도 적지 않았다.

## 3. 사용자가 지적한 일반 대화 병목 (사실 정리, 재실행 안 함)
- 부모가 관측한 대로, 기본 버전(원본 Qwen3-1.7B)은 일반 대화 경로에서 업무 메일 다듬기 요청의 뜻을 바꿨다.
  - "참석 가능한지 알려 달라"는 확인 요청을 "참석 가능합니다"라는 확정으로 바꿨다.
  - 질문도 새로 덧붙였다.
- **정정: 원인을 모델 능력·크기·가중치로 단정할 근거는 아직 없다.** 이전 판의 “모델 능력 문제”, “기반 모델 선택으로만 가능” 문구를 철회한다.
  - A024–A032의 검색·라우팅·조건 수정은 이 일반 대화 의미 보존 실패의 해결 증거가 아니다.
  - 프롬프트 3종 × 디코딩 3종의 직접 근거는 `GENERAL-CHAT-003.md`다. `C-030.md`는 검색 회귀 보고이므로 이전 인용은 잘못됐다.
  - 기존 진단에서는 일부 실패가 모든 조건에 남았고 일부는 설정·표본에 따라 달라졌다. 이를 이번 메일 요청의 단일 원인 증명으로 확대하지 않는다.
  - 운영 요청 구성과 canonical Qwen 계약의 정적 비교를 `CHAT-ROOT-A-001.md`에 기록한다. 새 추론·프롬프트 수정·모델 교체·학습은 하지 않는다.
- 수집·인프라·배포를 완료했다고 해서 대화 개선으로 보고하지 않는다.

## 4. 안전 상태 보충 (2026-11-02 05:52 UTC)
- operator 프로필 `DescribeInstances`로 smoke 2대의 `terminated`를 다시 확인했다. watchdog 2대는 이번 응답에서 빠졌으며, 종료 근거는 기존 A031 실측 기록이다. 새 CPU/GPU 기동 0, GPU 추론 0, smoke 종료 2, GPU 장부 40/230분 유지.
- 05:52Z에는 두 일정 모두 ENABLED, `ActionAfterCompletion=DELETE`였다. 06:00Z 이후 1차 일정은 ResourceNotFoundException을 확인했다. 2차는 06:18:23Z 이후 확인할 잔여 항목이며 삭제 완료로 보고하지 않는다.
- A032는 현재 응답 중인 배포를 유지한다. 압력 2 재발은 별도 보고에 명시했다. 재시작·추가 배포하지 않는다. SHARING·README·별도 배포 보고 정리 완료.

SUPERVISION_REPORT_READY MODEL-RESUME-A-034-HANDOFF
