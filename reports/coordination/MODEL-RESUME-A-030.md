# MODEL-RESUME-A-030 (A): AWS watchdog·smoke는 신원 여유 부족으로 차단, 대기 중 후속 조건 승계 개선

작성: A, 2026-11-02 (04:20–04:35 UTC).

## 상태 요약 (실제와 미실행 분리)

| 항목 | 상태 |
|---|---|
| operator 세션 확인 | **실행.** 남은 시간 최대 88.6분이다. 90분 기준에 못 미쳐 **차단**했다(`MODEL-RESUME-A-030-AUTH.md`, `0c84018`). 새 로그인은 하지 않았다 |
| `watchdog-test` (t3.micro 2대) | **미실행** (차단) |
| C-029 READY 확인 | **확인함.** `MODEL-RESUME-C-029-READY.md`(`f3be11b`): READY. c-runner 비root 신원은 04:20:45Z에 확인됐고, 공개 2턴 smoke 세트와 키 쌍이 준비됐다 |
| `launch --mode smoke` (g6.2xlarge) | **미실행.** watchdog 통과가 선행 조건이고 그것도 차단됐다 |
| `MODEL-RESUME-A-030-SMOKE.md` (C 인계) | **미작성.** run_id와 instance_id가 아직 없다 |
| GPU 장부 / 비용 | 0분 / 0원. AWS 자원 생성 0 |
| 대기 중 개발: 국내·국외 조건의 후속 승계 | **실행·검증 완료.** 별도 브랜치 `model-resume-030-followup`(`ff59b93` 기준). 운영 배포 없음 |
| 동결 후보 `ff59b93`, 운영 `6a5eb37`, 맥미니 | 변경 0 |

## 1. AWS 차단과 재개 조건
- **관측 (04:20:25Z):**
  - operator STS는 정상이다(`user/trid-aws-eval-006-operator`).
  - 로그인 시작은 2026-11-01 17:48:59Z다. 세션 최대 12시간이라 끝은 최대 05:48:59Z다.
- **판정:** watchdog 판정에 약 20분, smoke 예약에 70분이 든다. 세션 잔여가 90분 미만이라 시작하면 중간에 operator 확인과 정리를 할 수 없게 된다. 그래서 시작하지 않았다.
- **재개 조건:** 사용자가 `aws login --profile trid-aws-eval-006-operator`로 다시 로그인한다.
  - A는 새 로그인을 감시 중이다. 로컬 login cache의 발급 시각만 읽고, 비밀값은 읽지 않는다. 감시는 6시간이다.
  - 새 로그인이 확인되면 A-028 계약 순서대로 진행한다.
    1. `preflight`
    2. `watchdog-test`, 판정은 `watchdog-status`를 +4·+7·+16분에
    3. 통과하면 `launch --mode smoke` 1회
    4. `MODEL-RESUME-A-030-SMOKE.md` 인계
- **c-runner 세션:** 최대 13:42:11Z다. C의 smoke 공급도 그 전에 끝나야 한다.
- **쓰지 않은 신원:** root와 B 수집 계정(`trid-b-corpus`) 세션은 login cache에 있지만 쓰지 않았다.

## 2. 대기 중 개발: 후속 질문의 국내·국외 조건 승계 (C-026 4절 한계 1)

**브랜치**
- `model-resume-030-followup` (worktree `../trid-model-030`): `ff59b93` 위에 `a74f9cd`(코드·테스트), `2b8ba92`(문서)
- `ff59b93`은 C 최종 검증 중이라 그대로 두었다.
- 모델·prompt 변경은 0이다.

**변경 (`scripts/shared_demo.py`, 기존 별표 조건 판정을 재사용)**
- **승계:** 검색 답의 세션 기록에 그 답이 말했거나 이어받은 조건(`conditions`)을 남긴다. 다음 턴이 같은 문서를 잇는 진짜 후속(`continued`)이면 그 조건을 "말한 것"으로 본다(`inherited` → `carried`).
  - 예: "…국내 출장 식비는?" → "그럼 숙박비는?"은 별표 2(국내)로 답한다.
  - 승계한 조건은 `conditions_carried` 필드와 `scope` 문장("앞 질문에서 말한 조건(국내)을 … 적용")에 드러난다.
- **해제:** 다음 경우에는 승계하지 않는다.
  - 질문이 이 규정 표의 다른 경우를 직접 말할 때. 지금 말한 쪽이 우선한다.
  - 기관이나 규정을 명시해 전환할 때(후속 아님)
  - 문서를 고정(pin)했을 때
  - 새 대화를 시작할 때. 기록이 지워진다.
- **판정 경계:** 표 하나의 조건 단위를 모두 말해야 그 경우를 말한 것으로 본다. 예: "출장"만으로는 별표 8(상시·출장)이 아니다. 낱말 하나 때문에 승계가 풀리지 않는다.

**검증 (각 1회)**
- **단위 테스트:** 관련 suite **138 OK**.
  - 새 경계 테스트 `test_follow_up_carries_the_stated_case_until_it_states_another_or_leaves`가 확인하는 것:
    - 승계, 후속의 후속 유지
    - 다른 경우 우선
    - 이전 문서 없음이면 해제
    - 공유 낱말 하나로는 해제하지 않음
  - 이 테스트는 `ff59b93`에 `inherited` 인자가 없어서 그 코드로는 성립하지 않는다.
- **실제 CPU 경로** (`followup-2b8ba92-cpu.json`, 실행기 `run_followup.py`)
  - 조건: MacBook 격리 인스턴스 1개, `LLAMA_ARG_DEVICE=none`·`N_GPU_LAYERS=0`. 맥미니 0. 모두 HTTP 200, 포트 해제, 압박 전후 1.

| 턴 | 결과 |
|---|---|
| F1 공무원 여비 규정 국내 출장 식비 | 별표 2 행 |
| F2 그럼 숙박비는? | 이어 찾음, **국내 승계**, 별표 2 행, scope에 승계 문구 |
| F3 그럼 일비는? | 이어 찾음, **국내 계속 승계**, 별표 2 행 |
| F4 그럼 국외는? | 이어 찾지 못함 → not_found(irrelevant). 승계 없음, 틀린 표 없음. **아래 한계** |
| F5 한국부동산원 여비규정에서 숙박비 | 명시 전환, 승계 없음, 부동산원 답 |
| F6 그럼 일비는? | 부동산원 이어 찾음, 승계 없음(부동산원 답에 표 조건 없음) |
| F7·F8 새 대화 → 그럼 숙박비는? | 승계 없음, 되묻기 |
| F9 새 세션 "공무원 여비 규정에서는 숙박비가…" | A-025와 같은 경우 안내(별표 2·4 제외) |

- **다른 경우 우선의 추가 확인** (모델 없이 v5 선택만, `conflicting-followup-2b8ba92.txt`, 직전 문서 = 공무원 여비 규정, 이어받을 조건 = 국내)
  - "그럼 국외 출장 숙박비는?", "국외 출장이면 숙박비는?" → 이어 찾음, 조건 국외, 별표 4 행, 승계 없음
  - "그럼 숙박비는?" → 국내 승계, 별표 2 행
  - "그럼 국외는?" → not_found(irrelevant)

**한계 (정직 보고)**
- **"그럼 국외는?"처럼 주제어 없는 아주 짧은 후속:** 기존 후속 판정(직전 문서 관련도 ≥ CLOSE)과 근거 적합성 검사를 넘지 못해 "찾지 못함"으로 간다. 승계 코드와는 별개인 기존 동작이다. 틀린 표를 붙이지 않는 쪽이다.
- **조건 동의어**("해외" ↔ 국외)와 **암시 조건**("서울로 출장" → 국내)은 범위 밖이라 그대로다.
- **운영 반영:** C 검증 전이며 배포하지 않았다. `ff59b93` 판정과 섞이지 않게 별도 브랜치로 두었다.

## 3. 파일
- `reports/coordination/MODEL-RESUME-A-030-AUTH.md` (차단 보고)
- `reports/coordination/MODEL-RESUME-A-030/`
  - `run_followup.py`
  - `followup-2b8ba92-cpu.json`, `followup-2b8ba92-cpu.server.txt`
  - `check_conflicting_followup.py`, `conflicting-followup-2b8ba92.txt`
- 후속 브랜치 `model-resume-030-followup` `2b8ba92`: `scripts/shared_demo.py`, `tests/test_shared_demo.py`, `docs/SHARING.md`

SUPERVISION_REPORT_READY MODEL-RESUME-A-030
