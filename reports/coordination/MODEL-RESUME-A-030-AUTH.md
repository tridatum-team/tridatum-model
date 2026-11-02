# MODEL-RESUME-A-030-AUTH (A): watchdog-test·smoke 실행 차단 — operator 세션 잔여 90분 미만

작성: A, 2026-11-02 04:20 UTC (13:20 KST).

**판정: 차단.** 지시 조건은 "세션 유효, 남은 시간 최소 90분"이다. operator 로그인 세션의 남은 시간이 **최대 88.6분**이라 조건을 만족하지 못한다. 그래서 `watchdog-test`와 `launch --mode smoke`를 시작하지 않았다.
- 새 로그인은 지시대로 임의로 하지 않았다.
- AWS 자원 생성·시작은 0이다.

**관측 (`reports/coordination/MODEL-RESUME-A-028/check_identity.py`, 비밀값 출력 없음)**

| 항목 | 값 |
|---|---|
| 확인 시각 | 2026-11-02T04:20:25Z |
| 자격 증명 환경변수 | 없음(profile 우회 없음) |
| operator STS | `arn:aws:iam::148761639313:user/trid-aws-eval-006-operator` (유효) |
| 현재 임시 자격 증명 | 04:35:26Z 만료, 15분 주기 자동 갱신 |
| operator 로그인 시작 (login cache idToken `iat`) | 2026-11-01T17:48:59Z |
| 세션 상한 (AWS 문서: IAM 주체 세션 기간, 최대 12시간) | **최대 2026-11-02T05:48:59Z** |
| **남은 시간** | **최대 88.6분** (실제 세션 기간 설정이 12시간보다 짧으면 더 적다) |

**왜 90분이 필요한가**
- watchdog-test 판정에 약 20분이 걸린다.
- 이어서 smoke는 예약 70분에 C 전달·수집·종료 확인까지 들어간다.
- 실행 중 세션이 끝나면 operator로 `status`와 `terminate` 확인을 할 수 없다. 인스턴스 종료 자체는 OS 타이머와 Scheduler가 자격 증명과 무관하게 수행한다.

**해제 조건 (사용자 조치)**
- 사용자가 operator 프로필로 새로 로그인한다: `aws login --profile trid-aws-eval-006-operator`. 새 12시간 세션이 생긴다.
- 로그인되면 A는 같은 확인을 다시 한다. 90분 이상이면 계약 순서대로 진행한다.
  - `preflight`
  - `watchdog-test`, 판정은 `watchdog-status <stamp>`를 +4, +7, +16분에
  - C-029 READY 확인
  - `launch --mode smoke` 1회

**기다리는 동안의 작업:** 지시대로 C-026 한계 중 "국내/국외 조건의 후속 질문 승계"를 `model-resume-024`에서 갈라 낸 별도 후속 브랜치에서 개선한다. 운영 배포는 하지 않는다. 결과는 `MODEL-RESUME-A-030.md`에 적는다.

**참고 (사용하지 않음):** login cache에는 root 세션과 `trid-b-corpus`(B 수집 계정) 세션도 있다. 지시대로 유료 실행에는 쓰지 않는다. 이번 확인에서도 신원만 읽고 호출하지 않았다.

SUPERVISION_REPORT_READY MODEL-RESUME-A-030-AUTH
