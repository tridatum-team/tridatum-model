# MODEL-RESUME-A-030-SMOKE (A → C): AWS-EVAL-006 smoke 실행 메타 인계

작성: A, 2026-11-02 04:49 UTC. 비밀이 아닌 실행 메타만 적는다. 모델명, 매핑, 패키지 세부는 적지 않는다.

| 항목 | 값 |
|---|---|
| run_id | `aws-eval-006-smoke-20261102T044859Z` |
| instance_id | `i-0a924d4ca69ca3f5d` (g6.2xlarge, ap-northeast-2a) |
| 기동 | 2026-11-02T04:49:01Z |
| OS 타이머 | 60분 (`shutdown -h +60`, 약 05:49Z) |
| Scheduler 강제 종료 | **2026-11-02T05:59:01Z** (일정 `trid-aws-eval-006-i-0a924d4ca69ca3f5d`, 실행 뒤 자동 삭제) |
| 코드 묶음 sha256 | `8b45bd9178b7a4bc1ca66851758b13dfbb731b69de220635d20eeb5f2f79548d` |
| llama.cpp | `8ca4060b7e30e76baaa5670ee1ec1b760b7f8552` (CUDA sm_89) |
| GPU 장부 | 예약 70분 / 230분 |
| 선행 확인 | watchdog-test 통과(04:32–04:49Z): OS 타이머 종료, Scheduler 종료, 일정 삭제 모두 실측 |

**C 실행 순서 (계약 그대로)**
1. `phase.json`이 `awaiting-input`이 되고 `runs/<run>/input-cert.pem`이 게시되면(cert-ready) 전달한다.
   - 빌드와 자산 준비가 먼저라 기동 뒤 수십 분 걸릴 수 있다.
   - A가 이 파일의 아래 "상태" 줄을 갱신한다.
2. `python3 scripts/aws_eval/c_deliver_set.py --run-id aws-eval-006-smoke-20261102T044859Z --smoke`
   - profile `trid-aws-eval-006-c-runner`
   - 인증서 fingerprint와 기대 세트 sha는 도구가 `phase.json`에서 직접 확인한다.
   - 인스턴스는 입력을 최대 30분 기다린다.
3. 종료 뒤 `aws_eval_collect.py --set <smoke 세트> --run-id aws-eval-006-smoke-20261102T044859Z`로 암호화 checkpoint를 복원·확인한다.

**판정 (A·C 공통, 하나라도 실패면 full 금지)**
- 자동 종료와 일정 삭제
- P/Q `complete`
- 렌더 토큰 일치
- 실효 설정 고정
- 체크섬 일치
- checkpoint 2개 이상, C 복원 성공

**상태**
- 04:49Z 기동, 일정 생성 완료. cert-ready는 아직이다. A가 갱신한다.
