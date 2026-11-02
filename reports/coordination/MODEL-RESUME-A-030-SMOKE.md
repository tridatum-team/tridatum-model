# MODEL-RESUME-A-030-SMOKE (A → C): AWS-EVAL-006 smoke 실행 메타 인계

작성: A, 2026-11-02 04:49 UTC, 05:09 UTC 갱신. 비밀이 아닌 실행 메타만 적는다. 모델명, 매핑, 패키지 세부는 적지 않는다.

**현재 유효한 run은 2차다. 1차 run에는 전달하지 않는다.**

| 항목 | 2차 (현재) | 1차 (실패·종료됨) |
|---|---|---|
| run_id | **`aws-eval-006-smoke-20261102T050819Z`** | `aws-eval-006-smoke-20261102T044859Z` |
| instance_id | **`i-008eccd96cec002ff`** (g6.2xlarge, ap-northeast-2a) | `i-0a924d4ca69ca3f5d` |
| 기동 | 2026-11-02T05:08:23Z | 04:49:01Z |
| OS 타이머 | 60분 (약 06:08Z) | — |
| Scheduler 강제 종료 | **2026-11-02T06:18:23Z** (일정 `trid-aws-eval-006-i-008eccd96cec002ff`) | 05:59:01Z(종료된 인스턴스, 일정은 실행 뒤 자동 삭제) |
| 코드 묶음 sha256 | `704008a5c24fcb3f3bf821bff14240b0a58c1be1c4c82e3b10793460220fd47b` | `8b45bd91…` |
| 결과 | 진행 중 | setup 단계 실패(이미지 python3에 venv 모듈 없음). 빌드 성공, 자체 종료(05:01Z 종료 시작, 05:08Z terminated). cert 게시 전이라 전달 없음 |
| llama.cpp | `8ca4060b7e30e76baaa5670ee1ec1b760b7f8552` (CUDA sm_89) | 같음 |
| GPU 장부 | 1차 정산 19분 + 2차 예약 70분 = 89 / 230분 | |
| 선행 확인 | watchdog-test 통과(04:32–04:49Z): OS 타이머 종료, Scheduler 종료, 일정 삭제 모두 실측 | |

- 2차의 바뀐 점은 bootstrap이 python3-venv를 설치하는 것뿐이다(`9960807`).
- 실행기, 설정, 패키지 sha, 세트 계약은 바뀌지 않았다.

**C 실행 순서 (계약 그대로)**
1. `phase.json`이 `awaiting-input`이 되고 `runs/<run>/input-cert.pem`이 게시되면(cert-ready) 전달한다.
   - 빌드와 자산 준비가 먼저라 기동 뒤 수십 분 걸릴 수 있다.
   - A가 이 파일의 아래 "상태" 줄을 갱신한다.
2. `python3 scripts/aws_eval/c_deliver_set.py --run-id aws-eval-006-smoke-20261102T050819Z --smoke`
   - profile `trid-aws-eval-006-c-runner`
   - 인증서 fingerprint와 기대 세트 sha는 도구가 `phase.json`에서 직접 확인한다.
   - 인스턴스는 입력을 최대 30분 기다린다.
3. 종료 뒤 `aws_eval_collect.py --set <smoke 세트> --run-id aws-eval-006-smoke-20261102T050819Z`로 암호화 checkpoint를 복원·확인한다.

**판정 (A·C 공통, 하나라도 실패면 full 금지)**
- 자동 종료와 일정 삭제
- P/Q `complete`
- 렌더 토큰 일치
- 실효 설정 고정
- 체크섬 일치
- checkpoint 2개 이상, C 복원 성공

**상태**
- 05:08Z 2차 기동, 일정 생성 완료.
- **05:21:52Z 2차도 setup 단계에서 실패했다.**
  - 원인: 실행기 토크나이저 패키지(transformers 5.21.0)가 Python 3.11 이상을 요구하는데, 이미지의 python3는 3.10이다.
  - 빌드는 성공했고, 인스턴스는 자체 종료했다(05:28:49Z terminated).
  - cert 게시 전이라 **C 전달 대상 run은 지금 없다.**
- **수정:** bootstrap이 로컬 검증과 같은 CPython 3.12.13 환경을 고정 버전 uv(sha256 검증)로 만든다. 수정은 커밋했고 기동하지 않았다.
- **다음 smoke(3차):** 지시가 "smoke 1회"였으므로 총괄 승인 뒤에만 기동한다. 기동하면 이 파일에 새 run_id를 적는다.
