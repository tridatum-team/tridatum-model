# MODEL-RESUME-A-031 (A): AWS-EVAL-006 실제 실행 — watchdog 통과, smoke 2회 setup 실패(원인 수정, 미기동)

작성: A, 2026-11-02 (04:32–05:35 UTC). A-030 CPU 개선 보고(`MODEL-RESUME-A-030.md`)는 그대로 두었다.

## 결론
- **watchdog-test: 통과(실측).** OS 타이머 종료, Scheduler 종료, 두 일정의 자동 삭제를 모두 확인했다.
- **smoke: 통과하지 못했다.** 2회 기동했고 두 번 모두 빌드 뒤 setup 단계에서 실패했다.
  - 두 인스턴스 모두 스스로 종료했다(`Client.InstanceInitiatedShutdown`).
  - cert 게시 전이라 C 전달, checkpoint, complete 판정은 **발생하지 않았다.**
- **full: 금지 상태다.** smoke 미통과다.
- **원인 수정:** 두 번째 원인까지 고친 bootstrap을 커밋했다(`b57887e`). 3차 smoke는 지시가 "smoke 1회"라 **기동하지 않았다.** 총괄 승인을 기다린다.
- **지킨 것:**
  - operator 전용으로 실행했다. root, c-runner, B 계정은 유료 실행에 쓰지 않았다.
  - 포트·정책·기존 자원 변경 0
  - IMDSv2, 암호화·삭제형 EBS, 무인바운드 SG 유지
  - C 문항, gold, 키 열람 0

## 1. 신원
- 사용자가 operator로 다시 로그인했다(login cache 발급 04:32:02Z). 세션 상한은 최대 16:32:02Z로, 시작 시 잔여 720분이었다.
- c-runner 세션 상한은 최대 13:42:11Z다.
- A-030 감시는 실행을 시작하지 않았다. 새 로그인만 감지하고 끝났다.

## 2. watchdog-test (실제 기동, t3.micro 2대, stamp `20261102T043245Z`)

| 인스턴스 | 기동 | 결과 | 상태 사유 | 일정 |
|---|---|---|---|---|
| os-timer `i-0e6945ad10818d1f2` | 04:32:47Z | 04:36:50Z 확인 시 terminated(3분 OS 타이머) | `Client.InstanceInitiatedShutdown` | 15분 backstop(04:47:47Z)이 실행된 뒤 04:48:50Z에 `deleted` |
| scheduler `i-048546d89da7a64b5` | 04:32:50Z | **04:39:15Z terminated**(일정 04:38:50Z, 약 25초 지연) | `Client.UserInitiatedShutdown`, "User initiated (2026-11-02 04:39:15 GMT)" | 04:39:50Z에 `deleted` |

- 따라서 다음이 실측으로 확인됐다.
  - Scheduler `CreateSchedule`과 `iam:PassRole`
  - Scheduler 역할의 AssumeRole과 TerminateInstances
  - 일회성 일정의 자동 삭제
  - OS 타이머
- 원문: `MODEL-RESUME-A-031/watchdog-launch.json`, `watchdog-status.txt`

## 3. smoke (g6.2xlarge, operator)

| | 1차 | 2차 |
|---|---|---|
| run_id | `aws-eval-006-smoke-20261102T044859Z` | `aws-eval-006-smoke-20261102T050819Z` |
| instance | `i-0a924d4ca69ca3f5d` | `i-008eccd96cec002ff` |
| 코드 묶음 sha | `8b45bd91…` | `704008a5…` |
| 기동 → 실패 | 04:49:01Z → 05:01:21Z | 05:08:23Z → 05:21:52Z |
| 단계 | start(L4 23,034 MiB, 드라이버 595.112.07) → build 통과(llama.cpp `8ca4060` CUDA sm_89, 약 11분) → **setup 실패** | 같음 → **setup 실패** |
| 원인 (bootstrap.log) | 이미지 python3에 `ensurepip`가 없어 venv를 만들 수 없음(python3-venv 미설치) | venv는 해결됐다. 그러나 `transformers==5.21.0`이 **Python ≥ 3.11**을 요구하는데 이미지 python3는 3.10이라 설치할 수 없음 |
| 종료 | EXIT trap 자체 종료, 05:08:04Z terminated | 같음, 05:28:49Z terminated |
| 장부 정산 | 19분 | 21분 |

- 두 정산 모두 기동부터 terminated 관측까지로 올림 계산했다. 과금은 shutting-down부터 멈추므로 일부러 많게 잡았다.
- **장부:** 40 / 230분 (`AWS-EVAL-006-GPU-LEDGER.json`)
- **수정 내역 (`scripts/aws_eval/bootstrap.sh`):**
  - 1차 뒤(`9960807`): python3-venv를 apt로 설치(dpkg lock 대기, 시간 제한)
  - 2차 뒤(`b57887e`): Python 환경을 apt 없이, 로컬 검증과 같은 **CPython 3.12.13**으로 만든다.
    - 고정 버전 **uv 0.11.28**을 쓴다. 릴리스 tar.gz sha256 `e490a646…`은 로컬에서 다운로드해 일치를 확인했다.
    - uv 0.11.28이 `cpython-3.12.13-linux-x86_64-gnu`를 받을 수 있음도 로컬 목록으로 확인했다.
    - 설치 패키지와 버전(transformers 5.21.0, tokenizers 0.23.3, jinja2 3.1.6)은 그대로다.
    - `bash -n` 통과, user-data 10,473 B(16 KB 한도 이내)
  - **바꾸지 않은 것:** 실행기, 공통 설정, 패키지 sha, 세트 계약, llama.cpp 커밋·빌드 플래그, 인스턴스 사양·보안 설정
- **아직 확인되지 않은 것 [추정 위험]:** setup 이후 단계는 원격에서 한 번도 실행된 적이 없다. 3차에서 처음 실측된다.
  - HF 자산 다운로드와 sha
  - cert 게시와 C 전달
  - P/Q 실행
  - checkpoint 업로드
- **S3에 남은 것:** 두 run의 `code.tar.gz`(각 16,061 B), `bootstrap.log`, `phase.json`만 있다. 결과 객체는 없다.
- **남은 일정:** 두 run의 강제 종료 일정(05:59:01Z, 06:18:23Z)은 이미 종료된 인스턴스를 대상으로 실행된 뒤 스스로 삭제된다. 삭제 확인은 다음 보고에 적는다.

## 4. 비용 (실측 시간 기준, 온디맨드 단가 = Pricing API)

| 자원 | 계산 | 금액 |
|---|---|---|
| g6.2xlarge | 장부 40분(과대) × $1.20208/h | ≤ **$0.80** |
| t3.micro | 2대 × 약 4–7분 × $0.013/h | < $0.01 |
| EBS gp3 | 100 GB × 2 × 20분 + 8 GB × 2 | < $0.02 |
| 공인 IPv4 | 4대 × 20분 이하 × $0.005/h | < $0.01 |
| S3·SSM·Scheduler | 소량 | ≈ $0 |
| **합계** | | **약 $0.84 이하**(약 1,200원) |

- 예산 대비: 추가 30만원 중 본평가 3만원, GPU 총 4시간 이하 범위 안이다.

## 5. full 가능성 (계산만, 이번 배정 아님)
- **조건:** smoke 통과가 먼저다. 지금은 불가다.
- **장부 계산:**
  - 남은 장부는 190분이다.
  - 3차 smoke를 승인하면 70분을 예약하고, 끝난 뒤 실제 시간으로 정산한다. 정상이면 빌드 11분 + setup·자산 다운로드 + C 전달 대기 + 2턴 실행으로 약 35–45분 예상 [추정].
  - 정산 후 남는 시간은 약 145–155분이다.
  - full 계약 예약은 160분(150 + 10)이다. 남는 시간에 들어가지 않을 수 있다.
  - 이 경우 `--max-minutes`를 135–145로 낮추거나(입력 대기 30분 포함), 총괄이 장부 한도를 다시 정해야 한다.
- **자격 증명:**
  - operator는 16:32Z까지다.
  - c-runner는 13:42Z까지다. C 전달·수집이 그 전에 끝나야 한다. 그렇지 않으면 사용자가 c-runner로 다시 로그인해야 한다.
- **비용:** full 150분 × $1.20208/h ≈ $3.0 + EBS·IPv4 소액. 본평가 3만원 범위 안이다.

## 6. 다음 행동 (승인 필요)
- **3차 smoke 1회:** 승인되면 `AWS_PROFILE=trid-aws-eval-006-operator .venv/bin/python scripts/aws_eval/aws_eval_006.py launch --mode smoke`를 실행한다.
  - 시작 전 `preflight`를 다시 한다.
  - C에는 `MODEL-RESUME-A-030-SMOKE.md`로 새 run을 인계하고, cert-ready를 알린다.
  - 판정 하나라도 실패하면 full은 금지다.
- **같은 원인이 다시 나오면:** 재시도하지 않고 보고한다.

SUPERVISION_REPORT_READY MODEL-RESUME-A-031
