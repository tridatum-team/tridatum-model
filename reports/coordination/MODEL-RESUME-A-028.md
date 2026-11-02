# MODEL-RESUME-A-028 (A): AWS-EVAL-006 읽기 전용 preflight 실행, 자동 종료 실측 준비

작성: A, 2026-11-02 (관측 시각 01:53–01:57 UTC).

**결론**
- operator 신원으로 계약 순서 1단계(`preflight` + `launch --mode smoke --dry-run`)를 실제로 실행했다. 모두 통과했다.
  - 신원, 리전, 할당량, AMI, 인스턴스 프로파일, scheduler 역할, 보안 그룹, 비용 gate 확인
  - EC2 DryRun은 "would succeed"
- 다음 단계인 `watchdog-test`(자동 종료 실측)는 실행하지 않았다. 실행할 명령, 범위, 비용 상한, 판정 기준은 4절에 있다.
- 만든 자원·시작한 자원·바꾼 정책은 0이다(3.3절).
- 코드 변경은 `watchdog-test`의 안전 보완과 읽기 전용 확인 명령 하나뿐이다(2절).

**지키고 있는 것**
- 후보 `ff59b93`: C-028 검증 중이라 동결했다. 건드리지 않았다.
- 운영 `6a5eb37`과 맥미니: 변경 0
- 기본 profile(root): 쓰지 않았다. 모든 AWS 호출은 `AWS_PROFILE=trid-aws-eval-006-operator` 또는 `--profile`을 명시했다. 환경변수 자격 증명이 없음도 확인했다.
- c-runner profile: STS 신원 확인에만 썼다.
- 비공개 문항, gold, P/Q 매핑 접근 0. B collector와 C 문항 파일 변경 0.

## 1. 실제 관측 (실행한 명령과 결과)

### 1.1 신원과 자격 증명 여유 (`check_identity.py`, 비밀값 출력 없음)

| profile | STS ARN | 현재 임시 자격 증명 만료 | 로그인 세션 시작(idToken iat) | 세션 상한 |
|---|---|---|---|---|
| trid-aws-eval-006-operator | `arn:aws:iam::148761639313:user/trid-aws-eval-006-operator` | 02:08:38Z (15분 주기 자동 갱신) | 2026-11-01 17:48:59Z | **최대 05:48:59Z (14:49 KST)** |
| trid-aws-eval-006-c-runner | `arn:aws:iam::148761639313:user/trid-aws-eval-006-c-runner` | 02:08:39Z | 2026-11-02 01:42:11Z | 최대 13:42:11Z (22:42 KST) |

- **근거:** AWS 문서(<https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sign-in.html>)에 따르면 `aws login` 자격 증명은 15분마다 자동 갱신되고, 세션은 IAM 주체의 세션 기간(최대 12시간)까지 유효하다.
- **[추정]:** 위 "세션 상한"은 iat + 12시간이다. 사용자별 실제 세션 기간 설정은 operator 권한으로 읽을 수 없다. 실제 만료는 더 이를 수 있다.
- **여유:** operator는 관측 시점 기준 최대 약 3시간 52분 남았다.
  - `watchdog-test`(15분 이내) 한 번에는 충분하다.
  - smoke GPU(최대 70분)는 05:48Z 전에 끝나려면 늦어도 04:30Z 전에 시작해야 한다. 그 전에 끝나지 않으면 status 확인과 수동 terminate는 새 로그인 뒤에만 할 수 있다. 다만 인스턴스 종료 자체는 OS 타이머와 Scheduler가 자격 증명과 무관하게 수행한다.
- **만료 시:** 새 로그인을 임의로 하지 않는다. 사용자에게 요청한다.

### 1.2 `aws_eval_006.py preflight` (계약 스크립트, 읽기 전용)

| 항목 | 값 |
|---|---|
| identity / region | operator ARN / ap-northeast-2 |
| G·VT vCPU 할당량 (L-DB2E81BA) | **48** (g6.2xlarge 1대 = 8). 스크립트 주석의 "8"보다 크다 |
| 실행 중 인스턴스 | 0 |
| g6.2xlarge 제공 AZ | 2a, 2c, 2d (스크립트의 세 subnet과 일치) |
| AMI `ami-032ead1ff8d54d3f5` | Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20261027, available |
| 인스턴스 프로파일 / scheduler 역할 | 있음 / 있음 |
| 보안 그룹 | sg-0e7e5b23c7c152ebc |
| B 비용 gate | pass |

### 1.3 추가 읽기 전용 점검 (`aws_readonly_check.py`, 결과 `aws-readonly-check.json`)
- **기존 리소스:**
  - aws-eval-006 태그 인스턴스는 상태와 무관하게 0개다.
  - 종료되지 않은 인스턴스는 stopped 3대뿐이다: kms2026-gpu-smoke(g6e.xlarge), trid-c-taskgen-01(g6e.xlarge), trid-c-taskgen-32b-01(g6e.2xlarge).
  - 볼륨 3개(100/150/300 GB)는 모두 위 인스턴스에 붙어 있다. 그중 2개는 암호화되지 않았다(C 작업용 2대).
  - 모두 기존 자원이라 건드리지 않았다(INFRA 9절과 같음).
- **S3:** `eval/aws-eval-006/` prefix 객체 0개. GPU 예약 장부(`AWS-EVAL-006-GPU-LEDGER.json`)는 없다. 이번 승인 230분 중 사용량은 **0분**이다.
- **할당량:** 표준 vCPU(L-1216C47A) 64. watchdog용 t3.micro 2대 = 4 vCPU라 여유가 있다.
- **IAM (`get-role` 읽기):** 두 역할의 trust 문서가 저장소 `deploy/aws-eval-006/*-trust.json`과 같다. scheduler는 `aws:SourceArn` = `schedule-group/default` 정확 일치이고, 최대 세션은 3600초다. 인스턴스 프로파일의 역할은 `trid-aws-eval-006-instance`다.
- **보안 그룹:** 인바운드 규칙 0, 아웃바운드 전체 허용(SSM·S3·HF 다운로드용, 설계대로)
- **subnet 3개:** default VPC, 공인 IP 자동 할당, 여유 IP 4,088–4,091
- **Scheduler 권한:**
  - `GetSchedule`은 허용된다(존재하지 않는 이름 → ResourceNotFound).
  - `ListSchedules`는 **AccessDenied**다(operator 정책에 없음, 설계대로).
  - 그래서 operator는 기존 일정을 목록으로 볼 수 없고, 이름을 알 때만 확인할 수 있다.
- **watchdog AMI (SSM 공개 파라미터):** `ami-05becc39a92b78fa5`(al2023 2023.12.20261102.0, Amazon 소유, root `/dev/xvda`, available)
- **On-demand 단가 (Pricing API, 서울):** g6.2xlarge **$1.20208/h**, t3.micro **$0.013/h**

### 1.4 EC2 DryRun (자원 생성 없음, 권한·매개변수 검증만)
- `launch --mode smoke --dry-run` → `DryRunOperation (would succeed) in subnet-025ee7d30ecf0228c`
  - 코드 묶음 sha `7ef6cb89…`
  - S3 업로드·장부 기록 없음
- `watchdog-test --dry-run` → os-timer, scheduler 두 경우 모두 "would succeed". 변경 전과 변경 후에 각각 한 번씩 돌렸다.

## 2. 이번 변경 (자동 종료 실측 준비, `scripts/aws_eval/aws_eval_006.py`)

**변경 1: OS 타이머 인스턴스에도 Scheduler backstop**
- 기존 `watchdog-test`는 OS 타이머 인스턴스(3분 `shutdown -h`)에 Scheduler 일정을 두지 않았다. OS 타이머가 실패하면 그 인스턴스를 끝낼 독립 장치가 없었다.
- 이제 그 인스턴스에도 15분 Scheduler backstop을 둔다.
- 시험의 뜻은 그대로다. 3분에 끝나면 OS 타이머가 끝낸 것이고, 15분 일정은 실패할 때만 쓰인다.
- 출력에 stamp, 일정 이름, 일정 시각을 넣었다.

**변경 2: `watchdog-status <stamp>` (읽기 전용) 추가**
- 두 인스턴스의 상태와 StateReason을 보여 준다.
  - `Client.InstanceInitiatedShutdown` = OS 타이머
  - `Client.UserInitiatedShutdown` = TerminateInstances, 여기서는 Scheduler
- 일정이 남았는지 삭제됐는지도 `GetSchedule`로 보여 준다.
- 기존 `status`는 run_id 하나만 보기 때문에 판정 근거가 부족했다.

**바꾸지 않은 것**
- 정책, IAM, 인스턴스 사양, 암호화, 태그, GPU 경로(`launch`), 장부 규칙, 시간 상한
- 이 파일은 A 소유 계약 파일이고 기존 WIP가 없었다.

**검증 (마지막에 한 번)**
- `watchdog-test --dry-run` → 두 경우 would succeed
- `watchdog-status 20261102T000000Z` → 빈 결과. 해당 인스턴스 0개, 오류 없음.
- `--help`에 새 명령이 나온다.

**문서:** `AWS-EVAL-006-INFRA.md` 7절 1단계에 완료를 적고, 2단계에 backstop과 `watchdog-status` 판정 기준을 추가했다.

## 3. 미실행, 확인 불가, 차단

### 3.1 미실행 (지시 범위 밖)
- `watchdog-test` 실제 기동(t3.micro 2대)
- smoke·full GPU 기동
- S3 업로드
- 정책·권한 변경
- c-runner의 세트 전달

### 3.2 이번 범위로 확인할 수 없는 것 (실제 watchdog-test로만 확인됨)
- Scheduler `CreateSchedule`과 그 `iam:PassRole`. EventBridge Scheduler에는 DryRun이 없다.
- Scheduler 역할의 실제 AssumeRole과 TerminateInstances
- 일회성 일정의 자동 삭제
- 인스턴스 OS 타이머 종료
- 인스턴스 역할의 SSM 등록

### 3.3 자원 변화 확인
- 마지막 점검 결과:
  - aws-eval-006 태그 인스턴스 0
  - S3 prefix 객체 0(KeyCount null)
  - 장부 파일 없음
- 이번 작업으로 만든 EC2, 볼륨, 일정, S3 객체, IAM 변경은 0이다.

### 3.4 차단·주의
- **operator 로그인 세션 상한:** 최대 05:48:59Z(1.1절). 그 뒤에는 사용자의 새 로그인이 필요하다.
- **기존 일정 확인:** operator는 `ListSchedules` 권한이 없어 이름 없이 남은 일정을 찾을 수 없다. 이번 승인 경로의 일정 이름은 모두 `trid-aws-eval-006-<instance-id>`라 인스턴스 id로 확인할 수 있다.

## 4. 다음 실제 자동 종료 smoke: 정확한 명령과 범위 (별도 승인 뒤 실행)

**전제**
- 별도 승인이 있다.
- operator 세션이 30분 이상 남았다. 확인 명령: `.venv/bin/python reports/coordination/MODEL-RESUME-A-028/check_identity.py`
- 저장소 루트에서 실행한다.

**명령**
```bash
# 0. 신원 재확인 (root면 스크립트가 유료 기동을 거부한다)
AWS_PROFILE=trid-aws-eval-006-operator .venv/bin/python scripts/aws_eval/aws_eval_006.py preflight
# 1. 기동: 출력의 stamp, 두 instance_id, 일정 이름·시각을 기록
AWS_PROFILE=trid-aws-eval-006-operator .venv/bin/python scripts/aws_eval/aws_eval_006.py watchdog-test
# 2. 판정: +4분, +7분, +16분에 한 번씩
AWS_PROFILE=trid-aws-eval-006-operator .venv/bin/python scripts/aws_eval/aws_eval_006.py watchdog-status <stamp>
# 3. 실패 시에만: 남은 인스턴스를 즉시 종료
AWS_PROFILE=trid-aws-eval-006-operator .venv/bin/python scripts/aws_eval/aws_eval_006.py terminate aws-eval-006-watchdog-os-timer-<stamp>
AWS_PROFILE=trid-aws-eval-006-operator .venv/bin/python scripts/aws_eval/aws_eval_006.py terminate aws-eval-006-watchdog-scheduler-<stamp>
```

**판정 기준**
- **통과:**
  - os-timer: 기동 후 약 3–4분에 `terminated`, `Client.InstanceInitiatedShutdown`
  - scheduler: 약 6–7분에 `terminated`, `Client.UserInitiatedShutdown`
  - scheduler 일정 `deleted (not found)`
  - os-timer의 15분 backstop 일정: 실행 시각이 지나면 삭제된다. 인스턴스는 이미 종료됐으므로 TerminateInstances는 효과가 없다. 그 전에 확인하면 `exists`다.
- **실패:**
  - os-timer가 15분 backstop으로 끝나면 OS 타이머 실패다.
  - scheduler 인스턴스가 7분 넘게 running이면 Scheduler 실패다. 3번 명령으로 즉시 종료하고 원인(PassRole·AssumeRole·trust)을 보고한다.
  - `CreateSchedule`이 실패하면 스크립트가 해당 인스턴스를 즉시 terminate한다. 이때 다른 인스턴스는 남을 수 있으니 3번으로 정리한다.

**범위와 비용 상한**
- 만드는 것:
  - t3.micro 2대(Amazon Linux 2023, default VPC subnet-025ee7d30ecf0228c, 무인바운드 SG, IMDSv2, 태그 `trid:eval=aws-eval-006`)
  - 8 GB gp3 암호화 볼륨 2개(종료 시 삭제)
  - 자기 삭제 일회성 일정 2개
- 만들지 않는 것: IAM·정책 변경
- 비용: 각 최대 15분 → 2 × 0.25 h × $0.013 ≈ **$0.007**에 EBS 수 센트 미만
- GPU: 0분. GPU 장부에 영향이 없다.
- 실행 시간: 약 20분(판정 포함)

**그다음 단계 (참고, 이번 승인 범위 밖)**
- `launch --mode smoke`(g6.2xlarge 1대, OS 60분 + Scheduler 70분 상한)
  - 장부 예약 70분 / 230분
  - 비용 상한 70/60 × $1.20208 ≈ **$1.40**, EBS 100 GB 소액
- C가 c-runner로 `c_deliver_set.py --run-id <smoke run> --smoke`를 실행한다.
- full은 C precommit 부록 뒤 150 + 10분이다.
- 기존 승인 한도 안이다: 추가 30만원 중 본평가 3만원, GPU 총 4시간 이하, 계획 230분(실패 종료 포함).

## 5. 파일
- `reports/coordination/MODEL-RESUME-A-028/`
  - `check_identity.py`: 신원과 자격 증명 만료, 비밀값 출력 없음
  - `aws_readonly_check.py`: 읽기 전용 점검
  - `aws-readonly-check.json`
  - `identity-and-credential-expiry.txt`
  - `launch-smoke-dry-run.txt`
  - `watchdog-test-dry-run-before-change.txt`
- `scripts/aws_eval/aws_eval_006.py`: 2절 변경
- `reports/coordination/AWS-EVAL-006-INFRA.md`: 7절 갱신

SUPERVISION_REPORT_READY MODEL-RESUME-A-028
