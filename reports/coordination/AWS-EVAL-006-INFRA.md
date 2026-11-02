# AWS-EVAL-006-INFRA (A): 단기 EC2 GPU 평가 인프라 — 준비 완료, 신원 승인 대기로 실행 0

**상태: 차단됨.**
- 차단 사유는 확인된 **실행 신원**이다. 유료 기동과 IAM 쓰기를 할 수 있는 신원이 root 하나뿐이다.
- **도달 가능한 준비는 모두 끝냈다.**
- **AWS에 만든 자원과 쓰기는 0건이다.**
  - 원격 공개 smoke, 독립 자동 종료 실측, 맥미니 경로 실측은 하지 않았다. 아래 6절의 1회 bootstrap이 승인돼야 할 수 있다.
- 학습, 사이트 모델 변경, SageMaker는 모두 0이다. 실서비스도 바꾸지 않았다.

## 1. 준비와 실행 구분

| 항목 | 준비 (A, 완료) | 실행·실측 (미실행, 사유) |
|---|---|---|
| 신원·권한 | 최소 권한 정책 4개와 trust 2개 작성. Access Analyzer 검증 finding 0, 허용·거부 19건 시뮬레이션 모두 기대대로 | 생성 0. root IAM 쓰기 금지, 사용자 승인 대기 |
| 인스턴스 기동 | 운영 스크립트, user data, 코드 묶음(결정적) 완성. 렌더 7,714 B (한도 16 KB). root 거부 확인 | 0. non-root 주체 없음 |
| 자동 종료 3중 | ① OS 타이머 + 종료 시 terminate ② 완료·실패 시 trap 종료 ③ EventBridge Scheduler 1회 terminate(인스턴스·맥과 독립) | 실측 0. t3.micro 2대 시험 명령 준비 |
| CUDA llama.cpp | 커밋 `8ca4060b7e30e76baaa5670ee1ec1b760b7f8552`, `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89`, `-ngl 999`. 바이너리 sha와 backend 문자열 기록 | 빌드·검증 0 |
| 모델·tokenizer | 공식 HF 고정 revision 11개 자산. LFS 4개는 HEAD의 `x-linked-etag`가 기대 sha256과 같고, 작은 파일 7개는 내려받아 sha256 일치(평가 문항 아님) | 원격 다운로드 0 |
| runner 원격판 | Linux 호스트·GPU 메모리 guard, 대화별 C 인증서 CMS checkpoint. 로컬에서 C 수집기 왕복 일치 | 원격 0 |
| 맥미니 경로 | 읽기 전용 점검만 | **차단.** 3절 |
| 예산 | 장부에 30,000원 예약(미예약 190,000 → 160,000원) | 사용 0원 |

## 2. 실제 권한 (읽기 전용 조회, 2026-10-31 12:48~13:03 KST)

- **신원:** 맥북 `default`와 `kms2026` 프로필 모두 `arn:aws:iam::148761639313:root`다. 맥북에서 쓸 수 있는 신원은 이것뿐이다.
- **IAM 사용자:** `snowflake_s3` 하나다. `AmazonS3ReadOnlyAccess`, access key 1개, 콘솔 없음. trid가 아니고 EC2 권한도 없다.
- **IAM Identity Center:** 인스턴스 0.
- **역할:** `kms2026-gpu-smoke-ssm`(EC2, SSM, **버킷 전체** Get/Put), `trid-sagemaker-lora`, `trid-sagemaker-c009-taskgen`(SageMaker trust).
  - 지시대로 SageMaker 역할의 trust나 권한을 바꿔 우회하지 않는다.
  - kms2026 역할은 버킷 전체 권한이라 최소 권한이 아니고 다른 프로젝트 것이다. 재사용하지 않는다.
- **맥미니:**
  - `trid-controller` 프로필이 `login_session = …:root`(root 감시 전용)이고 **세션이 만료됐다**(`aws login` 재인증 필요).
  - `default` 프로필은 없다.
- **결론:** 이번 작업에 허가된 non-root 주체는 **없다.**
  - MACMINI-DEPLOY-002 원칙("root는 감시 전용, 발행 금지")이 유지되므로 root로 유료 기동이나 IAM 쓰기를 하지 않았다.
  - 운영 스크립트도 root면 유료 기동을 거부한다. 실제 출력: `refused: caller is the root user; …`

## 3. 맥미니 경로 (읽기 전용 확인, 차단)

| 확인 | 결과 |
|---|---|
| 메모리 압박 단계 | **2 (경고)**. 이번 작업 중 두 번 확인했다(12:5x, 13:0x). 원칙에 따라 맥미니에서는 아무것도 실행하지 않았다 |
| AWS CLI | v2.37.13 있음. 유일한 프로필은 root login이고 만료됐다 |
| session-manager-plugin | **없음**. 맥북에는 있다. 설치는 새 설치라 하지 않았다 |
| 운영 서비스 | `co.tridatum.secho.*`, `co.tridatum.rag.*` 그대로. 건드리지 않았다 |

- **설계상 맥미니는 필수가 아니다.**
  - 인스턴스는 user data만으로 빌드, 다운로드, 실행, 업로드, 종료를 스스로 끝낸다.
  - 시간 상한은 인스턴스 OS 타이머와, 인스턴스·맥과 무관한 EventBridge Scheduler가 건다.
  - 그래서 기동 직후 맥북을 꺼도 작업과 종료가 진행된다(설계 기준, 미실측).
- **맥미니가 할 일(선택):** 모니터링(`status`)과 C의 문항 전달.
  - 조건: c-runner 또는 operator 로그인, 메모리 정상화
  - SSM 세션 방식을 쓰려면 plugin 설치 승인도 필요하다. 아래 5절의 `send-command` 방식은 plugin 없이 된다.

## 4. 자원·단가·견적

- **사양:** `g6.2xlarge` (NVIDIA L4 22,888 MiB(약 24 GB), 8 vCPU, 32 GiB), 서울 온디맨드 **$1.20208/h** (Price List, B와 같음).
- **쿼터:** G/VT on-demand 8 vCPU라 1대만 가능하다. 제공 AZ는 2a, 2c, 2d. 현재 running 인스턴스 0.
- **AMI:** `ami-032ead1ff8d54d3f5` = Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20261027. ID로 고정했다.
- **디스크:** gp3 100 GB, 암호화, 종료 시 삭제.
- **시간 상한:** smoke 60분 + full 150분(문항 대기 30분 포함) = **최대 3.5시간**(4시간 한도 안).
- **견적:** 스크립트로 계산했다(1,500원/USD, 부가세 10%).
  - GPU $4.207 + EBS $0.044 + IPv4 $0.018 + 감시 시험 t3.micro 2대(각 최대 12분) $0.0076 + S3 $0.05 = **$4.33 ≈ 7,138원**
  - 이번 cap 30,000원 안이다.
- **예산 장부:** `BUDGET-300K-RESERVATION-001.md`에 AWS-EVAL-006 예약 30,000원을 추가했다(미예약 190,000 → 160,000원). 기존 A·C 예약과 겹치지 않는다.
  - B의 실사용 추정 잔액 약 260,863원은 Cost Explorer 집계 지연과 재구성에 기댄 추정이다.

## 5. 설계 (파일)

**운영 스크립트 `scripts/aws_eval/aws_eval_006.py`** (AWS CLI v2 사용, boto3 설치 없음)

| 하위 명령 | 동작 |
|---|---|
| `preflight` | 읽기 전용 점검 |
| `bootstrap-identity --user-approved` | 1회. 사용자가 승인한 경우에만 계정 소유자가 실행 |
| `launch --mode smoke\|full [--dry-run]` | root 거부. B 비용 보고에 계정과 리전이 없으면 거부 |
| `watchdog-test` | t3.micro 2대로 자동 종료 실측 |
| `status RUN_ID` | 출력은 내려받지 않는다 |
| `terminate RUN_ID` | 즉시 종료 |

**인스턴스 사양**
- IMDSv2 필수(hop 1), 인바운드 0인 전용 SG, 공인 IPv4(외부 인터넷 송신용, 인바운드 없음)
- `InstanceInitiatedShutdownBehavior=terminate`, 태그 `trid:eval=aws-eval-006`, `trid:run-id`
- 기동 직후 `trid-aws-eval-006-<instance>` 1회 Scheduler 일정을 **default 일정 그룹**에 만든다. 시각은 상한 + 10분이고, 끝나면 일정이 자동 삭제된다. **일정 생성에 실패하면 인스턴스를 즉시 종료한다.**
- **시간 상한 검사:** `--max-minutes`는 1..60(smoke), 1..150(full) 정수만 받는다. 0, 음수, 초과는 거부한다.
- **GPU 예약 장부:** `AWS-EVAL-006-GPU-LEDGER.json`. 유료 기동마다 (상한 + 10분)을 예약하고, 이번 승인 전체 합계는 **최대 230분**이다((60+10) + (150+10), 4시간 안).
  - 합계를 넘으면 재기동과 재시도를 거부한다.
  - run-instances가 실패해 인스턴스가 없을 때만 예약을 0으로 돌리고, 그 기록은 지우지 않는다.
- **원자성 한계:** run-instances가 성공한 뒤 Scheduler 일정이 생기기까지 수 초 동안은 **인스턴스 OS 타이머가 유일한 방어**다. 그 사이 운영 프로세스나 연결이 끊기면 Scheduler 상한은 걸리지 않고 OS 타이머만 남는다. 완전히 원자적인 자동 종료라고 주장하지 않는다.

**user data `scripts/aws_eval/bootstrap.sh`**
1. 첫 명령으로 `shutdown -h +MAX`를 건다.
2. 코드 묶음 sha를 확인한다.
3. llama.cpp를 고정 커밋으로 CUDA 빌드한다.
4. pip 고정 버전(transformers 5.21.0, tokenizers 0.23.3, jinja2 3.1.6)을 설치한다.
5. 자산을 받으며 sha256을 확인한다. 불일치면 중단한다.
6. P/Q 별칭을 만든다. 모델은 package.json의 기대 sha로 찾으므로 매핑 원문이 필요 없다.
7. 실행 방식
   - **세트 전달 (smoke·full 공통, S3 미사용)**
     - 인스턴스가 부팅 때 RSA 키쌍을 만든다. 개인키는 RAM(`/run` tmpfs)에만 둔다.
     - 인증서만 `runs/<run>/input-cert.pem`에 올리고, phase에 SHA-256 지문과 기대 세트 sha를 적는다.
     - C는 `scripts/aws_eval/c_deliver_set.py`로 지문과 sha를 대조한다. 세트를 그 인증서로 CMS 암호화해 base64 조각(24,000자)으로 `AWS-RunShellScript` SendCommand를 보내고, 마지막에 `.ready`를 보낸다.
     - 인스턴스는 복호화하고 sha를 확인한 뒤 키와 암호문을 shred한다. 최대 30분 기다린다.
   - smoke: 공개 세트 첫 시나리오의 첫 2턴. **C가 같은 전달 경로로 보내므로, 이 경로를 비공개 실행 전에 실측한다.**
   - full: 비공개 80, 기대 sha `af46eec1…`
8. P, Q를 순차 실행한다(`TRID_RUNNER_REMOTE=1`, `-ngl 999`). checkpoint는 30초마다 S3에 동기화한다.
9. 끝에 `outputs.jsonl`과 모든 `sealed/` 파일을 C 인증서로 CMS 암호화하고 평문을 shred한다. 전달된 세트도 shred한다.
10. 업로드한다.
11. EXIT trap이 로그와 phase를 올리고 전원을 끈다.

**C 인증서:** `AWS-EVAL-006/c-checkpoint-cert.pem` sha256 `f23ebf28…7082`. 원격에서 sha를 대조한다.

**runner 원격판** (`scripts/base_dialogue_runner.py`, `scripts/chat_blind_runner.py`)
- env를 주지 않으면 로컬 동작은 그대로다(`REMOTE False`, Metal 서버, 맥 압박 단계와 실서비스 guard).
- 원격(`TRID_RUNNER_REMOTE=1`)
  - 서버 경로와 인자는 env로 준다.
  - guard: `/proc/meminfo` 가용 10% 미만 또는 GPU 여유 512 MiB 미만이면 단계 2로 보고 대화 경계에서 멈춘다.
  - 실서비스 확인은 `not-applicable`로 기록한다.
- `run()`에 대화 종료 hook을 넣었다. `chat_blind_runner --checkpoint-dir/--checkpoint-cert`가 대화마다 C 계약(`scripts/aws_eval_collect.py` 상단)대로 checkpoint를 만든다.
  - 내용: alias, dialogue, rows, status(package, shared settings sha, backend, parity, effective settings sha)
  - **평문을 디스크에 쓰지 않고** stdin으로 암호화한다.

**S3:** `trid-training-148761639313`(공개 차단 4개 모두 True, SSE AES256), prefix `eval/aws-eval-006/` 하나.
- `code/<run>`: A가 쓰고 인스턴스가 읽는다.
- `runs/<run>/{phase.json, bootstrap.log, checkpoints/*.cms, results/…}`: 인스턴스만 쓴다.
- **비공개 문항은 S3에 올리지 않는다**(C 계약).

## 6. 차단 해소에 필요한 것 (정확히 1회)

**사용자 승인 후 계정 소유자가 실행:**
```
python3 scripts/aws_eval/aws_eval_006.py bootstrap-identity --user-approved
```

**만들어지는 것** (모두 무료, 태그 `trid:eval=aws-eval-006`, 정책 원문은 `deploy/aws-eval-006/*.json`)

| 주체·자원 | 권한 범위 |
|---|---|
| 역할 `trid-aws-eval-006-instance` + 인스턴스 프로필 | `AmazonSSMManagedInstanceCore` + S3 `code/*` 읽기, `runs/*` 쓰기, `runs/` 목록 |
| 역할 `trid-aws-eval-006-scheduler` | trust는 scheduler.amazonaws.com이고, `aws:SourceAccount` = 148761639313과 `aws:SourceArn` = `arn:aws:scheduler:ap-northeast-2:148761639313:schedule-group/default`(정확 일치, StringEquals)로 제한. `ec2:TerminateInstances`는 태그가 있는 인스턴스만 |
| SG `trid-aws-eval-006-no-ingress` | 인바운드 0 |
| IAM 사용자 `trid-aws-eval-006-operator` (A) | **access key 없음.** `SignInLocalDevelopmentAccess`(`aws login`용) 부착. g6.2xlarge·t3.micro만, 태그·IMDSv2·암호화 볼륨·기본 VPC 조건 RunInstances. 태그 인스턴스만 종료. 두 역할만 PassRole. 이름 접두가 맞는 Scheduler 일정. S3 `code/*` 쓰기와 상태 파일만 읽기(checkpoint·출력 읽기 불가). 태그 인스턴스 SSM 세션 |
| IAM 사용자 `trid-aws-eval-006-c-runner` (C) | **access key 없음.** `SignInLocalDevelopmentAccess` 부착. 태그 인스턴스에 `AWS-RunShellScript` SendCommand(암호화된 세트 전달), 암호화 checkpoint·결과·phase·전달 인증서 읽기, `runs/` 목록. 종료·쓰기 불가 |

**사용자 콘솔 작업:** 두 사용자에 콘솔 비밀번호와 MFA를 설정한다. 이후 `aws login --profile trid-aws-eval-006-operator`(A, 맥북 또는 맥미니)와 `aws login --profile trid-aws-eval-006-c-runner`(C)를 쓴다. 세션은 최대 12시간이다.

**정책 검증 결과 (읽기 전용 API)**
- `accessanalyzer validate-policy`: 6개 문서 모두 finding 0
- `iam simulate-custom-policy` 19건이 모두 기대와 같다.

  | 결과 | 사례 |
  |---|---|
  | 허용 | 태그 종료, g6.2xlarge 태그·IMDSv2 기동, phase 읽기, C의 태그 인스턴스 SendCommand와 checkpoint 읽기, scheduler의 태그 종료 |
  | 거부 | 태그 없는 종료, p5.48xlarge, IMDSv1, 비암호화 볼륨, `iam:CreateRole`, SageMaker 학습, `trid-sagemaker-lora` PassRole, operator의 checkpoint·corpus 읽기, C의 Put과 종료, scheduler의 태그 없는 종료 |

## 7. 차단 해소 후 순서 (추가 질문 없이 권한 범위 안에서 진행)

1. `preflight`와 `launch --mode smoke --dry-run`(operator 신원). **완료 (MODEL-RESUME-A-028, 2026-11-02 01:53–01:56Z):** preflight 통과, launch·watchdog DryRun 모두 "would succeed", 만든 자원 0.
2. `watchdog-test`로 독립 종료를 실측한다.
   - OS 타이머 인스턴스가 3분 뒤 terminated(`StateReason` `Client.InstanceInitiatedShutdown`)
   - scheduler 인스턴스가 6분 뒤 terminated(`Client.UserInitiatedShutdown`)
   - 일정 자동 삭제 확인
   - A-028 보완: OS 타이머 인스턴스에도 15분 Scheduler backstop을 둔다. OS 타이머가 실패해도 독립 상한이 있다. 어느 장치가 끝냈는지는 `watchdog-status <stamp>`(읽기 전용)가 상태 사유로 구분한다.
3. `launch --mode smoke`를 한다. C가 `c_deliver_set.py --smoke`로 공개 2턴을 암호화 전달하고, P와 Q 모두 실행한다. 확인 항목:
   - complete, 렌더 토큰 일치 true, 실효 설정 고정
   - GPU 메모리 최대치(`gpu.csv`), CUDA·드라이버·바이너리 sha
   - checkpoint 2개 이상이 암호화돼 S3에 올라감, C 복원 확인
   - 정상 종료와 일정 삭제
4. 이 보고에 실측을 추가하고 C에 실행 가능을 인계한다.
5. C가 precommit 부록을 커밋하면 `launch --mode full`을 한다.
6. C가 c-runner로 `c_deliver_set.py --set <비공개 파일>`을 실행해 전달한다. A는 이 명령을 비공개 파일로 실행하지 않는다.

## 8. 인계

- **B:** 비용 검증 완료를 받았다. 장부에 30,000원 예약을 반영했다. 실행 후 실제 시간과 CE 반영분으로 정산을 요청할 예정이다. 추가 요청은 없다.
- **C (REQUEST-AB 항목별)**

  | 항목 | 상태 |
  |---|---|
  | 1. 비용·자동 종료 | 설계 완료, 실측 대기 |
  | 2. 환경 기록 | 빌드 플래그 확정. 바이너리 sha와 드라이버·CUDA는 smoke 때 기록 |
  | 3. 패키지 | **005와 같은 package.json(P `14c8dce1…`, Q `c381396d…`)과 commitment `bc582558…`을 그대로 쓴다.** 원격 sha 불일치면 runner가 거부한다. 완전 블라인드가 아니다(패키지 내용, 문체, 속도) |
  | 4. 실행기 | `run()` 로직 그대로, 공통 설정 그대로, 대화별 CMS checkpoint 계약 구현. 로컬에서 C 수집기 왕복 일치 확인(일회용 인증서, 평가 문항 아님) |
  | 5. S3 | 비공개 버킷, prefix 하나 |
  | 6·7. smoke와 실행 경로 | 차단 해소 뒤 |

  - **C에게 노출하지 않은 것:** 매핑, readlink 결과, 모델 실명·hash.
  - C는 아직 실행할 수 없다.

## 9. 현재 자원 상태 (변경 0)

- 이 작업으로 만든 EC2, IAM, SG, Scheduler, S3 객체는 0이다.
- 기존 stopped 인스턴스 3대(C 2대, kms2026 1대)와 EBS는 B 보고대로 그대로 두었고 건드리지 않았다.

## 10. 다음 행동

- **사용자:** 6절 1회 bootstrap(root IAM 쓰기)을 승인할지 결정하고, 승인하면 두 사용자의 콘솔 비밀번호와 MFA를 설정한다.
- **A:** 승인과 로그인이 확인되면 7절 1~4를 즉시 진행한다. 그 전에는 유료 기동과 IAM 쓰기를 하지 않는다.

## 11. 감독 리뷰 반영 (정정, 2026-10-31)

**1. scheduler trust 결함 (실제 차단 결함, 수정)**
- 처음 문서는 `aws:SourceArn`을 일정 이름 접두(`schedule/default/trid-aws-eval-006-*`, ArnLike)로 제한했다.
  - AWS 공식 문서([Confused deputy prevention in EventBridge Scheduler](https://docs.aws.amazon.com/scheduler/latest/UserGuide/cross-service-confused-deputy-prevention.html))는 "Do not scope the `aws:SourceArn` statement to a specific schedule or a schedule name prefix. The ARN you specify must be a schedule group."라고 적는다.
  - 그대로 두었으면 Scheduler가 역할을 맡지 못해 **독립 종료가 동작하지 않았을 것**이다.
- **수정:** `aws:SourceArn` = `arn:aws:scheduler:ap-northeast-2:148761639313:schedule-group/default`(StringEquals, 정확 일치), `aws:SourceAccount` 유지. 공식 예시와 같은 형태다.
  - 일정은 지금처럼 default 그룹에 만든다.
  - operator의 `scheduler:CreateSchedule` 이름 접두 제한과 PassRole 제한은 그대로다.
  - 전용 그룹은 만들지 않는다. 불필요한 추가를 피한다.
- **정정:** Access Analyzer `validate-policy`의 finding 0은 문법 검사일 뿐이다. 서비스가 실제로 보내는 SourceArn과 일치한다는 증거가 아니다.
  - 수정본도 finding 0이지만, **실제 AssumeRole 성공은 신원 승인 뒤 watchdog-test(7절 2)로만 확인한다.**

**2. 비공개 문항 전달 경로의 로그 노출 (수정)**
- 처음 계획은 SendCommand 매개변수로 세트를 보내는 것이었다. 평문이나 base64면 SSM 명령 기록(ListCommands·GetCommandInvocation, 계정 관리자가 볼 수 있음)과 CloudTrail `SendCommand` 이벤트에 **원문이 남는다.**
- **수정:** 인스턴스가 부팅 때 RAM에만 개인키를 둔 키쌍을 만들고, 세트는 그 인증서로 암호화해서만 보낸다(5절 7, `c_deliver_set.py`).
- **이제 AWS에 남는 것**

  | 위치 | 남는 내용 |
  |---|---|
  | SSM 명령 기록, CloudTrail | 암호문 base64 조각과 명령문 |
  | S3 `input-cert.pem` | 공개 인증서 |
  | 인스턴스 SSM agent 로그 | 명령문(암호문). 인스턴스 종료 때 볼륨과 함께 삭제 |

- **평문이 있는 곳**
  - C의 머신
  - 실행 중 인스턴스: RAM의 세트, runner 프로세스, 암호화 전 `out/`의 `sealed/turns.jsonl`. EBS는 암호화돼 있고 종료 때 삭제된다.
  - 인스턴스 root·SSM 세션 권한자는 실행 중 볼 수 있다. C precommit 3절이 인정한 한계와 같다.
- **검증 (로컬, 공개 smoke 세트와 일회용 키):**
  - C 쪽 smoke 세트 sha가 인스턴스 기대 sha와 같다.
  - 암호화 → 24,000자 조각 → 재조립 → 복호화 → sha 확인 성공. 명령문에 평문 없음.
  - 200 KB 세트라면 SendCommand 13회다.
- **C 계약 합의 필요:** C는 이 전달 방식(인스턴스 키 암호화, 조각 SendCommand, S3 미사용)과 남는 로그 범위를 확인하고 합의 여부를 회신해 달라. 합의 전에는 full을 하지 않는다.
  - 비공개 문항 원문은 어떤 경로로도 A에게 보내지 않는다. A는 `c_deliver_set.py`를 비공개 파일로 실행하지 않는다.

**3. 시간 상한 (수정, 5절)**
- `--max-minutes` 범위를 검사한다. 확인: 0, 61(smoke), 151·−5(full)를 모두 거부했다.
- GPU 예약 장부로 이번 승인 전체를 230분까지만 허용한다. 오프라인 확인: 70 + 160 = 230분 예약 뒤 추가 smoke 예약은 거부됐고, 기동 실패 시도는 0분으로 기록이 남았다.
- 기동과 일정 생성 사이의 비원자성 한계를 명시했다.

**4. 1회 bootstrap 범위: 변동 없음**
- 사용자 승인이 필요한 범위는 그대로다.
  - IAM 사용자 2개: operator, c-runner. access key 없이 `aws login`
  - 역할 2개와 인스턴스 프로필 1개: instance, scheduler
  - 인바운드 0 SG 1개
- 바뀐 것은 정책 내용 2곳뿐이다.
  - scheduler trust의 SourceArn
  - c-runner가 `runs/*/input-cert.pem`을 읽을 수 있게 함
- 정책 6개 모두 Access Analyzer finding 0, 시뮬레이션 22건 모두 기대와 같다(추가 3건: C의 전달 인증서 읽기 허용, code와 corpus 읽기 거부).
- **AWS 쓰기는 여전히 0건이다**(IAM, SG, EC2, Scheduler, S3 `eval/aws-eval-006/` 모두 없음을 조회로 확인).
- **추가 (13:15 KST): bootstrap 실행 완료.**
  - 사용자 승인 후 실행했다. inline 사용자 정책 2,048 B 한도 때문에 두 사용자 정책은 같은 내용의 고객 관리형 정책(`policy/trid/<user>`)으로 붙였다.
  - 생성물의 읽기 검증과 로그인 절차는 `AWS-EVAL-006-IDENTITY.md`에 있다.
  - 유료 자원은 여전히 0이다.

SUPERVISION_REPORT_READY AWS-EVAL-006-INFRA
