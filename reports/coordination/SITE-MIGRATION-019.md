# SITE-MIGRATION-019 — 기본 모델 중심 맥미니 이관 (사용자 승인)

**상태: 실패. 1단계(준비·관찰)에서 정지했다.**
- 맥미니 운영 launchd 서비스는 ready(13초)까지 갔다.
- 그러나 **압력 2가 30초 연속**(19:03:24–19:03:55)으로 준비 한도에 걸려 가드가 `launchctl bootout`했다.
- 공개 질문, 터널 전환, 맥북 정지, 외부 검증은 하지 않았다.
- **원점은 맥북 그대로다**(롤백할 변경 없음). 다른 서비스를 추가로 멈추지 않았다.

## 1. 준비한 것 (운영 파일; 이관 성공 표시 아님)

**`deploy/macmini-site/versions.json`** (sha `dcfa42fd…`, 맥미니 사본과 일치)
- 배포 `deploy/shared-demo/versions.json`의 기본 `trid-assistant-qwen3-1.7b`(default, 원본 표기, Apache-2.0)와 `trid-v0.1-reader` 두 항목만 담았다.
- 다른 필드는 그대로다. 기본 항목의 `version` 설명에서 서빙 장치만 "llama.cpp 8ca4060 빌드를 맥미니 CPU 전용으로 실행(LLAMA_ARG_DEVICE=none, GPU offload 0)"으로 바꿨다.
- 회귀 실험판 2개(pilot-sft-001, model-repair-002)는 이 파일에 없다. GGUF는 맥북에 그대로 보존했고 추가 복사는 하지 않았다.
- **삭제된 ID 처리:** 배포 코드(`9256df4`)는 목록에 없는 버전을 `DocumentError('목록에 있는 모델 버전을 선택해 주세요.')`로 거부한다(1896–1897행). 원본으로 몰래 연결하는 alias는 없다. 실측 확인은 질의 단계 전에 정지해 하지 못했다.

**`deploy/macmini-site/co.tridatum.trid.site.plist`**
- `TRID_DEMO_VERSIONS`를 위 파일로 지정했다.
- `LLAMA_ARG_DEVICE=none`, `LLAMA_ARG_N_GPU_LAYERS=0`을 넣었다.
- 나머지(`supervise --hours 168`, KeepAlive, `TRID_DEMO_TUNNEL=0`, 격리 HF 캐시)는 그대로다.

**맥미니 조건 확인**
- 비밀키 `.runtime/proxy-secret`과 터널 자격 증명은 모두 0600이다.
- 터널 config는 추적 사본과 같은 내용이다.
- `autoLoginUser=eomgyuhyeon`: 재부팅하면 GUI 자동 로그인 후 LaunchAgent가 뜬다. 재부팅 시험은 하지 않았다(SFT controller 위험).
- `pmset sleep 0`, `autorestart 1`.

## 2. 1단계 관찰 시험 (19:03, 실제 운영 launchd 작업)

**사전 확인 (19:03:11)**
- 압력 1, free 61%. 전송 없음(맥북·맥미니). trid-site와 bohun 프로세스 없음. Docker VM 정지.
- 기존 서비스 기준: 8000/8501/8502=200, 18765/8766=403.

**실행**
- plist를 `~/Library/LaunchAgents`에 복사하고 `launchctl bootstrap`했다(포트 51873, 터널 커넥터 없음 = 공개 트래픽 없음).
- 가드:
  - 준비 경고는 30초 한도, 심각(4) 즉시 정지, 기존 서비스 기준 이탈 즉시 정지.
  - 공개 질문은 ready 후 압력 1이 30초 안정된 뒤에만 시작하게 했다.

**사건 기록** (원문: `SITE-MIGRATION-019/events.txt`)

| 시각 | 사건 |
|---|---|
| 19:03:12 | bootstrap rc=0 |
| 19:03:24 | 경고 시작 (p=2) |
| 19:03:25 | 서비스 ready (healthz 200, t=13s) |
| 19:03:55 | **경고 30초 지속 → STOP, `launchctl bootout` rc=0** |

- 표본 41개 중 압력 1은 11개, 2는 30개. 심각(4) 0회.
- 질문은 0건이다. 안정 30초 조건을 채우지 못해 시나리오가 시작되지 않았다.

**메모리 시간표** (원문: `samples.txt`, 단위 GB)

| t | 압력 | free | active | inactive | wired | 압축기 | TRID RSS |
|---|---|---|---|---|---|---|---|
| 0s | 1 | 1.45 | 4.06 | 3.41 | 2.75 | 3.17 | 0.00 |
| 6s | 1 | 0.05 | 4.09 | 3.99 | 2.76 | 4.50 | 5.81 |
| 10s | 1 | 0.06 | 3.53 | 3.39 | 2.76 | 5.61 | 4.96 |
| 12s | 2 | 0.06 | 3.06 | 2.81 | 2.82 | 6.48 | 3.72 |
| 13–37s | 2 | 0.06–0.14 | 2.44–2.69 | 2.21–2.62 | **3.38–3.62** | 6.63–6.91 | 2.90 → 0.73 |
| 42s | 2 | 0.56 | 2.87 | 2.50 | 2.72 | 6.70 | 0.73 |

**정지 후 (19:04:37)**
- 작업이 언로드됐다("Could not find service"). trid-site 프로세스 0. 압력 1 ×5. 기존 서비스 전부 기준과 같다.
- **맥미니 `~/Library/LaunchAgents/co.tridatum.trid.site.plist`를 지웠다.** 다음 로그인 때 자동으로 떠서 메모리를 누르지 않게 하려는 것이다. 원본은 `~/Services/trid-site/repo/deploy/macmini-site/`에 남아 있다.

## 3. 해석과 한계

- **SITE-PRESSURE-DIAG-018과 같은 구성인데 결과가 달랐다.**
  - 018에서는 기동 경고가 5초로 끝났고, 이번에는 30초 이상 이어졌다.
  - 이번에는 경고 중 **wired가 2.76 → 3.4–3.6GB로 늘었다.** 018에서는 2.79로 일정했다.
  - TRID RSS는 0.73GB까지 압축되거나 밀려났는데도 압력 2가 계속됐다.
- **wired 증가의 출처는 확인하지 못했다.**
  - 이번 launchd 실행에서는 llama-server 환경 변수와 vmmap을 기록하지 않았다. llama 로그(기본 상세도, 11줄)에는 Metal/GPU 줄이 없다.
  - 5분 주기 작업(`com.crypto.monitor` 19:01:13, `com.crypto.near-momentum` 19:02:02의 마지막 로그)과 시각이 정확히 겹치지는 않는다.
  - 원인은 미확정이다.
- **결론:** 같은 조건에서도 실행마다 결과가 갈린다. 이번 실행은 사용자 승인 조건(준비 경고 30초 한도, 공개 전 압력 1 안정 30초)을 채우지 못했다.
- 승인 범위 안에서 다른 서비스 추가 정지, 기준 완화, 재시도는 하지 않았다.

## 4. 하지 않은 것

- 15분 연속 질의, 기관 전환, 문서 첨부, 독해기 선택, 새 대화 기능 관찰. 시나리오는 준비했으나 안정 조건 미충족으로 시작하지 않았다.
- 맥북 터널 bootout, 맥미니 커넥터 기동, 맥북 정지, 외부 브라우저 검증. 총괄에 알릴 외부 검증 시점은 없다.
- 맥북 옛 TRID launchd 작업 disable과 보관. 맥북이 계속 원점이므로 그대로 둔다.
- `docs/SHARING.md`·README의 서빙 장치를 CPU·맥미니로 바꾸는 일. 이관이 성공하지 않아 하지 않았다.

## 5. 현재 상태

| 대상 | 상태 |
|---|---|
| test.tridatum.co | 맥북 원점(launchd `co.tridatum.trid.site` + `co.tridatum.trid.site.tunnel`, 고정 named tunnel), 변경 없음 |
| 맥미니 TRID | 프로세스 없음, LaunchAgent plist 없음. 자산·운영 파일은 `~/Services/trid-site`에 준비된 상태 |
| 맥미니 다른 서비스 | 기준 그대로. bohun-test 꺼짐(재로그인 시 RunAtLoad로 다시 뜸), Docker VM 정지, ARDAgent 재시작 상태 |

SUPERVISION_REPORT_READY SITE-MIGRATION-019
