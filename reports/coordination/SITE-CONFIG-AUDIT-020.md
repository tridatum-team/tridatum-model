# SITE-CONFIG-AUDIT-020 — 018 수동 진단 성공과 019 launchd 실패의 구성 차이 조사

**상태: 원인 후보를 로그 근거로 특정했다.**
- 019 실패 구간(19:03:24–19:03:54)의 wired +0.7GB는 **019 절차 자체가 일으킨 디스플레이 깨움과 Aerial 배경화면 동영상 재생**과 시각이 정확히 겹친다.
  - 019 절차: 새 LaunchAgent 등록 → 백그라운드 항목 알림 → 알림 디스플레이 깨움 → Aerial 배경화면 동영상 재생.
- 018에는 이 사건이 없었다.
- 서비스 구성(코드, 모델, 바이너리, 버전 항목) 자체의 동작 차이는 찾지 못했다.
- 수정 검증 실행은 하지 않았다(총괄 검토 대기). 서비스 중지, 기준 완화, 재시험은 없다. 맥북 서비스는 그대로다.

## 1. 018 vs 019 실제 구성 비교

| 항목 | 018 (수동 진단, 성공) | 019 (launchd, 실패) | 동작 영향 |
|---|---|---|---|
| 실행 명령 | SSH zsh에서 env export 후 `share_demo_control.py start --hours 1` → `supervise --hours 1`(Popen, 새 세션) | launchd `gui/501/co.tridatum.trid.site`가 `supervise --hours 168` 직접 실행(WorkingDirectory=repo) | 같은 `supervise` 코드 경로. 차이 없음(코드 확인) |
| **LaunchAgent 등록** | 없음 | `~/Library/LaunchAgents`에 새 plist를 넣고 `launchctl bootstrap` | **있음 → 아래 §2 사건 연쇄** |
| 환경 전달 | env export → start → supervise → 백엔드 → llama 상속. **llama-server에서 `ps eww`로 확인**(`LLAMA_ARG_DEVICE=none LLAMA_ARG_N_GPU_LAYERS=0`), vmmap IOAccelerator 64K | plist `EnvironmentVariables` → supervise → 백엔드 → llama 상속(코드상 `env=` 지정 없음). **실측 없음** | 아래 §3 |
| 버전 파일 | `~/Services/trid-site/versions-stage1.json` | `repo/deploy/macmini-site/versions.json` | 두 항목 같은 ID. 다른 키는 기본 항목 `version` 설명 문자열 하나뿐(서빙 장치 문구). 동작 필드(`llama_server`, `model_file`, `model_sha256`, `context`, `port`, `threads`, 독해기 `run`)는 같다 |
| 모델·색인 경로 | `repo/runs/...`, `runs/workspace-index/v5`, `inst-reb-002`, `data/search-corpus-v4` | 같음 | 없음 |
| llama-server 바이너리 | `runs/tools/llama.cpp-8ca4060-metal/llama-server` sha256 `bf5eeac5…` | 같은 파일(맥북 원본과도 같음) | 없음 |
| 모델 파일 | `061b54da…`(시작 시 코드가 sha 검사) | 같음 | 없음 |
| 백엔드 포트 / 런타임 디렉터리 | 51973 / `.runtime-diag`(새로 만듦) | 51873 / `.runtime`(이전 로그 있음) | 없음(포트 충돌 없음, 런타임은 로그·상태만) |
| 로딩 순서·동시성 | State(llama 기동, 독해기) → Workspace v5 → 임베더 → inst | 같은 코드 | 없음 |
| 프로세스 중복 | 없음 | launchd 로그상 우리 라벨 spawn 1회(python 3423), 백엔드 3444, `stop_leftover` [] | 없음 |
| 질의 | ready 직후 기관 질문 1건 | 0건(안정 조건 미충족) | 019가 더 가벼운 부하였다 |
| 배경 부하 | 기존 고장 작업 `com.cloudflare.cloudflared`, `com.fileserver`의 5–10초 주기 재시작 반복(두 창 모두 관찰) | 같음 | 공통 |

## 2. wired 증가의 출처 (통합 로그, 읽기 전용)

| 시각 | 사건 (`log show` 원문 요약) |
|---|---|
| 19:03:12.236 | launchd `gui/501/co.tridatum.trid.site` spawn (python 3423) |
| 19:03:12.489 | `com.apple.backgroundtaskmanagement.agent` spawn (새 백그라운드 항목 처리) |
| 19:03:20.677 | kernel memorystatus: `knowledgeconstructiond` per-process-limit 종료(메모리 압박 징후) |
| 19:03:23.463 | NotificationCenter: `com.apple.BTMNotificationAgent` 알림 추가 |
| 19:03:23.719 | powerd: **"Turning on display for notification wake"** |
| 19:03:24.036 | `WallpaperAerialsExtension`(pid 774): **"Play Without Ramp Action: Switch to .playing"**(플레이어 2개) |
| 19:03:24.165 | `VTDecoderXPCService`(3623) spawn, 요청자 pid 774. 16:00–19:30 사이 유일한 디코더 기동 |
| 19:03:24 | 우리 샘플: **압력 2 시작**, wired 2.82 → 다음 초 3.46GB |
| 19:03:53–54 | Aerial: "Pause Without Ramp → .pendingStillWithoutRamp", URLAsset 해제 |
| 19:03:54 | 우리 샘플: wired 2.72GB(정상 복귀, 우리 bootout 19:03:55보다 앞섬) |

- 018 창(18:45:00–18:47:30)에는 디스플레이 깨움, BTM 알림, Aerial 재생이 0건이었다.
- 동영상 디코딩과 합성은 GPU/IOSurface 버퍼를 쓰며 wired로 잡힌다 [일반 사실, 이 기기에서 버퍼 크기는 직접 측정하지 않음].
- 시각 일치는 정확하다. 시작은 같은 초, 종료는 wired 복귀와 같은 초다.
- **결론(근거 수준):** 019의 30초 경고는 TRID 적재 여유가 빠듯한 순간에 이 외부 GPU·디스플레이 부하가 겹친 결과로 보인다. 019 등록 절차가 그 부하를 유발했다.
- **한계:** 경고 지속에 대한 기여도를 정량 분리하지는 않았다. TRID 적재만으로도 018에서 5초 경고가 났다.

## 3. CPU 적용 여부 (019)

- **확정 불가.** 019에서는 llama-server의 `ps eww`와 vmmap을 재지 않았다.
- **코드상 상속 경로**
  - launchd `EnvironmentVariables` → `share_demo_control.py supervise`의 `os.environ` → 백엔드 `subprocess.Popen`(env 지정 없음, `share_demo_control.py:165`) → llama-server `subprocess.Popen`(env 지정 없음, `shared_demo.py` 9256df4 1624–1627행).
  - llama.cpp는 CLI 인자가 env보다 우선하지만, 배포 코드는 `-ngl`이나 `-dev`를 넘기지 않는다. 그래서 env가 적용되는 구조다.
- **간접 근거:**
  - llama 모델 적재(19:03:13–14, 로그 `model loaded` +1.46초) 뒤 t=2–10초 동안 wired는 2.76–2.78GB로 그대로였다.
  - Metal 적재라면 SITE-MIGRATION-013처럼 적재 직후 wired가 약 +2.2GB 올랐을 것이다.
  - 이것은 CPU 적용과 맞지만 실측 확인은 아니다.

## 4. 수정 제안 (실행 안 함, 총괄 승인 필요)

시스템 설정, 임계값, 정밀도, 프롬프트, 검색 조건은 바꾸지 않는 범위다.

1. **등록과 기동을 분리한다.** 알림과 디스플레이 깨움이 서비스 적재와 겹치지 않게 하려는 것이다.
   1. `launchctl disable gui/501/co.tridatum.trid.site`
   2. plist 배치와 `bootstrap`(등록만, 비활성이라 실행 안 됨)
   3. BTM 알림, 디스플레이 깨움, Aerial 재생이 끝나고 wired와 압력 1이 60초 이상 안정될 때까지 대기
   4. `launchctl enable` + `kickstart`
2. **기동 직후 실측을 기록한다.** llama-server `ps eww` 환경, vmmap IOAccelerator와 GGUF 매핑, 1초 표본에 wired 출처 판단용 `WallpaperAerialsExtension`·`VTDecoderXPCService` 활성 여부를 남긴다.
3. 같은 019 가드(준비 경고 30초, 심각 4, 기존 서비스 기준, 공개 전 압력 1 안정 30초)로 1회 관찰한다. 공개 전환은 하지 않는다.

**운영 위험(사용자 결정 사항):**
- 맥미니는 알림이 올 때마다 디스플레이를 깨우고 Aerial 동영상 배경화면을 재생할 수 있다. 운영 중에도 같은 wired 증가가 무작위로 겹칠 수 있다.
- 정지 배경화면이나 알림 디스플레이 깨움 끄기는 맥미니 시스템 설정 변경이라 A가 하지 않았다.

## 5. 누락된 측정

- 019 llama-server의 env와 vmmap(§3).
- Aerial 재생 중 wired 증가분의 프로세스별 귀속. WindowServer, VTDecoderXPCService, WallpaperAerialsExtension의 IOAccelerator 크기는 지금 재생 중이 아니라 사후 측정이 불가능하다.

## 6. 현재 상태

- 맥미니 TRID 프로세스 0. `~/Library/LaunchAgents/co.tridatum.trid.site.plist` 없음(019 정리). 압력 1, wired 2.73GB.
- 맥북 원점(launchd 두 작업) 유지. 공개 전환 없음.
- SITE-MIGRATION-019 보고의 "같은 구성", "wired 출처 미확인" 문장은 정정 표시로 고쳤다.

SUPERVISION_REPORT_READY SITE-CONFIG-AUDIT-020
