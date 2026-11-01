# MINI-SERVICE-MAP-011: 맥미니 서비스 용도·의존관계 지도 (A 지원, 읽기 전용)

- **점검 시각:** 2026-11-01 15:40~16:30 KST.
- **방법:** SSH(키 고정, StrictHostKeyChecking=yes)로 읽기만 했다.
  - 대상: launchd plist, 프로세스, 포트, cloudflared 설정과 metrics, compose 파일, 로그 mtime과 접근 줄 수.
  - 로컬 HTTP `GET /`, ollama `/api/ps`·`/api/tags`, Docker 소켓 `GET` 1회.
- **하지 않은 것:** 삭제·kill·재시작·권한 상향·설정 변경 0. Docker CLI 재시도 0.
- **비밀값:** 비밀 env 값·토큰·개인 문서는 보고서에 옮기지 않았다.
  - dufs plist 인자에 계정 정보가 평문으로 있다. 값은 적지 않았다.
- **보존:** DATA-REB-009는 커밋(`4571562`)된 상태 그대로 대기 중이다.
- **판정 기준:** "사용 중/미사용"은 단정하지 않았다. 근거마다 강도를 적었고, 추정은 [추정]으로 표시했다.

## 1. A 즉시 인계: 후보와 불확실성

| 순위 | 후보 | 근거(관찰) | 멈추면 끊기는 것 | 불확실성 |
|---|---|---|---|---|
| 1 | `com.cloudflare.cloudflared` | 인자 없이 실행돼 바로 실패한다. launchd runs **679,632**, last exit 1. 에러 로그 335MB가 지금도 자란다 | 없음. 정상으로 돈 적이 없다(같은 터널 `2923ac78`은 `com.cloudflare.tunnel.dataprep`이 실행 중) | 낮음 |
| 2 | `com.fileserver` | :9999 bind 실패("Address already in use", dufs가 점유). runs **195,058**, 로그 1.19GB | 없음. 정상으로 돈 적이 없다. files.eommydocs.com은 dufs가 처리한다 | 낮음 |
| 3 | Docker Desktop: VM footprint 8.1G, CPU 약 320%, backend 3.5G | 엔진 소켓 `_ping`·`/containers/json`이 5~8초 안에 응답하지 않는다. 포워딩 포트 11개 모두 HTTP timeout | 지금 응답하는 Docker 서비스는 없다(아래 2절). compose 서비스를 쓰는 사람이 있으면 그 사람 | 중간. 엔진이 멈춰 컨테이너 목록·상태를 보지 못했다. 설정 파일은 macOS 권한(TCC)으로 못 읽었다 |
| 4 | Kafka (`homebrew.mxcl.kafka`, heap 1G) | 토픽은 `test-0` 하나뿐이다. 설정·코드에서 9092를 참조하는 클라이언트를 찾지 못했다. ESTABLISHED는 자기 컨트롤러(9093)뿐이다. launchd runs 2,382 | 찾은 의존 없음 | 중간. 다른 기기(LAN)가 9092에 붙는지는 모른다 |
| 5 | MySQL (데이터 320M) | 참조는 `.openclaw/workspace`의 적재 스크립트 3개와 presenton(실행 중 아님)뿐이다. 3306은 127.0.0.1에만 바인딩. 현재 연결 0 | 그 스크립트를 다시 돌릴 때 | 중간 |
| 6 | ollama :11434 (`homebrew.mxcl.ollama`) | 로드된 모델 0. 마지막 추론 계열 POST는 10-31 02:31. 11-01 요청은 B의 점검 GET뿐 | bohun·bohun-test의 LLM 호출. 기본값은 `FABRIX_ENDPOINT localhost:11434`, `exaone3.5:7.8b`, `start.sh` 기준 | 높음. 두 서비스가 쓸 때만 로드된다 |
| 7 | bohun 운영 :8000 + caddy :8010 | 접근 로그(stdout)의 마지막 쓰기가 07-25다. 프로세스는 09-22 시작. caddy :8010은 401(인증). postgres `bohun` numbackends 0 | :8010으로 붙는 사용자(LAN 도달 가능) | 높음. 이 프로세스의 접근 로그가 어디에 쓰이는지 확인하지 못했다 |
| 8 | crypto·scanner 계열 launchd 11개(`com.crypto.*` 9, `com.scanner.*` 2) | `surge-alert`·`market-monitor`는 KeepAlive인데 로그가 10-17 이후 멈췄다. `orderbook-capture`는 10-29. `momentum-alert`는 16:20에 실행 중 | 개인 알림(Slack 등) [추정] | 높음. trid와 관계없으며 소유자 결정 사항 |

**메모리 큰 항목** (top footprint, 압축·스왑분 포함)
- ARDAgent 42G: 4절 참고.
- Docker VM 8.1G, bohun-test 3.9G(in-process CPU bge-m3 [추정: `EMBED_DEVICE=cpu`]), Docker backend 3.5G, tridatum-rag 1.6G, loginwindow 1.35G, Kafka 1.2G.

**중복**
- bge-m3가 두 곳에 상주한다: rag용 ollama :11435(0.67GB, keep-alive 무기한)와 bohun-test 프로세스 안.
- trid 사이트를 이관하면 세 번째가 된다 [추정].
- ollama가 3개 떠 있지만 모델 저장소(`~/.ollama/models`, 13G)는 하나를 함께 쓴다.

**trid 사이트 이관 흔적**
- `~/Services/trid-site`가 생겼다(13:43~14:25, `repo`·`hf`·`venv`·`tunnel`·`versions-stage1.json`).
- A의 작업으로 보이며, B는 건드리지 않았다.

## 2. 지도: 공개 URL/포트 → 프로세스 → 모델/컨테이너 → DB

| 공개 경로 | 터널/프록시 | 포트 → 프로세스(launchd) | 모델 | DB·의존 | 사용 근거 |
|---|---|---|---|---|---|
| test.eommydocs.com | `com.cloudflare.tunnel.dataprep`(`2923ac78`) | :8001 bohun-ai-test uvicorn (`com.eommydocs.bohun-test`) | ollama :11434 `exaone3.5:7.8b`(기본값), bge-m3 in-process | postgres `bohun` | out.log 11-01 14:26. 터널 요청 1,269(이 터널 전체) |
| dataprep.eommydocs.com | 같은 터널 | :8501 data-prep-v2 `run.py` (`com.eommydocs.dataprep`) | 확인 안 함 | 확인 안 함 | stdout 11-01 14:57 |
| datasearch.eommydocs.com | 같은 터널 | :8502 streamlit (`com.eommydocs.datasearch`) | 확인 안 함 | 확인 안 함 | 로그 09-22 이후 없음(streamlit은 접근 로그를 남기지 않는다 [추정]) |
| files.eommydocs.com | 같은 터널 | :9999 dufs (`com.dufs.filebrowser`) | — | 파일 `apps/filebrowser/uploads` | dufs.log 최근 2일 줄 578 |
| rag.tridatum.co | Worker → VPC 서비스 → 터널 `3b900b28`(`co.tridatum.rag.tunnel`, 로컬 ingress는 404뿐) | :8766 tridatum-rag `server.py` (`co.tridatum.rag.app`) | 답변: OpenAI `gpt-5.4-mini`(env). 임베딩: ollama :11435 bge-m3 | `data/corpus.sqlite` 347MB | 터널 요청 868, 오류 0 |
| (정기 작업) | — | `co.tridatum.rag.refresh`(6시간마다, last exit 1), `co.tridatum.dashboard.refresh`(하루마다) | ollama :11435 | corpus.sqlite | rag.refresh 마지막 실행 실패 |
| seocho.tridatum.co → secho-api.eommydocs.com | `co.tridatum.secho.tunnel`(`ac31ac36`) | :18765 secho `server.py` ir-v0.8 (`co.tridatum.secho.api`, sandbox-exec 오프라인) | ollama :11436 `qwen2.5:7b`(약 4.7GB, 요청 때 로드. 지금은 미로드) | — | 터널 요청 134. `/api/chat` 10-31 13:30 |
| 공개 경로 없음(LAN) | caddy :8010(401 인증) | :8000 bohun-ai uvicorn (`com.eommydocs.bohun`) | ollama :11434 `exaone3.5:7.8b`(기본값) | postgres `bohun` | 1절 7번 |
| 공개 경로 없음(LAN) | — | :18789 openclaw gateway (`ai.openclaw.gateway`) | 확인 안 함 | 확인 안 함 | gateway.log 10-30 |
| 공개 경로 없음 | — | Docker 포워딩 :3000 open-webui, :3001 grafana, :3002 uptime-kuma, :5678 n8n, :6379 redis, :8123/:9001 clickhouse, :8888 jupyter, :9000 portainer, :9090 prometheus (`~/macmini-server/docker-compose.yml` 포트 대응). :8081·:9100·:30080·127.0.0.1:6443은 compose에 없음 [추정: 6443·30080은 Docker Desktop Kubernetes] | open-webui → ollama [추정] | compose 볼륨 | 11개 모두 HTTP timeout |
| — | — | Kafka :9092/9093, MySQL :3306/33060, Postgres :5432(homebrew 17), syncthing :22000 | — | — | 1절 4·5번. postgres DB는 `bohun` 1개 |

- **포트 5000/7000:** AirPlay 수신기(ControlCenter, AirTunes)다. 서비스가 아니다.
- **점검이 남긴 요청:** B의 `GET /` 점검이 bohun·bohun-test 로그에 한 줄씩 남았을 수 있다.

## 3. 멈추면 끊기는 서비스

| 멈추는 대상 | 끊기는 것 |
|---|---|
| `com.cloudflare.tunnel.dataprep` | test·dataprep·datasearch·files.eommydocs.com 4개 모두 |
| postgres@17 | bohun 운영과 bohun-test(둘 다 `bohun` DB) |
| ollama :11434 | bohun·bohun-test의 LLM 답변. rag·secho와 무관 |
| ollama :11435 | rag.tridatum.co 검색 임베딩, rag·dashboard refresh |
| ollama :11436 | seocho 데모 답변 |
| `co.tridatum.rag.*` | rag.tridatum.co |
| `co.tridatum.secho.*` | seocho.tridatum.co |
| Docker Desktop | compose 서비스 10개와 Kubernetes. 지금 모두 응답하지 않는다 |
| Kafka / MySQL | 찾은 의존 없음(1절) |
| `com.cloudflare.cloudflared`, `com.fileserver` | 없음(정상으로 돈 적이 없다) |
| `co.tridatum.trid.sft-controller` | 지금은 멈춰 있다 |

- **sft-controller 위험:** RunAtLoad=true, `KeepAlive {SuccessfulExit: false}`이고 AWS env(`AWS_PROFILE` 등)가 있다. 재부팅·재로그인하면 다시 뜬다. 유료 작업을 다시 시작하는지는 확인하지 않았다.

## 4. ARDAgent와 원격 제어·SSH

**관찰**

| 구성 | 실행 주체 | 포트 | 상태 |
|---|---|---|---|
| ARDAgent | 사용자 에이전트(`/System/Library/LaunchAgents/com.apple.RemoteDesktop.plist`), PID 60985 | :3283(Apple Remote Desktop) | 가동 21일. footprint 42G, RSS 4MB |
| screensharingd | root 데몬(`/System/Library/LaunchDaemons/com.apple.screensharing.plist`), PID 8022 | :5900 | 가동 30일. pmset 기준 sleep을 막고 있다 |
| ScreensharingAgent, SSMenuAgent | 사용자 에이전트 | — | 별도 프로세스 |
| sshd | `/System/Library/LaunchDaemons/ssh.plist` | :22 | 세션은 launchd(ppid 1)의 `sshd-session`으로 뜬다 |

- 점검 당시 ESTABLISHED 연결은 B의 SSH 하나뿐이었다. 화면 공유·ARD 연결은 0이었다.
- **SSH는 ARDAgent와 별개다.** 데몬, 소유자, 포트가 다르고 부모 프로세스 관계도 없다(관찰).
- **ARDAgent를 재시작하면:**
  - Apple Remote Desktop 앱의 관찰·제어·관리 명령(:3283)이 잠깐 끊기거나 다시 연결해야 한다 [추정].
  - 표준 화면 공유(:5900)는 screensharingd가 따로 처리하므로 계속될 가능성이 높다 [추정]. 다만 "원격 관리" 모드의 화면 공유는 ARD 구성요소를 거칠 수 있어, 그 세션은 끊길 수 있다 [추정].
  - SSH 세션은 영향받지 않는다 [추정: 프로세스가 분리돼 있다는 관찰에 근거].
- **42G의 의미:** footprint는 크지만 RSS는 4MB다. 대부분 스왑·압축에 있다. 재시작하면 스왑 21.9G 중 상당 부분이 풀릴 수 있다 [추정]. `footprint` 상세는 권한 문제로 출력되지 않았다.

## 5. 보안 관찰 (조치 아님)

- **방화벽 꺼짐:** macOS 애플리케이션 방화벽이 꺼져 있다(State 0, stealth off). en0이 공인 IP 211.48.84.156이다.
- **맥북에서 열려 있던 포트:** 8010, 8501, 8502, 9999, 5432, 33060, 9092, 18789, 3283.
  - 맥북도 211.48.x.x 대역이다. 같은 망 안에서 본 결과일 수 있고, 인터넷 노출 여부는 확인하지 않았다 [추정].
- **dufs 계정 정보 평문:** 계정 정보가 launchd plist 인자에 평문으로 있다. files.eommydocs.com으로 공개돼 있다.
  - 이 값은 B의 점검 도구 출력에 한 번 나왔다. 보고서에는 옮기지 않았다.
  - 값 교체가 필요한지는 소유자가 판단할 사항이다.

## 6. 읽지 못한 것

- **Docker:**
  - 컨테이너 목록·상태·메모리: 엔진 API가 timeout이었다(CLI·소켓 모두).
  - Docker Desktop 설정(VM 메모리·CPU, Kubernetes 여부): TCC 권한 거부.
- **접근 로그 위치:** data-prep·datasearch·openclaw의 실제 사용자 수와 bohun 운영 접근 로그 위치를 확인하지 못했다.
- **sft-controller:** 재시작하면 유료 작업을 다시 시작하는지 확인하지 않았다.

SUPERVISION_REPORT_READY MINI-SERVICE-MAP-011
