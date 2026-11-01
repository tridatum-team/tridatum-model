# SITE-MINI-AUDIT-010: test.tridatum.co origin과 맥미니(16GB) 읽기 전용 점검

- **점검 시각:** 2026-11-01 13:37~13:50 KST.
- **방법:** SSH `eomgyuhyeon@211.48.84.156`, 키 `~/.ssh/trid-macmini-ed25519`, `StrictHostKeyChecking=yes`(known_hosts 일치), BatchMode.
- **하지 않은 것:**
  - 프로세스 중지·재시작, 설정 변경, 모델 로드, 부하 검사 0. 유료 AWS 0.
  - 비밀값 출력 0. 터널 자격증명·토큰·키 파일은 존재만 확인했다.
  - co.tridatum.secho·rag·Docker·Java는 보기만 했다.
- **B 데이터 작업:** DATA-REB-009는 지시를 받은 상태로 멈춰 두었다. 이 점검 뒤 재개한다.

## 1. 결론 (A 직접 전달)

1. **사이트 origin은 지금 맥북이다. 맥미니 이관은 되지 않았다.**
   - 경로: `test.tridatum.co` → Cloudflare Worker `trid-model-test` → quick tunnel `pitch-award-phase-needle.trycloudflare.com` → 맥북 `127.0.0.1:51873` `scripts/shared_demo_live_9256df4.py`.
   - Worker 설정은 `deploy/shared-demo/worker.mjs`, `wrangler.json`의 `ORIGIN_URL`이다.
   - 13:4x 확인: 맥북 cloudflared metrics `/quicktunnel`의 hostname과 `wrangler.json` `ORIGIN_URL`이 같았다. `GET /`는 200(0.9초)이었다.
   - 맥북 프로세스 경과 시간이 약 2분이었다. 점검 직전에 A가 다시 띄운 것으로 보인다 [추정].
   - `HEAD /`는 404(JSON)다. 브라우저 접속과는 관계없다 [추정: Worker가 GET·POST만 처리].
2. **맥미니에는 1.7B 사이트 배포 자산이 없다.**
   - `~/trid-controller/repo`는 159MB짜리 SFT 감시용 부분 사본이다. `DEPLOYED_FROM_COMMIT e0779dd`, HEAD `2f69fe5`(10-30 "monitor completes").
   - 없는 것: `scripts/shared_demo.py`, `deploy/shared-demo/`, `runs/`(GGUF·색인·llama.cpp), `.shared-demo` 비밀 파일.
   - 홈 깊이 6 안에 100MB 넘는 `*.gguf`가 없고, 맥미니 터널 설정 어디에도 `test.tridatum.co`가 없다.
3. **16GB 맥미니에 최소 운영을 그대로 얹는 데는 실제 제약이 있다.** 1.7B 1개 + CPU bge-m3 + v5 색인을 기준으로 보면:
   - 지금도 메모리가 빠듯하다. 물리 15G 사용, 270M 여유, **스왑 21.9G/23.5G 사용**.
   - **Docker VM이 CPU 약 320%를 계속 쓰고**, 부하 평균은 6.1(10코어)이다.
   - 기존 서비스를 줄이지 않고 약 5~6GB를 더하면 스왑이 더 늘어 응답이 느려질 가능성이 크다 [추정]. 실측과 부하 검사는 A 몫이다.

## 2. 맥미니 자원

| 항목 | 값 |
|---|---|
| 하드웨어 | Apple M4, 10코어, 16GB, macOS 26.3.1, 가동 39일 |
| 메모리 | PhysMem 15G 사용(wired 2.9G, compressor 3.1G), unused 270M |
| 압박 수준 | `kern.memorystatus_vm_pressure_level` 1(정상 단계). `memory_pressure` free 61% |
| 스왑 | 23,552M 중 21,865M 사용(남은 1,687M). swapouts 418만 |
| CPU | user 34% / sys 10%. `com.apple.Virtualization`(Docker VM) 319% |
| 부하 평균 | 6.11 / 6.21 / 6.21 |
| 디스크 | Data 볼륨 460G 중 354G 사용, 59G 남음(86%) |

**메모리 상위**(top의 MEM 열은 압축·스왑분 포함 footprint)

| 프로세스 | footprint | 메모 |
|---|---|---|
| ARDAgent (원격관리) | 42G | RSS는 4MB. 대부분 스왑·압축된 것으로 보이며, 스왑 사용의 큰 몫일 수 있다 [추정: 누수] |
| Docker VM | 8.1G | |
| bohun-ai-test uvicorn :8001 | 3.9G | |
| Docker backend | 3.5G | |
| tridatum-rag server.py :8766 | 1.6G | |
| loginwindow | 1.35G | |
| Kafka(java) | 1.2G | |

- **Docker CLI 무응답:** `docker ps`가 8초 안에 답하지 않았다(B가 끝낸 것은 이 확인용 프로세스뿐). 2일 전 `docker ps`, 18일 전 `docker system df` 프로세스가 아직 남아 있다. `docker-ai docker-cli-plugin-metadata` 프로세스는 약 12시간마다 하나씩 쌓여 60개 이상이다 [추정: 플러그인 누수].

## 3. launchd·포트·터널

**tridatum 관련 launchd**

| 라벨 | PID / 마지막 상태 | 내용 |
|---|---|---|
| `co.tridatum.rag.app` | 45180 / -15 | `~/apps/tridatum-rag/server.py --port 8766`(127.0.0.1). 이전 종료가 SIGTERM이었고 지금은 실행 중 |
| `co.tridatum.rag.ollama` | 79645 | ollama :11435. llama-server 82428 모델 blob `daec91ff…` |
| `co.tridatum.rag.tunnel` | 79651 | cloudflared, 터널 `3b900b28…`. 로컬 ingress는 `http_status:404` 한 줄뿐. README는 `rag.tridatum.co → Worker → VPC 서비스 → :8766` |
| `co.tridatum.rag.refresh` | - / 1 | 마지막 실행이 실패로 끝남 |
| `co.tridatum.secho.api` | 88648 | `Services/secho-demo/releases/ir-v0.8-20261031-1323/server.py` :18765 |
| `co.tridatum.secho.ollama` | 82002 | ollama :11436 |
| `co.tridatum.secho.tunnel` | 10092 | 터널 `ac31ac36…` → `secho-api.eommydocs.com` → :18765 |
| `co.tridatum.dashboard.refresh` | - / 0 | |
| `co.tridatum.trid.sft-controller` | - / 0 | `drive_sft_campaign.py --interval 300`. RunAtLoad=true, KeepAlive=SuccessfulExit. 마지막 로그 10-30 15:03(S1-monitor, spent_usd 11.25). 지금은 멈춤 |

**그 밖에 상주하는 것:**
- 서비스: bohun(:8000, caddy :8010), bohun-test(:8001), datasearch(:8502), dataprep(:8501), dufs(:9999), openclaw gateway(:18789)
- DB·미들웨어: postgres(:5432), mysql(:3306/33060), kafka(:9092/9093)
- ollama 3개(:11434/11435/11436), syncthing, Docker 포트 13개
- crypto 스캐너 계열 launchd 다수

**Cloudflare 터널 3개**

| 터널 | 연결 |
|---|---|
| `com.cloudflare.tunnel.dataprep` `2923ac78…` | `test.eommydocs.com`→:8001, `dataprep.`→:8501, `datasearch.`→:8502, `files.`→:9999 |
| `co.tridatum.secho.tunnel` | 위 표와 같음 |
| `co.tridatum.rag.tunnel` | 위 표와 같음 |

- `com.cloudflare.cloudflared`는 마지막 상태 1(실패)이다.
- 터널 로그(10-30): QUIC 연결이 끊겼다가 다시 붙은 기록이 있다. cloudflared 2026.2.0은 구버전 경고가 나온다.

## 4. 16GB 맥미니 최소 운영에 대한 실제 제약

**필요량** (맥북 실측과 파일 크기)

| 구성 | 크기 / 사용량 |
|---|---|
| Qwen3-1.7B Q8_0 GGUF | 1,834,426,016 B. 맥북 llama-server(Metal) RSS 약 2.2GB/개 |
| 지금 사이트에 걸린 버전 | llama-server 3개(원본, pilot-sft-001-a, model-repair-002), 합계 약 6.8GB |
| shared_demo 서버 프로세스 | RSS 1.0~1.2GB |
| v5 색인 | 409MB(`documents.json` 158MB, `dense.f16.npy` 213MB, `meta.json` 13MB) |
| bge-m3 | CPU에서 float32로 올린다(`trid/service/embed.py`). 가중치는 약 2.3GB [추정: 568M 파라미터 × 4B]. HF 캐시는 4.3G |
| 맥미니 디스크 | 59G 남음. 자산 복사(GGUF 1.8G씩 + 색인 0.4G + bge-m3 4.3G)는 가능하다 |

**제약**

1. **메모리:** 1.7B 한 개 + bge-m3 + 서버·색인만 올려도 추가 5~6GB가 필요하다 [추정]. 지금 맥미니는 여유 0.3G, 스왑 남은 1.7G다. 기존 서비스를 줄이지 않으면 스왑이 크게 늘거나 메모리 압박 단계가 올라갈 위험이 높다 [추정]. 버전 3개를 동시에 올리는 지금 구성은 더 어렵다.
2. **CPU:** Docker VM이 3코어 이상을 상시 쓴다. bge-m3 질의 인코딩과 1.7B 생성이 이 코어를 두고 경쟁한다. 실제 지연은 A가 부하 검사로 확인해야 한다.
3. **경로:** 지금 Worker origin은 quick tunnel이라, 맥북 cloudflared가 재시작될 때마다 hostname이 바뀌고 Worker를 다시 배포해야 한다 [추정: quick tunnel 일반 동작]. 맥미니에는 named tunnel 3개가 있지만, 어느 것도 test.tridatum.co나 :51873으로 가지 않는다. 이관하려면 터널과 Worker `ORIGIN_URL`을 새로 연결해야 한다(A가 결정).
4. **운영 위험:**
   - `co.tridatum.trid.sft-controller`가 RunAtLoad=true다. 재부팅이나 다시 로그인하면 캠페인 드라이버가 다시 뜬다. 끝난 캠페인에서 유료 작업을 다시 시작하는지는 확인하지 않았다.
   - Docker CLI가 응답하지 않는다.
   - 둘 다 B는 손대지 않았다. 처리 여부는 감독과 A가 정한다.

## 5. A 인계

- **origin:** 지금은 맥북이다. 맥미니에는 사이트 코드·모델·색인·비밀 파일이 없다.
- **이관 전 결정할 것:**
  - 맥미니에서 끄거나 줄일 서비스. 예: Docker VM 8.1G·CPU 320%, ARDAgent 누수, ollama 3개 [추정]. 해당 서비스 소유자의 승인이 필요하다.
  - 버전 수. 1.7B 하나로 줄일지.
  - 터널을 named tunnel로 바꿀지.
- **부하 검사:** 1.7B + bge-m3 CPU 지연과 메모리 압박은 서비스 수정 담당인 A가 실측한다.

SUPERVISION_REPORT_READY SITE-MINI-AUDIT-010
