# MINI-CAPACITY-PLAN-011 — 맥미니 자원 정리 계획 (읽기 전용 측정, 실행 없음)

**상태: 계획 완료, 실행 대기(사용자 승인 필요).**
- A는 아무것도 멈추거나 재시작하지 않았다. 설정과 데이터도 바꾸지 않았다.
- 측정 시각: 2026-11-01 16:20–16:24 KST. 이 시각의 새 측정이며, 이전 관측값을 현재 값으로 쓰지 않았다.
- 맥북 사이트(고정 원점)는 그대로 운영 중이다. A의 맥북 무거운 작업(전송·테스트)은 없다(16:25 확인).
- 유료 AWS 0.

## 1. 지금 맥미니 메모리 (16:20–16:24 새 측정)

| 항목 | 값 |
|---|---|
| 물리 메모리 | 16GB, 10코어, 가동 39일(9-22 부팅), load 6.5 |
| 압력 | 1 (5초 간격 6회 모두 1), memory_pressure free 61–62% |
| 스왑 | 21.2GB 사용(변동 없음) |
| vm_stat (16KB 페이지) | free 0.29GB · active 4.9GB · inactive 4.9GB · wired 2.9GB · 압축기 점유 3.2GB(보관 논리량 약 83GB) |

**해석:** 압력 1은 지금 놀고 있을 때의 값이다. 압축 보관량(약 83GB)과 스왑(21GB)이 매우 커서 새로 5–6GB를 올리면 경고로 넘어간다. 실제로 SITE-RECOVERY-010 시험이 27초에 압력 2가 됐다.

## 2. 메모리 큰 것과 서비스 지도

`MEM`은 top의 메모리 사용량(압축분 포함), `CMPRS`는 압축되었거나 스왑된 양이다.

| 프로세스 (launchd) | MEM / CMPRS | 포트·공개 URL | 상태 |
|---|---|---|---|
| **ARDAgent** (`com.apple.RemoteDesktop.agent`, 시스템 서비스, 사용자 소유 pid 60985) | **42G / 42G** | 3283 | 10-10 23:52 시작. launchd 기록 runs=4, 마지막 종료 사유 `OS_REASON_JETSAM` |
| **Docker VM** (`com.apple.Virtualization`) | **8.1G / 13G** | — | 9-22 부팅 때부터 |
| Docker backend (`com.docker.backend`) | 3.5G / 1.6G | 3000, 3001, 3002, 30080, 5678, 6379, 8081, 8123, 8888, 9000, 9001, 9090, 9100, 6443(k8s) | **모든 포트 무응답.** HTTP는 연결 후 3초 timeout(curl exit 28), 6379 nc 실패, 엔진 소켓 API 6초 무응답 |
| Docker Desktop UI | 0.5G / 0.45G | — | |
| `com.eommydocs.bohun-test` (Python) | 3.9G / 0.8G | 8001 → test.eommydocs.com (302) | 사용 중으로 보임 |
| `co.tridatum.rag.app` (Python) | 1.6G / 1.6G | 8766, rag 터널 | 거의 전부 압축(유휴) |
| loginwindow | 1.35G / 1.3G | — | 콘솔 로그인 세션. 건드리면 로그아웃이라 후보 아님 |
| `homebrew.mxcl.kafka` (java -Xmx1G) | 1.2G / 0.54G | 9092, 9093 | 10-10 23:52:16 시작(ARDAgent 시작 4초 뒤) |
| `com.eommydocs.dataprep` worker (pid 1647) | 0.8G / 0.78G | 8501 → dataprep.eommydocs.com (200) | 유휴 worker |
| mysqld | 0.45G / 0.45G | 3306 | |
| `ai.openclaw.gateway` (node) | 0.44G / 0.09G | 18789 | |
| GUI 앱: Slack 3개, Safari + WebKit 3개, LibreOffice, Logi 2개 | 약 0.8G + 0.86G + 0.17G + 0.28G | — | 대부분 압축 |
| screensharingd (root) | 0.4G | 5900 | 지금 연결된 화면 공유 세션 없음 |
| ollama 3개 (11434, 11435, 11436) | 각 15–18M | | 11435(`co.tridatum.rag.ollama`)만 bge-m3 0.63G를 **무기한 유지**(runner 170M). 나머지는 적재 모델 없음 |
| 작은 것: secho.api 68M, datasearch 62M, bohun 51M, caddy, dufs, syncthing, postgres(13M), cloudflared 3개 | 합계 < 0.4G | 18765(403 정상 거부), 8502(200), 8000, 8010, 9999 | |

**ARDAgent "footprint 42GB인데 RSS 4MB"의 원인**
- top의 MEM(footprint)은 압축되었거나 스왑된 익명 메모리까지 센다. RSS는 지금 물리 메모리에 올라 있는 페이지만 센다.
- 그래서 ARDAgent의 42GB는 거의 전부 압축기와 스왑에 있다. 원격 관리 에이전트의 **메모리 누수**로 보인다.
- 근거: launchd 기록상 이미 jetsam(메모리 부족 강제 종료)으로 한 번 죽었다가 다시 떴다(runs=4).
- 10-10 시작 후 22일에 42GB이니, 평균 약 1.9GB/일 [추정]. 4분 간격 두 번 측정은 GB 단위라 증가 속도를 직접 재지 못했다.

**보안 관찰(조치 안 함):** 16:21에 `43.226.60.99`에서 SSH(22) ESTABLISHED 연결이 보였다. `who`와 `last`에는 그 로그인이 없다. 인증 전 스캔으로 보인다 [추정]. sshd 설정 점검을 권한다.

## 3. TRID 사이트에 필요한 양 [추정]

- 맥북 실측 기준 기본 1.7B llama-server 상주 2.2GB, 백엔드(bge-m3 float32 CPU 2.3GB + v5 색인 0.4GB + 본문·별칭 데이터 + 독해기 0.35GB).
- 최소 구성: **약 5–6GB.**
- 배포 파일 그대로 5개 버전: 실험판 GGUF 2개(각 1.8GB, mmap)가 쓰일 때 추가 [추정].
- 모델 양자화·임베더 교체·버전 삭제는 하지 않는다. 지금 ollama의 bge-m3를 같이 쓰면 메모리는 줄지만 다른 수치 표현이라 검색 결과가 바뀐다. 이것은 코드·평가 변경이므로 이번 후보에서 뺐다.

## 4. 정리 후보

모든 후보는 사용자 승인 전에 실행하지 않는다. 회수량은 전부 [추정]이다.

| # | 후보 | 회수 예상 [추정] | 멈추는 URL·작업 | 정지/재시작 | 복구 명령 | 데이터 손실 위험 | 실행 권한 |
|---|---|---|---|---|---|---|---|
| A | **ARDAgent 재시작** (`kill -TERM 60985`, launchd KeepAlive가 다시 띄움) | 압축기 물리 점유 약 1.5GB, 스왑 최대 약 20GB 해제(누수분). 압력 여유가 가장 크게 늘 후보 | 원격 관리(ARD 3283) 몇 초 끊김. 지금 화면 공유와 ARD 연결 없음. SSH, Tailscale, 공개 URL 영향 없음 | 재시작(자동) | 자동. 안 뜨면 `launchctl kickstart gui/501/com.apple.RemoteDesktop.agent`. 지수 재시작 제한이 있어 수십 초 걸릴 수 있음 | 없음(상태 없는 에이전트) | 같은 사용자 소유라 kill 가능 [확인: 소유자 eomgyuhyeon]. 시스템 plist라 설정 변경은 안 함 |
| B | **Docker Desktop 정지**(종료, 데이터 유지) | VM 상주 3–6GB + backend 수백 MB~1.9GB [넓은 추정] | 포트 13개 서비스(5678, 6379, 8123, 9000/9001, 9090, 3000번대, 8888, 8081, 30080, k8s 6443). **지금 이미 모두 무응답이다.** 쓰는 쪽이 있는지는 B 지도 필요 | 정지 | `open -a Docker`(같은 볼륨으로 다시 시작) | 볼륨은 유지된다. 엔진이 무응답이라 정상 종료가 안 되면 강제 종료가 되고, 컨테이너 DB(redis 등)가 마지막 쓰기를 잃을 수 있다(낮음~중간). volume prune·삭제는 안 함 | 사용자 앱이라 GUI 또는 `osascript -e 'quit app "Docker Desktop"'`. CLI는 무응답 가능 |
| B' | Docker Desktop 재시작 | 재시작 후 다시 차지할 것 | 잠시 중단 후 서비스 회복 가능 | 재시작 | 자동 | B와 같음 | 같음 |
| B'' | Docker 메모리 상한 축소 | 상한만큼 | 컨테이너가 OOM으로 죽을 수 있음 | 재시작 동반 | 설정 원복 | 중간 | 설정 파일은 SSH에서 TCC로 접근 불가, GUI 필요. **최후 수단** |
| C | GUI 앱 종료: Slack, Safari, LibreOffice, Logi Options | 물리 약 0.5–1GB(대부분 이미 압축) | 맥미니 화면의 앱 사용 | 정지 | 앱 다시 열기 | Safari·LibreOffice 미저장 문서가 있으면 손실 가능 → 저장 확인 필요 | 사용자 결정 |
| D | 유휴 Python 재시작: rag.app 1.6G, dataprep worker 0.8G | 물리는 적음(대부분 압축), 스왑 약 2GB | 재시작하는 몇 초 동안 rag, dataprep.eommydocs.com 중단 | 재시작 | launchd 자동 | 작업 중 요청 손실 가능, 낮음 | 사용자·서비스 담당 결정 |
| E | Kafka 정지 | 약 1GB | 9092/9093을 쓰는 쪽(미상, B 지도 필요) | 정지 | `brew services start kafka` | 로그 디렉터리 유지, 낮음 | 사용자 결정 |
| F | ollama 11435 bge-m3 무기한 유지 해제 | 약 0.2–0.6GB | rag 첫 요청 지연(재적재) | 언로드 | 자동 재적재 | 없음 | rag 담당 결정. 효과가 작아 권하지 않음 |

**후보 아님:**
- loginwindow: 로그아웃이 된다.
- 재부팅: 금지. SFT controller RunAtLoad와 다른 서비스 영향.
- 컨테이너 데이터와 volume 삭제.
- bohun-test 3.9G: 사용 중인 공개 URL이다.

## 5. 사용자 선택지 (추천 최소 조합)

| 선택 | 내용 | 기대 | 멈추는 것 |
|---|---|---|---|
| **① 추천: A만** | ARDAgent 재시작 → 같은 단계 부하 시험(1.7B + 독해기, 압력 가드) | 누수 42GB가 풀려 스왑과 압축기 여유가 가장 크게 생김. 이것만으로 될 가능성 있음 [추정] | 원격 관리 몇 초 |
| ② A + B | ①로 부족하면 Docker Desktop도 정지(지금 서비스 무응답) | 3–6GB 추가 [추정] | Docker 서비스 전체. **B 지도에서 쓰는 쪽 없음이 확인돼야 함** |
| ③ A + C | Docker 대신 GUI 앱 종료 | 0.5–1GB 추가 [추정] | 맥미니 화면 앱 |
| ④ 이관 보류 | 맥북 원점 유지(SITE-RECOVERY-010 구성) | 지금 작동 | 없음. 노트북 잠자기와 맥북 압력 위험은 남음 |

**ARDAgent 재발 방지(별도 결정):** 누수는 다시 쌓인다(약 1.9GB/일 [추정]). 원격 관리를 쓰지 않으면 시스템 설정에서 끄거나, 주기적으로 재시작해야 한다. 원격 접속 방식에 영향이 있으므로 사용자가 정한다.

## 6. 승인 후 실행 순서

1. 승인된 후보만 실행하고, 시각과 전후 압력·스왑·공개 URL 상태를 기록한다.
2. 다른 서비스 health를 확인한다: test/dataprep/datasearch/secho-api 공개 URL, 8001/8501/8502.
3. 같은 단계 부하 시험을 한다(SITE-RECOVERY-010 §6). 압력 2가 나오면 본인 시험만 정지한다.
4. 통과하면 이관한다: 맥미니 `co.tridatum.trid.site` → 로컬 `/healthz` → 맥북 터널 작업 bootout → 맥미니 터널 작업 bootstrap.
5. 외부 브라우저로 안녕과 원문 근거가 있는 업무 질문을 확인한다. 맥북 모델 서버는 정지 상태여야 한다. Worker는 바꾸지 않는다.
6. 롤백은 역순이다(커넥터는 한 호스트에서만).

## 7. 의존 정보 요청

- B에게 읽기 전용 용도·종속 지도와 자료 수집 일시 정지를 요청했다: `reports/coordination/MINI-CAPACITY-PLAN-011-A-TO-B.md`.
- ②와 E는 B 지도 없이 권하지 않는다.
- Docker 컨테이너 목록은 A가 읽지 못했다. 엔진이 무응답이고 설정 파일은 TCC로 거부됐다. `docker ps`를 반복하지 않았다.

SUPERVISION_REPORT_READY MINI-CAPACITY-PLAN-011
