# MINI-CAPACITY-PLAN-011 — A → B 요청 (읽기 전용)

사용자가 "맥미니 자원 정리 후 TRID 사이트 이관"을 골랐다. 정리 대상은 사용자 승인 전에는 아무것도 멈추지 않는다. 승인을 받으려면 A의 후보표(`MINI-CAPACITY-PLAN-011.md`)에 **용도와 종속 관계**가 필요하다.

**B 요청**
1. 맥미니 자료 수집 작업을 이 계획이 끝날 때까지 일시 정지한다.
2. 아래 항목마다 읽기 전용으로 적어 달라: 용도, 주인(누가 쓰나), 호출하는 쪽(종속), 공개 URL, 일시 정지 가능 여부와 그때 멈추는 기능, 데이터 상태(영속 볼륨·DB 여부).
3. 종료·재시작·설정 변경·`docker ps` 반복 호출·volume 정리는 하지 않는다. Docker 엔진은 A 측정에서 무응답이었다(소켓 API 6초 무응답, 공개 포트 13개 모두 연결 후 3초 timeout).

| 항목 | A가 측정한 사실 (2026-11-01 16:20–16:24 KST) |
|---|---|
| Docker Desktop (VM `com.apple.Virtualization`, `com.docker.backend`) | VM MEM 8.1G / 압축 13G, backend 3.5G. 포트 3000, 3001, 3002, 30080, 5678, 6379, 8081, 8123, 8888, 9000, 9001, 9090, 9100, 6443(k8s) — 모두 응답 없음. 컨테이너 목록·compose 프로젝트는 A가 읽지 못함(TCC로 설정 파일 거부, 엔진 무응답) |
| `homebrew.mxcl.kafka` (java -Xmx1G, 9092/9093/58513) | MEM 1.2G, 10-10 23:52 시작 |
| `com.eommydocs.bohun-test` (8001, test.eommydocs.com) | Python MEM 3.9G, 공개 302 |
| `com.eommydocs.bohun` (8000) + `com.eommydocs.bohun.caddy` (8010) | 51M / 21M |
| `com.eommydocs.dataprep` (8501, dataprep.eommydocs.com) + 자식 worker pid 1647 | worker 801M(대부분 압축), 공개 200 |
| `com.eommydocs.datasearch` (8502, datasearch.eommydocs.com) | 62M, 공개 200 |
| `co.tridatum.rag.app` (8766) + `co.tridatum.rag.ollama` (11435, bge-m3 keep_alive 무기한 0.63G, runner llama-server 53798) + `co.tridatum.rag.tunnel` | rag.app 1.58G(거의 전부 압축) |
| `co.tridatum.secho.api` (18765, secho-api.eommydocs.com) + `co.tridatum.secho.ollama` (11436, 적재 모델 없음) | 68M, 공개 403(정상 거부로 보임) |
| `homebrew.mxcl.ollama` (11434, 적재 모델 없음) | 16M |
| `ai.openclaw.gateway` (node 18789) | 442M |
| mysql(3306), postgresql@17(5432), syncthing, dufs(files.eommydocs.com), caddy | 작음 |
| GUI 앱: Slack, Safari, LibreOffice(soffice), Logi Options | 합계 약 1.7G(대부분 압축) |

결과는 `reports/coordination/MINI-CAPACITY-PLAN-011-B-MAP.md`로 부탁한다. A는 그 결과를 후보표의 "종속" 열에 반영한다.
