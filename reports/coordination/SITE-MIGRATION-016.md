# SITE-MIGRATION-016 — bohun 테스트 서비스 중단(사용자 승인) 후 맥미니 이관 시도

**상태: 차단.**
- 승인된 bohun-test 정상 종료는 성공했다.
- 그러나 배포판 최소 구성이 **첫 업무 질문에서 압력 2**가 나 이관하지 못했다.
- 원점은 맥북 그대로다(롤백할 것 없음). test.tridatum.co 18:37 GET 200.
- **bohun-test(:8001, test.eommydocs.com)는 사용자 승인대로 꺼진 상태다.** 다시 켜려면 명시 협의가 필요하다.

## 1. bohun-test 정상 종료

**신원 확인 (18:32, 종료 명령 안에서 다시 확인)**
- launchd `gui/501/com.eommydocs.bohun-test`, plist `~/Library/LaunchAgents/com.eommydocs.bohun-test.plist`.
- `/bin/bash /Users/eomgyuhyeon/apps/bohun-ai-test/start.sh`, `PORT=8001`, KeepAlive, RunAtLoad.
- 프로세스 pid 1067(uvicorn, 9-22 시작)이 launchd pid와 127.0.0.1:8001 리슨 pid와 같았다. MEM 3.86G, 압축 0.80G.

**실행:** 18:32:51 `launchctl bootout gui/501/com.eommydocs.bohun-test`.
- rc=0, 즉시 반환.
- 서비스 로그(`~/Library/Logs/bohun-test.err.log`): `Shutting down → Waiting for application shutdown → Application shutdown complete → Finished server process [1067]`. **정상 종료**이며 launchd 종료 제한(5초) 강제 종료는 일어나지 않았다.
- 작업은 언로드되어 다시 뜨지 않는다. 8001은 닫혔다.
- plist는 그대로 남아 있다. **재부팅이나 재로그인 때 다시 뜬다.** `launchctl disable`은 설정 변경이라 하지 않았다.

**영향 (18:33:28)**
- 8001: 응답 없음(승인된 영향).
- 다른 서비스는 전과 같다.
  - 로컬 8000=200, 8010=401, 8501=200, 8502=200, 18765=403, 8766=403, 9999=401.
  - 공개 dataprep=200, datasearch=200, secho-api=403, files=301.
  - mysql, postgres, kafka 연결 OK, ollama 3개 200.
- `test.eommydocs.com`은 종료 뒤에도 302였다. 앞단 인증 리디렉션으로 보이며 원점 동작 근거가 아니다 [추정].

**메모리 전후**

| 시각 | 압력 | free% | 스왑 | free | active | inactive | wired | 압축기 물리 |
|---|---|---|---|---|---|---|---|---|
| 18:32:31 (전) | 1 | 64% | 6,053MB | 0.14 | 4.97 | 4.93 | 2.74 | 2.67 |
| 18:33:28 (후) | 1 | 64% | **3,534MB** | 0.08 | 4.98 | 4.93 | 2.74 | 2.66 |

- 회수된 것은 스왑 −2.5GB뿐이다. **물리 여유(free + inactive)는 그대로다.** bohun-test 메모리는 대부분 이미 압축이나 스왑 상태였다.

## 2. 단계 시험 (배포판 9256df4, 기본값 그대로: Metal, 같은 모델 sha·정밀도·임베더·컨텍스트 4096)

- 격리 포트 51973, 1초 표본, 압력 가드, 버전 `versions-stage1.json`(기본 1.7B + 독해기).

| 시각 | 단계 | 압력 | free | wired | 압축기 | TRID RSS |
|---|---|---|---|---|---|---|
| 18:34:06 | 시작 | 1 | 0.09 | 2.75 | 2.65 | 0.01 |
| 18:34:15 | ready 직후 | 1 | 0.24 | 4.93 | 4.26 | 4.64 |
| 18:34:16–18:35:16 | 60초 유휴 | 1 | 0.06–0.09 | 4.93 | 3.97–4.04 | 4.73 |
| 18:35:17 | 대화 질문 "오늘 업무 시작 전에 짧게 응원 한마디 해 줄래?" | 1 | | | | 200, 1.5초, 대화 응답 |
| 18:35:20 | 기관 질문 "한국부동산원 여비규정에서 숙박비는 어떻게 정해져 있나요?" | **2** | 0.04 | 4.93 | 4.25 | 5.27 → 가드가 본인 시험 정지(요청 연결 끊김) |

- 72개 표본 중 압력 2는 1회. 시험 후 다른 서비스는 전부 기준과 같았다.
- 맥미니에 본인 프로세스, 시험 런타임, 임시 파일은 남아 있지 않다.

**전체 5버전 단계는 하지 않았다.** 최소 구성이 통과하지 못했기 때문이다.
- 그 준비로 실험판 GGUF 2개를 보내던 전송은 멈췄다. 맥북 압력 가드와 24Mbit/s 제한 상태였다.
- 부분 사본은 지웠다.
- 전송은 시험과 같은 때 맥미니 디스크에 쓰고 있었다. 파일 캐시에 영향이 있었을 수 있다. 다만 결과는 이전 4회 실측(SITE-MIGRATION-012/013, SITE-CPU-TRIAL-014)과 같은 양상이다.

## 3. 결론

- **이번 시험 구성(Metal, 실험판 GGUF 전송이 동시에 진행)에서 질의 시 압력 1 운영 기준에 못 미쳤다.** 원인은 물리 여유 부족으로 보인다. SITE-MEMORY-015 구성 요소 합산 추정 약 5.6–6.0GB(엄밀한 하한 아님), Metal 사용 시 wired +2.2GB.
- 지금까지 정리(ARDAgent 재시작, Docker VM 정지, bohun-test 중단)는 **스왑만** 줄였다. 21.2GB → 3.5GB.
- 물리 여유(free + inactive 약 4–5GB)는 거의 늘지 않았다. 정리한 프로세스들의 메모리가 이미 압축이나 스왑에 있었기 때문이다.
- 그래서 터널 전환, 맥북 비의존성 외부 검증, 버전 전환·문서 첨부 외부 검증은 할 조건이 없었다. 총괄에 알릴 외부 검증 시점도 없다.

## 4. 현재 상태와 다음 결정

| 대상 | 상태 |
|---|---|
| test.tridatum.co | 맥북 원점(launchd + 고정 named tunnel) |
| bohun-test (:8001, test.eommydocs.com) | **꺼짐**(사용자 승인). 재부팅·재로그인 시 RunAtLoad로 다시 뜸. 다시 켜기: `launchctl bootstrap gui/501 ~/Library/LaunchAgents/com.eommydocs.bohun-test.plist` (협의 후) |
| Docker | VM 정지, 앱 UI·backend 잔존 (SITE-MIGRATION-013) |
| ARDAgent | 재시작 후 pid 88363 |

**남은 선택지** (SITE-MEMORY-015 §6과 같음)
- 정확도 trade-off: 임베더 float16, 컨텍스트 축소, 양자화 변경.
- 더 큰 RAM 기기 또는 다른 호스트.
- 맥북 유지.
- ~~같은 수치를 지키는 서비스 정리로는 이 기기에서 질의 부하 기준을 맞출 수 없다는 것이 실측으로 반복 확인됐다.~~ **정정(SITE-CPU-POSTCLEAN-017, 총괄 검토):** 근거보다 강한 단정이었다. CPU-014는 bohun 중단 전이었고, 이번 016은 Metal에 GGUF 전송이 겹쳤다. 정확한 표현은 "지금까지의 시험 구성에서 압력 1 운영 기준에 못 미쳤다"이다. 정리 후 CPU 전용·전송 없음 조건은 SITE-CPU-POSTCLEAN-017에서 따로 시험한다.
- bohun-test를 다시 켤지도 사용자 결정이다. 지금 이관에 쓰이지 않으므로 복구해도 이관 결과에는 영향이 없다.

**ROOT-009:** 장애 복구로 중단한 상태 그대로다. watcher를 다시 돌리지 않았고 나중에 재개한다.

SUPERVISION_REPORT_READY SITE-MIGRATION-016
