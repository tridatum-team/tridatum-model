# SITE-RECOVERY-010 — test.tridatum.co 접속 복구와 맥미니 이관 시도

**상태: 부분 완료.**
- 접속 복구: **완료(맥북 원점).** 고정 원점 + launchd로 같은 장애가 다시 나지 않게 바꿨다.
- 맥미니 이관: **차단(자원).** 최소 구성에서도 메모리 압력 경고가 났다. 사용자 결정이 필요하다.
- 맥북 원점을 멈춘 상태의 외부 검증: **미실시**(이관을 못 했으므로).

담당: A가 유일한 서비스 변경 담당. 실서비스 코드는 배포판 `9256df4` 고정. 검증 전 ROOT-008/009 코드와 데이터는 섞지 않았다.

## 1. 단계별 상태

| 단계 | 상태 | 근거 |
|---|---|---|
| 접속 복구 | 완료 (13:38:42 KST) | 외부 브라우저 왕복 3회 + 총괄 실제 브라우저 PASS(POST /api/ask 200) |
| 재발 방지 | 완료 (14:29 KST) | 고정 원점(named tunnel) + launchd. supervisor 종료 후 13초 만에 launchd가 복구(실측) |
| 이관 준비 | 완료 | 맥미니에 코드·자산·환경·터널·launchd 파일 준비, sha256 일치 |
| 전 기능 이관 | **차단** | 최소 구성(1.7B + 독해기) 기동 27초에 압력 2 → 본인 시험만 정지 |
| 맥북 정지 외부 검증 | 미실시 | 위 차단 때문 |

## 2. 원래 장애 원인 (11-01 새벽)

로그 원본은 `.shared-demo/incident-20261101/`에 보존했다.

1. **01:08 KST쯤부터 임시 터널이 끊겼다.**
   - `tunnel.log`: `Unauthorized: Tunnel not found`. Cloudflare가 임시(quick) 터널을 없앴다.
   - supervisor는 새 임시 터널을 띄우고 Worker를 다시 배포해야 했다. 하지만 복구가 반복 실패했다(`Tunnel recovery failed: RuntimeError`).
   - Wrangler 로그: `auth token has expired and could not be refreshed because the Cloudflare auth server could not be reached`. 맥북 네트워크 문제 정황이다.
2. **02:04:42 KST에 24시간 실행 기한이 끝났다.**
   - `--hours 24`로 10-31 02:04:42에 시작한 supervisor가 기한에 도달했다.
   - 백엔드, llama-server, 터널을 모두 정상 종료했다. `state.json`은 `status: stopped`였다.
3. **설계 문제:**
   - 임시 터널은 다시 연결할 때마다 주소가 바뀐다. 그때마다 Worker 재배포, 즉 Wrangler 인증과 네트워크가 필요하다.
   - 실행 기한이 끝나면 아무도 다시 띄우지 않는다.

## 3. 복구 중 재발(13:44:14 KST, 총괄 실패 1건): 원래 장애와 다른 원인

- **증상:** 총괄 브라우저에서 업무 질문의 `POST /api/ask`가 `net::ERR_CONNECTION_CLOSED`로 끝났다(상태 코드 없음).
- **배제된 원인(증거 있음):**
  - 백엔드 재시작 없음: pid 64525, 13:38:42부터 계속 실행.
  - 터널 복구 0회. 13:44:51 health `ready`.
  - Worker 재배포 없음(마지막 배포 13:39).
  - `server.log`에 traceback 없음, `tunnel.log`에 오류 없음.
  - 모델·색인·임베더 적재 실패 아님: 기동 시 5개 버전 준비, 임베더 가중치 391/391.
- **유력 원인(\[추정\], 증명 못 함): A의 맥미니 파일 전송.**
  - 13:41부터 모델 1.8GB, 13:43부터 색인·데이터 1.2GB를 80초에 보냈다(약 120Mbit/s 업로드). 13:44:14가 이 구간에 들어간다.
  - 같은 때 맥북은 메모리 압력 2, 스왑 17.1/18.4GB였다. rsync 1개가 1.3GB를 차지했고, 백엔드 Python 상주 메모리는 32MB(스왑으로 밀려남)였다.
  - 업로드 포화는 브라우저나 터널 연결을 끊을 수 있고, 상태 코드 없는 연결 종료와 맞는다.
  - 요청 단위 로그(백엔드 요청 로그 없음, Worker 관측 꺼짐)가 없어 확정하지 못했다.
- **조치:**
  1. 13:46에 전송을 멈췄다. 20초 안에 압력이 1로 돌아왔다.
  2. 같은 종류의 다른 질문을 외부에서 보냈다: 200, 9초, 한국부동산원 여비규정 제14조 원문 인용.
  3. 이후 전송은 24Mbit/s로 제한했다. 맥북 압력이 1이 아니면 전송을 자동으로 멈추거나 다시 시작하게 했다(14:11 등 2회 작동).
- **총괄 재검증 PASS:** 같은 질문이 `POST /api/ask` 200, 한국부동산원 여비규정 출처로 응답했다. 답은 "검증된 인용 없음 · 해설만"이라 **품질 합격은 아니다.** 15분 세션 만료는 서버 중단이 아니다.

## 4. 복구와 재발 방지 내용

**4.1 접속 복구 (13:38:42 KST, 맥북)**
- `scripts/share_demo_control.py`에 선택 설정 `TRID_DEMO_SERVICE`를 추가했다.
  - 서비스 파일 경로만 바꾼다. 기본값은 기존과 같다.
  - `scripts/shared_demo_live_9256df4.py`는 `git show 9256df4:scripts/shared_demo.py` 사본이다(sha256 앞자리 `ab0574790f0788c6`, 커밋과 같음). git 무시 목록에 넣었다.
- 함께 import하는 `evaluate_external_chat.py`, `trid/`는 `9256df4` 이후 변경이 없다(작업 트리 포함).
- 임시 터널로 시작하고 Worker를 배포했다(Version `470b2a2e-…`).

**4.2 고정 원점 (14:27 KST)**
- 새 named tunnel `trid-test-origin`(`4562e85a-de9b-4567-a926-47c381d7102e`)을 만들었다. 호스트명은 `trid-test-origin.eommydocs.com`이다.
  - 맥미니 `cert.pem`의 계정이 Worker 계정 `a57b51f…`과 같음을 확인했다(토큰은 읽거나 출력하지 않음).
  - 기존 터널 3개(data-prep, tridatum-rag-mini, tridatum-secho)의 설정은 건드리지 않았다.
- **실수 기록:** 첫 `route dns`가 기본 `~/.cloudflared/config.yml`의 `tunnel:`(data-prep)을 따라 새 CNAME을 data-prep 터널로 만들었다(04:44:50Z).
  - 14초 뒤 전용 설정과 `--overwrite-dns`로 새 터널로 바꿨다(04:45:04Z).
  - 영향은 새 호스트명 하나뿐이다. data-prep ingress에 그 규칙이 없어 404였고, 기존 레코드는 바꾸지 않았다.
- 원점 확인: 비밀키 포함 `/healthz` 통과, 비밀키 없이 403.
- `share_demo_control.py deploy --origin URL`을 추가했다. URL이 비밀키 health를 먼저 통과해야 Worker를 바꾼다. Worker Version `a3bb3219-a675-4e72-9269-8e3f2f17eb1a`, 무중단 전환이다.

**4.3 launchd (14:28–14:29 KST)**
- `~/Library/LaunchAgents/co.tridatum.trid.site`
  - `supervise --hours 168`, `TRID_DEMO_TUNNEL=0`, 배포판 사본 서비스.
  - KeepAlive를 켜서 기한 만료나 장애로 끝나도 다시 뜬다.
- `co.tridatum.trid.site.tunnel`: 위 named tunnel 커넥터.
- 파일: `deploy/macbook-site/`, 맥미니용은 `deploy/macmini-site/`.
- **전환 중 중단:** 임시 터널 supervisor를 내리고 launchd supervisor가 뜨기까지 약 22초.
- **재시작 실측:** supervisor에 SIGTERM(기한 만료와 같은 정상 종료 경로)을 보냈다. launchd가 13초 만에 새 supervisor를 띄웠고 모델 서버도 같은 배포판으로 돌아왔다. 외부 GET / 200.

## 5. 현재 정상 요청 샘플

- 검증용 URL: https://test.tridatum.co
- 원점: `trid-test-origin.eommydocs.com` → 맥북 `127.0.0.1:51873`

| 시각 (KST) | 경로 | 질문 | 결과 |
|---|---|---|---|
| 13:39 | 임시 터널 | 안녕하세요 | 인사 응답, "일반 대화 · 문서 근거 없음" |
| 13:39 | 임시 터널 | 공무원 국내 출장 숙박비 상한은 어떻게 정해져 있나요? | 출처 인사혁신처 공무원 여비 규정(law.go.kr 링크), 해설만·표 한계 |
| 13:46 | 임시 터널 | 한국부동산원 여비규정에서 숙박비는 어떻게 지급하나요? | 200, 9초, 제14조 원문 인용 |
| 총괄 | 임시 터널 | 한국부동산원 여비규정에서 시내지역 출장에도 식비를 지급하나요? | 200, 한국부동산원 여비규정 출처, 해설만 |
| 14:30 | **고정 원점·launchd** | 안녕 | 200, 3초, 인사 응답 |
| 14:30 | **고정 원점·launchd** | 한국부동산원 여비규정에서 일비는 어떻게 지급하나요? | 200, 13초, 한국부동산원 여비규정 게시본 출처, 해설만 |

전부 연결 검증이다. 모델 정답 측정이 아니다.

## 6. 맥미니 이관: 준비 내용과 차단 사유

**준비 완료** (`~/Services/trid-site`, 기존 서비스 디렉터리와 분리)
- `repo/`: `git archive 9256df4` 코드.
- 자산 71개 파일(v5 색인, inst-reb-002, passages, 기관·약칭 데이터, 별표 행, llama.cpp `8ca4060`): 맥북과 sha256 집합 일치.
- Qwen3-1.7B Q8_0 `061b54da…`, 추출 독해기 5개 파일: 일치.
- 임베더: `hf/` 아래 bge-m3 필요한 파일만(onnx 제외). `pytorch_model.bin` blob sha 일치.
- `venv/`: Python 3.12, 맥북 `.venv` freeze와 같은 68개 패키지.
- `tunnel/`: 전용 config와 자격 증명(0600). 비밀키는 `.runtime/proxy-secret`(0600).

**단계 부하 시험**

| 항목 | 내용 |
|---|---|
| 기준 | 압력 1, free 56%, 스왑 21.9/23.5GB, 기존 웹 서비스 8001/8501/8502 200, secho-api 18765 403(정상 거부) |
| 1차 (1.7B만) | 배포 코드가 "추출 독해기 버전이 하나는 있어야 함"으로 시작을 거부했다. 업로드 문서 분할에 독해기 토크나이저가 필요하다. 최소 구성은 1.7B + 독해기 |
| 2차 (1.7B + 독해기) | 기동 27초(백엔드 ready 출력 직후)에 **압력 2**. 가드가 본인 시험만 정지했다. 이후 압력 1로 복귀, 기존 웹 서비스 200 유지, 본인 프로세스와 포트 51873/51912 잔존 없음 |
| 결론 | 현재 맥미니 부하(Docker, ollama 3개, 기존 llama-server, java, 스왑 거의 소진)에서는 **최소 구성도 압력 기준(1)을 못 지킨다.** 버전 수를 줄여도 해결되지 않는다(이미 최소). 3버전 동시 상주는 시험하지 않았다 |

**사용자 결정이 필요한 선택지** (A는 기존 서비스를 끄거나 설정을 바꾸지 않았다)

| 선택지 | 효과 | 비용·위험 |
|---|---|---|
| ① 맥미니 메모리 확보 후 재시험 (Docker VM 메모리 축소, 안 쓰는 컨테이너·ollama 모델 정리 등) | 이관 가능성. 재시험은 같은 단계 시험으로 바로 할 수 있다 | 다른 서비스 영향. 사용자나 해당 서비스 담당이 결정해야 한다 |
| ② 맥북 원점 유지 (현재) | 지금 작동. 고정 원점·launchd로 기한 만료와 터널 주소 변경 재발은 막았다 | 노트북 덮개 닫힘·네트워크 끊김에 취약(`caffeinate -i`는 유휴 잠자기만 막음). 맥북 자체도 압력 2가 반복된다(14:31 전송 없이 압력 2, 스왑 17.0/18.4GB) |
| ③ 맥미니 지연 로딩·버전 축소 설계 | 메모리 절감 | 서비스 코드 변경이라 검증과 C 확인이 필요하다. 최소 구성도 경고가 났으므로 이것만으로는 부족할 가능성이 크다 |

**이관 절차(준비됨):**
1. 맥미니에서 `co.tridatum.trid.site`를 bootstrap한다. versions는 기본값인 배포 파일 전체를 쓰며, 버전을 줄이려면 사용자 결정이 필요하다.
2. 로컬 `/healthz`를 확인한다.
3. 맥북 `co.tridatum.trid.site.tunnel`을 bootout한다.
4. 맥미니 `co.tridatum.trid.site.tunnel`을 bootstrap한다. Worker는 그대로 둔다.
5. 외부 검증을 한 뒤 맥북 모델 서버를 bootout한다.
- 롤백은 역순이다. 커넥터는 한 호스트에서만 띄운다.

**맥미니 관찰만 한 것**
- `co.tridatum.trid.sft-controller`: RunAtLoad true, KeepAlive 설정 있음. 현재 실행 중 아님(마지막 종료 0).
- 재부팅 실험은 하지 않았다.

## 7. 롤백

- **고정 원점 → 임시 터널:**
  1. `launchctl bootout gui/$(id -u)/co.tridatum.trid.site.tunnel`
  2. `launchctl bootout gui/$(id -u)/co.tridatum.trid.site`
  3. `share_demo_control.py start --hours 24`
  4. `share_demo_control.py deploy`
- **Cloudflare 정리(필요 시):** `cloudflared tunnel delete trid-test-origin`과 CNAME `trid-test-origin.eommydocs.com` 삭제. 맥미니 cert 계정에서 실행한다.
- **Worker 이전 버전:** `470b2a2e-…`(임시 터널 주소, 이미 무효), `c8470e7e-…`(장애 전).

## 8. 테스트 timeout·브라우저 불안정과 맥북 과부하 (\[추정\], 실측값만)

- 실측값:
  - 맥북 스왑 17.0–17.1/18.4GB.
  - 압력 2 관측: 13:44 전후, 13:56, 14:11(전송 중), 14:31(전송 없음).
  - 상주 프로세스: 다른 omp 세션 3개(0.55–0.64GB), ChatGPT/Codex 0.93GB, Chrome.
- 같은 시기 증상:
  - 0.1초 걸리는 `select`가 벽시계 1029초.
  - 전체 테스트가 subprocess에서 64개 25.7초(이전 9.35초).
  - 셸 백엔드가 600초 무응답(bg_118).
  - eval 커널이 응답하지 않아 강제 종료됨.
- 이 증상들은 과부하와 맞지만 인과는 증명하지 못했다.

## 9. 미완료

- 맥미니 이관과 맥북 정지 외부 검증: §6 사용자 결정 대기.
- 맥북 원점은 노트북 잠자기와 네트워크에 의존한다.
- 요청 단위 관측 수단이 없다(백엔드 요청 로그 없음, Worker 관측 꺼짐). 13:44 실패 원인을 확정하지 못한 이유다. 켜는 것은 개인정보 정책 판단이 필요해 이번에 하지 않았다.
- RETRIEVAL-ROOT-009는 이 작업으로 중단했다. 배포 금지는 유지된다.

SUPERVISION_REPORT_READY SITE-RECOVERY-010
