# SITE-MEMORY-022 — 같은 결과를 유지하는 TRID 로딩 메모리 절감과 맥미니 재검증

**상태: 부분 완료. 절감 1건은 구현·검증했다(결과 동일). 이관 조건은 미충족이다.**
- 절감: 상시 익명 메모리 −0.40GB, 기동 최고치는 그대로.
- 맥미니 전체 기동 2회 중 1회는 기동 뒤 **디스플레이 켜짐과 Aerial 배경화면 재생 순간 압력 2가 30초** 이어져 가드가 정지했다.
- 그래서 launchd 15분 관찰, 터널 전환, 맥북 정지는 하지 않았다. 원점은 맥북(배포판 `9256df4`) 그대로다.
- 배경화면, 알림, 권한, 다른 서비스는 바꾸지 않았다.

## 1. 구현 위치

- `9256df4`에서 갈라진 별도 브랜치 `site-memory-022`(worktree `../trid-memory-022`), 커밋 `2ff6822`.
- 미완성 ROOT-009와 미배포 ROOT-008은 섞지 않았다. main에는 아직 병합하지 않았다.

## 2. 후보별 결과 (각각 근거)

| # | 변경 | 결과 | 처리 |
|---|---|---|---|
| ① | 색인 dense 벡터: `dense.f16.npy`를 메모리에서 float32로 바꾸는 대신, 정확히 변환한 `dense.f32.npy`(f16 sha256 기록 `dense.f32.json`)를 읽기 전용 mmap (`trid/service/embed.load_dense`, `write_dense_f32`) | Workspace v5+inst 상시 own RSS **1.368 → 0.970GB**, physical footprint **1.4 → 1.0GB**. 최고치 2.277GB 그대로(최고치는 그 전 `Index.load` 단계). 질의 후 RSS는 같다(2.07GB). 차이는 dense 0.4GB가 익명 메모리가 아닌 회수 가능한 파일 페이지라는 점 | **유지** |
| ② | 독해기 가중치를 텐서 하나씩 복사(두 벌 동시 보유 제거) | own 0.356GB, 최고 0.881GB, **변화 없음**. 텐서 99개 digest 동일 | 효과 없음 → **되돌림** |
| ③ | 검토 파일(74MB)과 별칭 파일(18MB)을 줄 단위로 읽기 | 줄 목록은 같았다(41,369 / 46,626줄). 그러나 단계별 vmmap에서 **footprint가 1.1 → 1.9GB로 나빠졌다**(`MALLOC_LARGE (empty)` 1.0GB dirty). 기존 통째 읽기가 `Index.load`의 큰 해제 블록을 재사용하고 반환하는 동작이 사라진 것으로 보인다 [할당기 수치에서 추정] | **되돌림** |
| — | 시작 후 `malloc_zone_pressure_relief` 호출 | 측정 변화 없음 | **되돌림** |

## 3. 같은 결과 검증 (변경 ①, 맥미니, 압력 가드, 매번 압력 1)

| 항목 | 결과 |
|---|---|
| f32 사본 = `np.load(f16).astype(float32)` | `np.array_equal` True, dtype float32, C-contiguous |
| 행렬 곱 | 무작위 단위 벡터 64개 모두 비트 동일, 결과 타입 `ndarray` |
| `Workspace.select()` 32문항 | 결과 dict 전부 동일(기존 출장비·여비·양식·기관·공개 고정 질문, bge-m3 실제 질문 벡터) |
| cos 앞부분 | 동일 |
| 검토·별칭 기록 | 경로 필드 말고 동일 |
| 오래된 사본 방지 | 새 회귀 테스트 `tests/test_service_embed.py`: f16이 바뀌면 사본을 쓰지 않고 메모리 변환으로 돌아간다. 이 테스트가 처음에 fallback이 `np.memmap` 타입을 반환하는 결함을 잡았고, `np.asarray`로 고쳤다(값은 원래 같았음) |
| 회귀 테스트 1회 | `test_service_embed`, `test_service_index`, `test_shared_demo`, `test_extractive` **87개 OK**. worktree에 git 무시 데이터는 심볼릭 링크로 연결 |

## 4. 맥미니 배포와 전체 기동 시험

**배포**
- 맥미니 `~/Services/trid-site/repo`에 `2ff6822`의 두 파일을 넣었다. 이전 파일은 `rollback-9256df4/`에 보존했다.
- `DEPLOYED_FROM_COMMIT=2ff6822…`. f32 사본은 v5와 inst 모두 썼다(검증본과 바이트 동일).
- 서비스는 지금 실행 중이 아니다.

**시험 조건**
- 018과 같은 수동 기동, CPU 전용, 운영 버전 파일(기본 + 독해기), 격리 포트 51973.
- 120초 watchdog, 경고 30초 / 심각 4 / 기존 서비스 기준 가드.
- 전송 없음. Docker VM 정지.

| | 첫 기동 (21:23:44) | 연속 기동 (21:25:48) |
|---|---|---|
| CPU 적용 (llama-server 실측) | 인자에 `-ngl`·`-dev` 없음. env `LLAMA_ARG_DEVICE=none LLAMA_ARG_N_GPU_LAYERS=0`. IOAccelerator 64K, GGUF mapped 1.8G(상주 1.4G) | 같음(GGUF 상주 1.6G) |
| 백엔드 footprint (ready 직후) | 1.9G, 최고 2.6G | 1.9G, 최고 2.6G |
| 기동 경고 | 21:23:51–21:24:00 **7초**(018은 5초) | **없음** |
| 기관 질문 "한국부동산원 여비규정에서 숙박비는 어떻게 정해져 있나요?" | HTTP 200, **20.1초**, status=quoted, 한국부동산원 여비규정, 원문 일치 3건 | HTTP 200, **20.2초**, 같은 결과 |
| 이후 | **21:24:50 디스플레이 켜짐 → Aerial 재생 → 21:24:52부터 압력 2가 30초 → 가드 정지(21:25:23)** | 115초 끝까지 압력 1(표본 106개 모두 1) |
| 최대값 | 압축기 6.94, TRID RSS 5.73, wired 4.19 | 압축기 5.96, TRID RSS 6.08, wired 2.97 |
| 정지 후 | 본인 프로세스 0, 기존 서비스 기준 그대로 | 같음 |

**첫 기동 경고의 시스템 로그 (읽기 전용)**
- 21:24:50.538 `powerd: Display state: On NotificationWake : 0`(알림 깨움 아님, 깨운 원인은 기록에서 확인하지 못함).
- 21:24:50.621 `WallpaperAerialsExtension` "Switch to .playing"(2개), 21:24:51.13 "beginRampingDown".
- 우리 표본: 21:24:54 wired 4.19GB, TRID RSS 1.00GB(압축됨). 이후 wired 2.94–2.98로 돌아왔는데도 압축기 6.5–6.7GB, 압력 2가 이어졌다.

## 5. 결론과 남은 실제 제약

- **수치:** 같은 결과를 유지하며 줄인 것은 dense 0.4GB(상시 익명 → 파일 페이지)뿐이다. 기동 최고치(`Index.load`의 문서 JSON 전체 문자열과 파싱 구조)는 그대로다. 줄이려면 색인 로더를 다시 써야 해서 이번 범위 밖이다.
- **이관 조건:**
  - 맥미니에서 TRID가 떠 있는 동안 **디스플레이가 켜지는 일**(알림 깨움 외에도 있음)이 생기면 Aerial 배경화면 재생이 겹쳐 압력 2가 30초 이상 갈 수 있음을 실측했다.
  - 같은 날 다른 두 시험(018, 이번 연속 기동)에서는 디스플레이 이벤트가 없었고 압력 1이었다.
  - 배경화면과 디스플레이 설정은 사용자 결정대로 그대로 두었다. 그래서 019 기준(경고 30초 한도)을 지속적으로 지킨다고 볼 근거가 없다.
- **성능:** CPU 질의 약 20초(첫 질의와 연속 질의 같음). 맥북 Metal은 9–13초다.

## 6. 하지 않은 것

- launchd 15분 관찰(대화, 기관 전환, 첨부, 독해기, 새 대화, 버전 목록, 삭제 ID 거부).
- named tunnel 전환, 맥북 TRID 정지와 disable, 외부 검증. 총괄에 알릴 외부 검증 시점은 없다.
- docs 실제 상태 갱신. 이관 전이다.
- 브랜치 `site-memory-022` main 병합. 배포 판단과 함께 총괄이 정한다.

## 7. 현재 상태

| 대상 | 상태 |
|---|---|
| test.tridatum.co | 맥북 원점, `9256df4` 사본, 변경 없음 |
| 맥미니 TRID | 실행 중 아님. `repo`는 `2ff6822` 파일 + f32 사본, 되돌리기 파일은 `rollback-9256df4/`. LaunchAgent 없음 |
| 원시 로그 | `reports/coordination/SITE-MEMORY-022/start-{first,warm}-{events,samples}.txt` |

SUPERVISION_REPORT_READY SITE-MEMORY-022
