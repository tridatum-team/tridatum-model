# SITE-MEMORY-023-DOCS — 배포 브랜치·main 문서를 실제 운영 상태로 맞춤

**상태: 완료.** 문서와 이미 배포된 운영 설정 파일만 바꿨다. 코드 변경, 서비스 재시작, 테스트 재실행은 없다. ROOT-009 코드는 섞지 않았다(main 작업 트리의 미커밋 ROOT-009 파일은 그대로다).

## 1. 바로잡은 불일치

| 위치 | 이전 내용 | 바꾼 내용 |
|---|---|---|
| `README.md` 39행 "실험 모델 이름" | 실험판·수리 실험판·기준선을 사이트 선택지로 둔다(브랜치는 C-028 평가 대기로 남아 있었음) | C-028 결과(원본 대비 회귀)를 적고, 두 실험판과 기준선은 2026-11-01부터 사이트에서 제공하지 않으며 모델 파일은 맥북에 보존한다고 적음 |
| `README.md` 41행 "기존 모델·테스트 서비스" | "v0.1 추출 독해기, 기준선, TRID 1.7B 실험판도 고를 수 있다" | 2026-11-01 22:08 KST부터 맥미니 CPU(GPU offload 0) + 고정 named tunnel, 기본 원본과 독해기 두 버전만 제공, 운영 코드는 `site-memory-022` `6a5eb37`(main 미병합), CPU 기관 질문 약 17–21초 |
| `docs/SHARING.md` 5–22행 | 임시 터널 → 이 Mac, 2026-10-31 22:48 자동 종료, 5버전 표 | 고정 named tunnel → 맥미니 CPU, launchd 두 작업 상주, 맥북 원점 정지·비활성화, 현재 코드·색인, 검증 한계, 제공 2버전 표, 미제공 3버전(삭제 ID는 422, 원본 대체 없음) |
| 브랜치 `docs/SHARING.md` "중지 / 재실행" | 맥북 임시 터널 `start --hours 24` → `deploy` 절차 | 맥미니 원점 운영: 배치 경로, launchd 작업과 환경 설정(CPU env), named tunnel 단일 커넥터, 파생 색인 파일 재생성 명령, 맥북 롤백은 main 문서로 안내, 로그 위치 |
| main `docs/SHARING.md` 운영 절 | (SITE-MEMORY-023에서 이미 맥미니로 갱신) | 그대로 두고 상단 절만 위와 같은 문장으로 맞춤 |

- 두 브랜치의 README 39·41행과 SHARING 상단 절은 같은 내용이다. 다른 점은 두 가지다.
  - 브랜치 README는 "이 브랜치의 코드 커밋 `6a5eb37`"이라고 쓴다.
  - main SHARING의 model-repair 줄은 C-028 수치를 함께 적는다.

## 2. 커밋과 경로

| 브랜치 | 커밋 | 경로 |
|---|---|---|
| `site-memory-022` (배포 브랜치, worktree `../trid-memory-022`) | `68cc09b` | `README.md`, `docs/SHARING.md` |
| | | `deploy/macmini-site/{versions.json, co.tridatum.trid.site.plist, co.tridatum.trid.site.tunnel.plist, tunnel-config.json}`: main의 같은 파일을 경로 지정 `git checkout main -- …`으로 가져왔다. 맥미니 배치본과 sha 일치(`versions.json` `dcfa42fd…`, site plist `95fc4a36…`, LaunchAgents 사본도 같음) |
| main | 이 보고서 커밋 | `README.md`, `docs/SHARING.md`, 이 보고서 |

- 배포 브랜치는 보존했다. 코드 경로(`scripts/`, `trid/`, `tests/`)는 `6a5eb37` 이후 바뀐 것이 없다(`git diff --stat 6a5eb37 HEAD -- scripts trid tests` 빈 출력).
- main에 병합하지 않았다(강제 병합 없음).
- 맥미니 `DEPLOYED_FROM_COMMIT`는 코드 커밋 `6a5eb37`을 가리킨다. `68cc09b`는 문서와 이미 배치된 설정 파일의 기록이며, 맥미니에 새로 배치할 코드 변경이 없다.

## 3. 현재 원점과 검증 한계 (문서에 적은 것과 같음)

- **원점:** test.tridatum.co → Worker → `trid-test-origin.eommydocs.com` → 맥미니 launchd `co.tridatum.trid.site.tunnel` → `co.tridatum.trid.site`(127.0.0.1:51873, CPU 전용).
  - 맥북 TRID site, 모델, 터널은 정지했고 `launchctl disable` 상태이며 plist를 보관했다.
  - 총괄 외부 Chrome 확인: 기관 질문 POST 200(17.539초), 숙박비 원문 인용 3건.
- **재부팅 미검증:** 맥미니 재부팅 뒤 자동 복구는 시험하지 않았다. `autoLoginUser`만 확인했다(SFT controller 위험으로 재부팅 금지).
- **화면 깨움 미재현:** 디스플레이가 꺼진 상태에서 켜지며 배경화면 동영상이 재생되는 순간의 메모리 여유는 023 코드로 재현하지 못했다(022에서 wired +1.48GB, 압력 2가 30초 지속된 사례).
- **기관 전환 결함:** 같은 대화에서 다른 기관 규정으로 바꿔 물으면 이전 기관 규정이 이어진다(관찰 7회 모두 unverified). 배포판 동작이며 이번 이관과 별개다.
- **성능:** CPU라 기관 질문 약 17–21초. 성능 보장 없음.
- **전제:** 맥미니 bohun-test 정지(재로그인 시 다시 뜸), Docker VM 정지.

## 4. 다음

- ROOT-009는 별도 명령이 올 때까지 재개하지 않는다.

SUPERVISION_REPORT_READY SITE-MEMORY-023-DOCS
