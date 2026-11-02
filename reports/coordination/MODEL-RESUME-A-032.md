# MODEL-RESUME-A-032: 2b8ba92 배포와 현재 안전 상태

## 결론
- 배포 완료: 맥미니 운영 식별자 `6a5eb37` → `2b8ba92 (branch model-resume-030-followup)`.
- 2026-11-02 05:37:19Z backend 시작, 기존 startup 로그의 ready 시각 05:37:30Z. 약 12초는 로그 시각 차이이며, 외부 연속 가용성 측정은 아니다. 당시 readiness 감시의 `ready_after_s=0`은 성공 증거로 사용하지 않는다(0이 초기값이기도 함).
- **검색·라우팅 배포이며 일반 대화 품질 개선이 아니다.** 새 모델·가중치·프롬프트 변경 없음.
- 현재 추가 배포·재시작·모델 실행은 중지. 부모의 일반 대화 실패 관측은 그대로 수용하며 재현하지 않았다.

## 변경 범위와 복구
- 두 스크립트: `scripts/shared_demo.py`, `scripts/share_demo_control.py`.
- 고정 계약 세 파일: `reports/coordination/DATA-CONTINUE-007/{term-map.jsonl,applies-to-backfill.jsonl}`, `reports/coordination/DATA-GAPS-008/applies-to-backfill.jsonl`.
- staging의 MANIFEST 5개 sha 일치 확인 후 설치. 05:53Z 읽기 전용 확인에서도 shared_demo sha `83ba188af1e1c821469983687501f6fc481f718670717e9a641469911c251b45`, control sha `2e0b3a7af4c82581cdd4dab266d97ffd259ba1a447f19389f056ea8d932e06b1`.
- 유지: CPU-only 환경(`LLAMA_ARG_DEVICE=none`, `LLAMA_ARG_N_GPU_LAYERS=0`), float32 색인 적재, 버전 목록 2개(기본 원본 Qwen + reader), 터널·보안·다른 서비스. 연구 GGUF 복원 없음. main 코드 WIP 변경 없음.
- 복구본: `~/Services/trid-site/rollback-6a5eb37/`의 두 스크립트·DEPLOYED_FROM_COMMIT, 추가 파일 목록. 기존 `rollback-9256df4/` 보존.
- 복구 절차: 두 스크립트와 배포 식별자를 복원하고 추가된 위 계약 파일 세 개를 제거한 뒤 기존 LaunchAgent를 kickstart한다. 이번에는 복구 실행하지 않았다. 현재 압력 경고만을 이유로 재시작을 반복하지 않는다.

## 실제 경로 smoke (이미 실행한 1회, 재실행 없음)
근거: `MODEL-RESUME-A-032/site-smoke-2b8ba92.json`. Mac mini origin에 SSH port forward로 요청했고 실제 모델·색인 경로를 통과했다. 부모의 최종 실제 UI 확인과는 별개다.

| 항목 | 관측 |
|---|---|
| healthz / 버전 | 200 / 원본 + reader 2개 |
| 한국부동산원 기관 질문 | HTTP 200, quoted, 17.1초 |
| 공무원 여비 규정 명시 전환 | HTTP 200, unverified, 15.9초; 검증된 답으로 주장하지 않음 |
| “그럼 숙박비는?” | HTTP 200, continued=true, conditions_carried=[국내], 별표 2, 14.5초 |
| 새 대화 후 동일 후속 | HTTP 200, clarify, 승계 없음 |
| 제거된 version ID | HTTP 422 |
| 공개 페이지 | 기존 HTTP 측정 200, 옵션 2개; 이후 읽기 전용 페이지 조회도 응답 |

## 메모리와 미해결 위험
- 배포 전 압력 2가 지속돼 staging 후 대기했다. 05:37Z에는 압력 2인 상태에서 복구본을 확보하고 재시작했다. **압력 1 안정 확인 후 배포한다는 당초 보수적 기준을 충족한 것은 아니다.**
- 재시작 후 05:38–05:46 표본은 압력 1. 05:46Z 여유 55%, backend RSS 약 1.78GB, llama-server 약 3.31GB, swap 사용 약 4GB.
- **05:53Z 읽기 전용 재확인에서는 압력 2로 돌아왔다.** 이 때문에 장기 안정화나 메모리 문제 해결을 선언하지 않는다. 공개 페이지 응답과 운영 PID는 확인됐고 추가 생성 부하는 주지 않았다. 압력 변화 원인을 이번 코드로 특정할 비교 근거도 없다.
- 알려진 기능 한계: 매우 짧은 “그럼 국외는?” 후속 인식 실패, 조건 동의어·암시 미처리, 되묻기 별표 짝짓기의 주제 미검사, 약칭 규정명 되묻기, 미기록 적용 대상의 찾지 못함, 미검증 답변 다수.
- 일반 대화 의미 보존 실패는 미해결. 모델 크기·가중치 원인 단정은 철회했고 `CHAT-ROOT-A-001.md`로 정적 진단을 분리한다.

## 문서 및 중지 상태
- main README·SHARING의 배포 식별자를 실제와 맞췄다. 배포 브랜치 SHARING 갱신은 docs-only `9d7dcc2`이며 운영 코드 식별자는 여전히 `2b8ba92`다.
- 공개 버전 설명의 “6개 메시지·greedy”와 실제 일반 대화의 “24개 메시지·sampling” 차이를 정적 진단에서 발견했다. 최신 변경 금지 지시에 따라 운영 파일은 고치지 않고 진단 보고에 남긴다.
- A033 자동 후속 CPU/GPU 기동 없음. A031 로그·장부 보존.

SUPERVISION_REPORT_READY MODEL-RESUME-A-032
