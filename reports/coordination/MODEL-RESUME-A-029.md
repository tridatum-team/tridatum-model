# MODEL-RESUME-A-029 (A): test.tridatum.co 연구 모델 노출·보존 현황 (읽기 전용 조사)

작성: A, 2026-11-02.

**결론**
- 사용자가 본 대로 지금 사이트에는 기존 두 모델만 있다.
  - 기본: `trid-assistant-qwen3-1.7b`(원본 Qwen3-1.7B)
  - `trid-v0.1-reader`
- 사라진 것은 세 항목이다.
  - 연구 모델 2개: `trid-qwen3-1.7b-pilot-sft-001`, `trid-qwen3-1.7b-model-repair-002`
  - 중복 기준선 1개: `baseline-qwen3-1.7b`
- **세 항목 모두 삭제가 아니라 숨김이다.**
  - 맥미니 이관(2026-11-01 22:08 KST)에서 운영 버전 목록(`deploy/macmini-site/versions.json`)에 넣지 않았다.
  - 모델 파일은 맥북에 그대로 있다.
  - 맥북의 이전 목록(`deploy/shared-demo/versions.json`)에도 다섯 항목이 그대로 있다.
  - 맥미니에는 두 연구 GGUF를 복사한 적이 없다.
- **목록에서 뺀 이유는 두 가지다.**
  - 이관 때 맥미니 메모리를 줄이려고 기본 모델과 독해기만 남겼다(SITE-MIGRATION-019).
  - 두 연구 모델은 C 평가에서 원본 대비 회귀로 판정돼 이미 "사용 비권장"으로 표시돼 있었다.
  - C-028은 수리판에 대해 "회귀를 표시하고 수리 실험판 게시는 유지"를 권고했다. 그러니 **사이트에서 뺀 결정 자체는 C 권고가 아니라 이관 때 메모리를 위해 내린 결정**이다.
- **C-028(이번 라운드) 구별:** C-028은 후보 `ff59b93`의 **검색·라우팅 수정 통과**다. 모델 가중치 개선이 아니다.
  - 위 두 연구 모델의 평가는 이전 라운드의 C-023(pilot)과 C-028(수리판, 2026-10-30 평가 보고 `reports/coordination/C-028.md`)이다.
  - 같은 번호 계열이라 헷갈리기 쉽다. 이 보고에서 "C-028 평가"는 수리판 평가 보고를 가리킨다.

**하지 않은 것 (지시대로)**
- 모델 파일 삭제·복원·이동·복사, 모델 실행, 서비스 재시작, 배포, 설정 변경: 0
- GGUF 재해시: 0. 기존 기록(versions 파일의 sha, C 보고의 일치 확인, 학습 manifest)과 stat(크기·시각·inode)만 썼다.
- 맥미니: SSH 읽기 전용(설정 파일, `ls`, `stat`, `ps`, `vm_stat`, `sysctl`)만 했다.
- 단순 재현 검사(사이트 목록 확인): 하지 않았다. 사용자 관측을 사실로 둔다.
- AWS 0. 비밀값·개인정보 출력 0. proxy secret, 터널 자격 증명, 접속 주소는 보고에 적지 않았다.

## 1. 모델별 현황

| exact ID | 명칭(사이트 label) | 지금 상태 | 파일 (맥북, stat) | 기록된 sha256 | 현재 원본과의 관계 | 제외·평가 근거 |
|---|---|---|---|---|---|---|
| `trid-assistant-qwen3-1.7b` | TRID 기본 · Qwen3-1.7B 원본 (업무 특화 학습 없음) | **노출(기본)** | 맥미니 `repo/runs/open-weight-base/Qwen3-1.7B-Q8_0.gguf` 1,834,426,016 B (맥북에도 같은 크기) | `061b54da…` (versions 파일) | 원본 자체: 공식 Qwen/Qwen3-1.7B-GGUF Q8_0, rev `90862c4b` | — |
| `trid-v0.1-reader` | trid v0.1 추출 독해기 | **노출** | 맥미니 `repo/runs/reader-factorial-01-cpt-v1-v2-v4/reader.safetensors` 366,735,756 B | `90ccb60f…` | 자체 91M 독해기, LLM과 무관 | 업로드 문서 분할 토크나이저로 필수(코드가 요구) |
| `baseline-qwen3-1.7b` | 기준선: Qwen3-1.7B 원본 (미특화, trid 아님) | **숨김**(맥미니 목록에 없음, 요청 시 422) | 기본과 **같은 파일** | `061b54da…` (기본과 같음) | 기본과 같은 가중치. 디코딩만 다름(`grammar_instructed`, 기본은 `case_system_256`) | 중복 기준선이라 목록에서 뺐다(SITE-MIGRATION-019). C 회귀 판정 대상이 아니다 |
| `trid-qwen3-1.7b-pilot-sft-001` | TRID 1.7B · 실험판 (특화 시도 · 회귀) | **숨김**(맥미니에 파일 없음, 요청 시 422) | `runs/pilot-sft-001-a-qwen3-1p7b/pilot-sft-001-a-qwen3-1p7b.q8_0.gguf` 1,834,426,208 B, 2026-10-30 13:42, 링크 1. 같은 폴더에 `train/{adapter,merged}`, `DONE.json`(complete 15단계, manifest `8f0c41cd…`, merged `82f09136…`) | `0c78b0c2…` (versions 파일, C-022가 실제 GGUF hash 일치 확인) | 원본 HF rev `70d244cc`에 LoRA r16(C017 학습 사례 57건 × 2 epoch)을 병합하고 로컬 Q8_0 변환. **공식 GGUF `061b54da`와 다른 변환 계열**(비교 원본은 localconv `f2ba848c…`) | **C-023 회귀:** pilot_dev 18문항 결론 정확 원본 9 → 2, 원본만 정답 7 · 튜닝본만 0(p≈0.016). 해석례 과제 5문항 중 1(원본 5) |
| `trid-qwen3-1.7b-model-repair-002` | TRID 1.7B · 수리 실험판 (평가 완료 · 원본 대비 회귀 · 사용 비권장) | **숨김**(맥미니에 파일 없음, 요청 시 422) | `runs/model-repair-002-qwen3-1p7b/model-repair-002-qwen3-1p7b.q8_0.gguf` 1,834,426,208 B, 2026-10-30 21:30, 링크 2(blind 평가 패키지 사본과 하드링크, 바이트 동일·추가 용량 0). 같은 폴더에 `train/`, 재저장 원본 GGUF | `21715b4f…` (versions 파일, C-028 보고에 C 계산값으로 기록) | 원본 HF rev `70d244cc`에 LoRA r16(B case-training-view-v1r1 34건 × 2 epoch)을 병합. 비교 원본은 재저장본(텐서 바이트가 localconv `f2ba848c`와 같음) | **C-028(수리판 평가) 회귀:** case 50문항 결론 정확 31 → 22(원본만 11 · 수리본만 2, p≈0.023), 256토큰 안 EOS 32 → 3/50, 값 계약 grammar 98 → 14, 과보류 128/228. "성능 향상 아님", 게시는 회귀 표시와 함께 유지 권고 |

**언제 사라졌나**
- 맥북 원점(`deploy/shared-demo/versions.json`)은 다섯 항목을 서빙했다.
- 맥북의 `.shared-demo/llama-trid-qwen3-1.7b-{pilot-sft-001,model-repair-002}.log` 최종 기록은 **2026-11-01 22:08**이다. 원점이 맥미니로 바뀐 시각과 같다.
- 그 뒤 원점은 맥미니(`6a5eb37`, 두 항목 목록)다.

**맥미니 실제 설정 (읽기 전용 확인)**
- `DEPLOYED_FROM_COMMIT`: `6a5eb37 (branch site-memory-022)`
- LaunchAgent `co.tridatum.trid.site`:
  - `TRID_DEMO_VERSIONS=…/repo/deploy/macmini-site/versions.json`
  - 그 파일 sha `dcfa42fd…`는 저장소 파일과 SITE-MIGRATION-019 기록과 같다.
  - `LLAMA_ARG_DEVICE=none`, `LLAMA_ARG_N_GPU_LAYERS=0`
- 같은 폴더의 `versions-stage1.json`(sha `8e3415b3…`)도 같은 두 ID다.
- 맥미니 `repo/runs`에는 `open-weight-base`, `reader-factorial-01-cpt-v1-v2-v4`, `tools`, `workspace-index`만 있다. 연구 GGUF 폴더는 없다.

**사이트에 올린 적 없는 맥북 연구 파일 (참고, stat만)**
- 사이트 버전 이력(첫 버전 목록 `4ac68ee`부터)에 없는 파일들이다.
  - `trid-sft-01/02`, `trid-cpt-sft-01/02`(각 2,225,613,728 B)
  - `open-weight-lora-kanana-01`, `open-weight-lora-qwen3-01`
  - 비교용 `qwen2.5-7b`, `gemma3-4b`, `gemma-4-E4B`
- 모두 맥북에 그대로 있다. 이번 "사라짐"과는 관계없다.
- 맥북 디스크 여유는 352 GB다.

## 2. 연구 모델을 기본 추천과 분리해 다시 체험하게 하려면 (제안, 구현하지 않음)

**지금 서빙 구조 (`6a5eb37` `State.__init__`)**
- 목록의 llama 버전마다 llama-server를 **시작할 때 모두 띄운다.**
- 같은 (GGUF, 서버, ctx)는 서버 하나를 같이 쓴다.
- 시작마다 GGUF 전체를 다시 해시해(`digest`) sha를 확인한다.
- 화면은 `label` 그대로 하나의 `<select>`에 나열하고, 카드에 평가와 한계를 보인다.

**필요한 최소 변경**

1. **목록:** 맥미니 versions 파일에 두 연구 항목을 다시 넣는다. 중복 기준선은 넣지 않는다. 새 필드 `"research": true`를 단다.
   - label, status, evaluation(C-023 / C-028 수치), limits는 맥북 파일에 있는 문구를 그대로 쓴다.
   - default는 원본 그대로 둔다.
2. **UI (서버 렌더 몇 줄 + 카드):** `research` 항목은 `<optgroup label="연구 모델 · 원본 대비 회귀 확인 · 업무에 쓰지 마세요">`로 기본 목록과 분리한다.
   - 선택하면 카드 첫 줄과 각 답변 머리에 "회귀 판정된 연구 모델" 표시를 붙인다.
   - 기본 추천·자동 선택(`defaultModel`)에는 쓰지 않는다.
3. **서빙 (필수, 메모리 때문):** 연구 버전은 시작할 때 띄우지 않는다.
   - **첫 요청 때 하나만 띄우는 지연 로딩 단일 슬롯**으로 한다. 다른 연구 버전을 고르면 앞의 것을 내리고 바꾼다.
   - 유휴 10분이면 내린다.
   - 시작 전 메모리 확인을 둔다(압박 수준 ≥2이거나 여유가 3 GB 미만이면 거부). 거부할 때는 "지금은 연구 모델을 띄울 메모리 여유가 없습니다"로 답하고 기본 모델로 대신 답하지 않는다.
4. **파일:** 두 GGUF(약 3.7 GB)를 맥미니에 복사한다. 맥미니 디스크 여유는 62 GB다.
   - 매 시작 전체 재해시를 피해야 한다. 크기·mtime·inode가 기록과 같으면 기록된 sha를 쓰고, 다르면 거부한다.
   - 처음 한 번은 해시해 기록한다. 기존 sha `0c78b0c2…`, `21715b4f…`와 같아야 한다.
5. **평가 조건 표시:** 연구 판정은 C 평가 조건에서 나왔다(CPU 빌드, case_system 256토큰). 맥미니 CPU 서빙과 조건이 같은지 카드에 적는다.

**맥미니 RAM 위험 (실측값)**
- 맥미니 상태:
  - 16 GiB(Mac16,10), 압박 수준 1(정상), memory_pressure 여유 56%
  - **swap 4.0 / 5.1 GB 사용 중**
  - 운영 llama-server RSS **2.32 GiB**(1.7B Q8, CPU, ctx 4096)
  - 다른 서비스: Docker 백엔드 1.9 GB, 다른 터널·ollama 등이 같이 돈다.
  - 이관 때 warm-up에서 30초 압박 경고가 난 이력이 있다(SITE-MIGRATION-019).
- **시작할 때 모두 띄우면(지금 구조 그대로 목록만 늘리면):** 같은 크기의 서버 2개가 늘어 **+약 4.6 GiB가 상주**한다. swap이 이미 4 GB이고 가용 여유가 작아, 지속적인 압박과 응답 지연, 다른 서비스 영향 위험이 크다. **권하지 않는다.**
- **지연 단일 슬롯:** 연구 모델을 쓸 때만 +약 2.3 GiB, 동시에 하나만이다.
  - 첫 응답 전에 GGUF 매핑과 준비로 수 초에서 수십 초가 더 든다[추정, 미측정].
  - 쓰는 동안 압박 2로 올라갈 수 있다. 3번의 메모리 확인과 유휴 해제가 필요하다.
- **ctx를 줄이는 방법(4096 → 2048, KV 448 → 약 224 MiB):** 평가 조건과 달라진다. 권하지 않는다.
- **[추정]** 위 수치는 운영 llama-server 실측 RSS와 같은 크기·구조의 GGUF라는 점에 근거한 추정이다. 연구 GGUF를 맥미니에서 띄워 잰 값이 아니다.

**결정이 필요한 것 (사용자·총괄)**
- 회귀 판정 모델을 사이트에 다시 보일지
- 보인다면 위 지연 단일 슬롯과 메모리 확인을 구현할지
- 아직 강제 복원 지시가 아니므로 이번에는 아무것도 바꾸지 않았다.

## 3. 범위 밖 (참고)
- "2 GB에서 늘리는 방법"(수집 저장 한도)은 B의 CPU 수집과 S3 전환 작업으로 넘어갔다. A는 다루지 않았다.
- 후보 `ff59b93`은 C 최종 검증 중이라 동결 상태다. 운영은 `6a5eb37` 그대로다.

SUPERVISION_REPORT_READY MODEL-RESUME-A-029
