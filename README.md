<h1 align="center">trid</h1>

<p align="center"><b>공공기관·공기업 내부업무를 위한 폐쇄망 한국어 sLLM</b><br>
공개 가중치 소형 모델에 검증된 업무 데이터를 추가 학습하고, 근거에 기반한 정확한 답변을 우선합니다.</p>

<p align="center">
<a href="#개발-방향과-현재-상태">개발 방향</a> · <a href="#평가">평가</a> · <a href="#개발과-git-관리">Git 관리</a> ·
<a href="docs/SERVICE.md">실행 안내</a> · <a href="docs/SECURITY.md">보안</a>
</p>

## 소개

trid는 대한민국 공공기관·공기업의 **내부 직원**이 규정 해석, 예산·회계·계약, 인사·복무·여비 등 업무를 수행하도록 돕는 소형 언어모델과 폐쇄망 서비스 개발 프로젝트입니다. 민원인용 일반 상담 서비스가 아니라, 근거의 적용 대상·조건·예외·시점·단위를 구분해 답하는 것을 목표로 합니다.

## 개발 방향과 현재 상태

**현재 개발 목표 (2026-11-04 사용자 정정):** 제품 주경로는 **Qwen 사전학습 가중치를 이어 학습한 공공기관·공기업 업무 특화 trid 모델**입니다. 검색 결과를 붙여 전문성을 대신하는 RAG 구축이 아니라, 모델 가중치 자체의 전문성을 개발합니다.
- 기반 계열은 Qwen으로 고정합니다. 정확한 크기·revision·adapter는 실행별로 명시하며, 과거 다른 모델의 점수를 현재 trid 점수로 대체하지 않습니다.
- 검증된 업무 자료로 추가 학습하고, 본문·검색 결과를 주지 않는 동일 조건에서 원본 Qwen 대비 전문성과 기본 대화 능력의 변화를 확인합니다. 기존 LoRA/SFT 경로를 우선 재사용합니다.
- 배포하는 버전에는 기반 모델명, 라이선스, 버전, 제한사항을 표시합니다.
- trid를 "독자 기초모델"로 부르지 않습니다.

**이전 자체 모델 실험:** 무작위 초기화부터 학습한 91M·390M 실험은 보존하고 비교 자료로만 씁니다.
- 자체 기초모델의 추가 scratch 학습·평가·환경 준비는 현재 개발 범위에서 제외합니다. 과거 실행 승인이나 미완료 작업을 근거로 재개하지 않습니다.
- 아래 구조·학습 절은 이 자체 모델 실험의 기록입니다.

### 개발 흐름

1. **원본 기준선:** 선택한 Qwen 원본과 특화 후보를 동일한 질문·히스토리·템플릿·생성 조건에서 비교합니다. 검색·본문 제공 독해와 검색 없는 모델 자체 전문성은 별도 평가하며, 파라미터 수만으로 선정하지 않습니다.
2. **데이터 수집·검증:** CPU와 네트워크로 공개 규정·실무 지침·공식 질의회신·서식·감사 사례를 확보합니다. 권리·개인정보·시점·추출 품질·중복·평가 데이터 겹침을 확인한 묶음부터 인계합니다. 수집 자체에 GPU는 필요하지 않습니다.
3. **가중치 업무 특화:** 기관·적용 시점·조건·예외가 명확하고 원천으로 정답이 검증된 학습 사례를 사용합니다. RAG 사례에서 본문이나 “확인 불가” 표시만 제거해 사실 학습 데이터로 바꾸지 않습니다. 원본 대비 전문성·자연스러움·지시 이행·대화 능력 회귀를 함께 확인합니다.
4. **실험 배포:** 완성된 모델 버전마다 [테스트 사이트](https://test.tridatum.co)에 배포하고 실제 질문·응답을 확인하는 절차를 연결합니다. 중간 체크포인트 전체를 배포하지 않으며, 이전 버전 복구 경로를 유지합니다.

업무 용어·절차·규정의 적용 범위와 불확실성 처리를 가중치에 학습합니다. 모델이 모든 개정 규정과 기관별 실시간 값을 기억한다고 가정하지 않으며, 알 수 없는 시점·잔액·집행액·결재 상태는 지어내지 않아야 합니다. 기존 검색 서비스는 별도 운영 이력으로 보존하되 이번 모델 특화의 구현이나 성능 증거로 대신하지 않습니다. 미확보 기관 데이터를 만들어 학습시키지 않고, 수집량과 학습 적격량을 별도로 집계합니다.

아래 `runs/`·`data/` 링크는 별도 보관한 로컬 실행 증거·입력의 위치입니다. Git에서 제외되는 자료이며 이 저장소의 clone에 포함되거나 링크가 열릴 것을 보장하지 않습니다.
2026-11-11에는 공개 대상 소스만 구성한 별도 checkout에서 Qwen 관련11개 테스트 모듈의 **328개 테스트 통과**와 native·서빙 적격성 CLI의 `--help` exit0를 확인했습니다. Native·서빙 테스트는 공개용 소스 fixture5개와 임시 합성 메타데이터를 사용하며, 제외된 `runs/`·비공개 DEV 자료에 의존하지 않습니다. **사전학습 모델 실행·품질·배포 검증은 아닙니다.**

| 구분 | 현재 상태 |
|---|---|
| 기반 모델 | Qwen 계열로 고정. 정확한 base revision·adapter·생성 계약을 실행별로 결박하고 검색 없는 원본 대비 평가로 선정 |
| 실험 모델 이름 | **TRID 1.7B · 실험판** = Qwen/Qwen3-1.7B(rev `70d244cc`, Apache-2.0) + LoRA 특화(pilot-sft-001, 학습 사례 57건 × 2 epoch, 15단계). 서비스 표시명은 TRID이고, 기반 모델과 라이선스는 사이트 버전 상세와 이 문서에 공개합니다. 독자 기초모델이 아닙니다. **TRID 1.7B · 수리 실험판**(MODEL-REPAIR-002, 2026-10-30)은 같은 원본과 같은 학습 설정을 쓰되, 출처 번호·256토큰 계약을 맞춘 B 학습 사례 34건(문장별 AI 검토)으로 맥북 Apple GPU에서 다시 학습한 소규모 진단 모델입니다. 추가 AWS 비용은 0원이고 학습은 513초 걸렸습니다. **C-028 blind 평가 완료: 원본 대비 회귀, 사용 비권장**(case 결론 정확 31 → 22, EOS 종료 32/50 → 3/50, RAG310 grammar 98 → 14, 과보류 128/228; `reports/coordination/C-028.md`). 두 실험판과 "기준선: Qwen3-1.7B 원본" 선택지는 2026-11-01부터 사이트에서 제공하지 않으며, 모델 파일은 기록용으로 맥북에 남깁니다. 기본값은 원본입니다. |
| 특화 학습 데이터 | 공식 자료와 근거 기반 사례를 정제·검토 중. 후보 전체가 학습 승인된 것은 아님 |
| 검색 없는 특화 개발 실험 | `qwen-specialization-002`: 적격 법령해석 사례 42건 × 2epoch, 21step 실제 학습 완료. 학습 전 고정한 24시나리오를 원본과 동일 조건으로 비교하고 두 채점자가 블라인드 판정했습니다. 법령해석 일부 개선에도 불확실성 처리의 날조와 일반대화 퇴행으로 **배포 평가 FAIL**입니다. 실제 서비스는 6GiB 조건에서 첫 요청 응답 후 두 번째 요청 중 RSS 제한으로 종료되어 연속 요청 안정성도 FAIL입니다. 운영 반영하지 않습니다. 기존 RAG 평가·native 자체 모델 점수와 합산하지 않습니다. [실행·판정 근거](reports/coordination/CHAT-ROOT-C-004.md#87-고정된-비검색-비교-완료-특화-후보-배포-평가-fail) |
| 특화 모델 품질 | `qwen-specialization-012`는 선호91쌍·23update **학습 완료**, checkpoint8/16/23 저장과 실제 F32 native adapter3개 변환, 고정 DEV108문항-arm/138turn **평가 완료**입니다. 그러나 세 후보 모두 고정 품질 기준에서 **불합격**, 선택 결과는 **NO_CANDIDATE**입니다. 해당 순서로 중대 오류4/5/3건(허용0), 일반능력 보존 실패3/5/6건/12(허용최대1)이며, 최종24는 열지 않았습니다. **학습·체크포인트 생성·평가 완료는 품질 합격이나 배포가 아닙니다.** [기존 상세 결과](reports/coordination/CHAT-ROOT-C-004.md#113-dpo012-독립-판정과-정답-보존-실패) |
| 현재 개발 방식 | `qwen-specialization-013`: 동일 초기값·선호91쌍에 원 TRAIN288건의 정답 CE replay를 결합합니다. 길이별 microbatch와 유효한 행·토큰 위치만 계산하는 packed-head 엔진을 사용하고, 명시적 CUDA 진입점·별도 CUDA 캐시 경로를 분리했습니다. 선호 sequence SUM·replay sequence별 token MEAN·FP32·원본 보존 조건을 유지합니다. 실제 전체 학습 속도·품질 개선은 아직 검증하지 않았습니다. |
| 검증 완료 | 기존 Mac·Linux 참조 캐시91쌍은 보존했습니다. 새 엔진의 관련 회귀 **48개**를 검증했고, 작은 합성 Q8 head·LoRA 구성에서 누적 손실·기울기·optimizer 한 단계·역전파 head 작업 한도를 실행 확인했습니다. 전체 logits 기준 대비 최대 LoRA 기울기 절대차는 **7.15×10⁻⁷**입니다. 이 합성 검증은 실제 사전학습 가중치나 TRAIN/DEV 학습과 구분합니다. cloud lifecycle·native pipeline의 최종305초 admission 수정을 포함한 **통합 회귀72건도 통과**했습니다. |
| 이전 packed 실행 결과 | `qwen013-packed-20261109070127`은 **새 packed 참조 캐시91쌍 수락 후, 2-update 스모크 미수락으로 종료**했습니다. 캐시6개 자식 실행 합계는443.53014726초이며 전체 workflow 시간이 아닙니다. native runtime 빌드는 완료했습니다. replay288행은 입력 적격 판정이며 전체 소비량이 아닙니다. 최초4쌍의 정책/참조 최대 차이는0.0(허용0.001)이었습니다. 로그에서 optimizer update1회 완료·선호4쌍·replay12행을 확인했습니다. 이때 trainer 누적 경과157.1초는 순수 update 소요시간이 아닙니다. **이 이전 실행의 정확한 총 update 수는 미확정**입니다. 1회는 확인했고 2회째는 실행됐을 가능성이 있으나, 해당 로그·체크포인트가 validation 뒤에 기록되어 관측하지 못했습니다. |
| 이전 packed 실행의 중단·안전 조건 | supervisor는175.145661851초에 `owner_signal_SIGTERM`을 기록했습니다. 관측 RSS4,416,978,944bytes(약4.11GiB)이며 swap·OOM·PSI full 증가는 없었습니다. **2-update 스모크 수락·체크포인트·본학습·native 생성·새 후보·DEV 평가는 없습니다.** 스모크 수락 이후 native 추론·900초 적합 판정·본학습·고정 DEV 독립 판정이 필요합니다. 진단180초(systemd175+5)·RSS6GiB·Linux MemAvailable 진입≥8GiB/실행≥6GiB·PSI 증가 중단·swap0 조건과 원본 가중치·기존 캐시·TRAIN288건·품질 기준·최종24 봉인·운영을 유지합니다. 이전 엔진 실행과는 [별도 기록](reports/coordination/CHAT-ROOT-C-004.md#122-별도-linux-캐시91쌍-완료와-학습-스모크-미수락)입니다. |
| 최근 traced 실행 결과 | `qwen013-traced-20261109203337`은 **SMOKE_FAILED_NO_CANDIDATE**로 종료했습니다. 기존 캐시91쌍을 재생성 없이 재사용했고 최초4쌍 최대 오차는0입니다. optimizer1회(선호4쌍·replay12행) 완료를 기록했으며 update2의 마지막 관측은 두 번째 쌍 backward입니다. `owner_signal_SIGTERM`으로 중단됐으며 감독기 총 경과는175.368533초입니다. **2-update 수락·완료된 checkpoint·유효한900초 예측·본학습·native 생성·DEV 평가·새 후보는 없습니다.** 승인한1회 실행은 소비됐고 자동 재시도하지 않습니다. 세부 계산 시간·메모리·회수 한계는 [학습 단계별 중단 진단](#학습-단계별-중단-진단)에 구분해 기록합니다. |
| 최신 GPU 실행 | `qwen013-cuda-20261111030853`: **23/23update·선호91쌍·replay288행의 학습 계산과 저장은 완료했지만, host memory PSI full 증가로 감독 프로세스가 거부되어 `FAILED_NO_RETRY`**입니다. checkpoint8/16/23·최종 adapter는 진단용으로만 보존합니다. **수락된 본학습·native DEV·후보 선정·서빙·게시 결과는 없고 final24는 봉인 상태입니다.** [실행 결과·정리](#frozen-head-vjp-실험과-cuda-실행). |
| 기존 모델·테스트 서비스 | 테스트 사이트 기본 엔진은 **Qwen3-1.7B 원본(업무 특화 학습 없음)** 이다(TRID-CONVERSATION-001, 2026-10-30). 인사·사용법·글쓰기 같은 일반 대화는 원본의 기본 chat template으로 답하고, 최근 대화를 기억하며 근거 검색을 하지 않는다. 사용자가 말하지 않은 숫자·기한·조문 문장은 숨긴다. 업무 규정 질문은 공개 법령정보·공공데이터 FAQ·편람 14,672건을 검색한다. 검색은 bge-m3 임베딩과 BM25를 합친 hybrid이며, 법령·규정을 민원 게시판보다 우선하고, ALIO 공식 기관 목록으로 기관 별칭을 연결한다. 모델을 부르기 전에 근거 적합성을 검사한다(HYBRID-RAG-001, TRID-ANSWER-002). B API 스냅샷 신규 5건을 포함한다. 새 API 수집과 S3 업로드, 정형 통계 값은 아직 0건이다. **2026-11-01 22:08 KST부터 사이트는 맥미니 CPU(llama.cpp GPU offload 0)에서 고정 named tunnel로 서비스하며, 기본 원본과 v0.1 추출 독해기 두 버전만 고를 수 있다**(SITE-MEMORY-023, `docs/SHARING.md`). 운영 코드는 2026-11-02 05:37 UTC부터 브랜치 `model-resume-030-followup`의 `2b8ba92`이며 main에는 아직 병합하지 않았다(A032). 검색·후속 조건 수정 배포이며 일반 대화 품질 개선은 아니다. 직전 `6a5eb37`과 `rollback-9256df4`는 보존했다. 기준선과 TRID 1.7B 실험판 두 개는 사이트에서 뺐다. 실험판 pilot-sft-001은 C-023 블라인드 진단(18문항, AI 1인 채점)에서 원본 대비 회귀(결론 정확 9 → 2, 응답 행동 적절 16 → 4)했고, model-repair-002도 C-028에서 회귀했다. 모델 파일은 맥북에 보존한다. CPU 실행이라 기관 질문 답은 약 17–21초 걸린다. 원문과 글자가 일치하는 문장만 인용으로 표시하고(답인지는 원문에서 확인), 모델 해설과 일반 대화는 "검증 안 됨"으로 표시한다 |
| 일상 대화 상태 | **미해결.** 라우팅은 좋아졌지만(일상 19/20이 대화로 감) 내용 품질은 합격이 아니다. 조건 갱신, 다듬기, 날조 실패가 남아 있다. 대안 3종(Qwen2.5-7B, gemma3:4b, 설정 변형)은 내부 검토만 했고 미합격이며 운영 모델이 아니다. 기본 원본 1.7B는 현상 유지다(`docs/SHARING.md` 알려진 한계) |
| 원격 실행·회수 | **이전 packed 실행:** 추가상한$3·최대2시간·누적상한$8로 승인한 instance `i-03b52e35e613dd584`는 자체 shutdown으로 종료됐고, **전용 EBS0·종료 예약 없음·해당 S3 객체5개 삭제·두 run prefix 비움·CloudShell 임시 경로 정리**를 확인했습니다. 보존 파일70개는 모두 독립 SHA 검증했으나, continuation 이전 export-manifest 부재로 `export_complete=false`라는 완전성 한계는 유지합니다. 인증한28항목 `executed-sources.tar`와 controller/pricing 감사 자료·승인 및 실행 영수증은 보존하고 로컬 중복 `inputs.tar`만 제거했습니다. 원본 가중치·데이터·캐시는 불변입니다. 당시 보수적 추가 비용 상한 **약$2.00**·누적 **약$6.86**은 실제 청구액이 아닙니다. [이전 실행 결과](runs/qwen-specialization-013/qwen013-packed-20261109070127/run-outcome.json) · [이전 저장소 정리](runs/qwen-specialization-013/qwen013-packed-20261109070127/storage-cleanup-receipt.json) · [이전 실행 소스](runs/qwen-specialization-013/qwen013-packed-20261109070127/executed-sources.tar). **최근 traced 실행:** 이후 승인한 **추가$3·최대2시간·누적상한$10의 새1회 실행은 소비됐습니다.** 기존 Chrome 인증으로 실행했고 instance `i-0215cbe013f8b3eae`는 종료, EBS0·종료 예약 삭제·정확한 S3객체5개 및 소유 CloudShell 임시 경로 삭제를 확인했습니다. durable HOME intent는 보존했습니다. 보존35항목 해시를 모두 검증했지만 manifest 부재로 `export_complete=false`는 그대로입니다. 현재 보수적 승인 범위 내 상한은 **추가$1.999099837·누적$8.859892561이며 실제 청구액이 아닙니다.** 공개 사이트·모델 registry는 변경하지 않았고 새 유료 실행에는 새 승인이 필요합니다. [현재 회수 검증](runs/qwen-specialization-013/qwen013-traced-20261109203337/output-recovery-verification.json) · [controller 감사](runs/qwen-specialization-013/qwen013-traced-20261109203337/controller-audit-verification.json) · [현재 저장소 정리](runs/qwen-specialization-013/qwen013-traced-20261109203337/storage-cleanup-receipt.json) |
| 최종 검증 | 독립 공공기관 업무 벤치마크 준비 중. 현재 결과는 개발 평가이며 제품 정확도나 보안 적합성을 보증하지 않음 |

외부 API 없이 기관 내부에서 가중치와 검색·추론 구성요소를 실행하는 것이 목표입니다. 모델뿐 아니라 설치 의존성, 외부 통신, 권한, 로그와 기관별 보안 요건도 별도로 검증해야 합니다.

### 공개 가중치 LoRA 입력 계약

`scripts/train_open_weight_lora.py`는 manifest에 고정된 파일 해시와 선택 split을 확인합니다.
`prompt_answer` 행의 선택적 `history`는 `[{"user": "이전 질문", "assistant": "이전 답변"}, ...]`이며,
시스템 지시 → 이전 대화 쌍 → 현재 `prompt` 순서로 기반 모델의 chat template에 넣습니다.
이전 assistant 답변을 포함한 입력 전체와 배치 padding은 `-100`으로 마스킹하고, 현재 `answer`와 기존 EOS만 학습합니다.

- `history` 생략 또는 `[]`는 기존 단일 턴 동작입니다. `null`, 잘못된 역할/타입, 빈 내용 및 기존 스키마의 예약 제어 토큰은 파일·행 번호·ID와 함께 가중치 로딩 전에 거부합니다.
- `case`의 비어 있지 않은 history는 지원하지 않아 명시적으로 거부합니다. 기존 출처·답변 예산 검사는 유지합니다.
- 문맥 한도를 넘으면 행 전체를 제외·집계합니다. 과거 대화만 잘라 맞추지 않습니다.
- `scripts/audit_case_training_contract.py`도 trainer와 같은 메시지 조립을 사용합니다. 데이터/토크나이저 유틸리티를 import할 때 PEFT는 필요 없으며, PEFT·가중치 로더는 학습 진입점에서만 가져옵니다.
- 모델 가중치 없이 입력·loss 마스크만 확인하는 회귀 검증: `python -m unittest discover -s tests -p 'test_train_open_weight_lora.py' -k HistoryContractTests -v` (기존 torch/tokenizers 환경에 transformers 필요, 다운로드 없음). 입력 계약 클래스만 실행하며, 같은 파일의 PEFT·Q8 등 전체 학습 엔진 검증은 포함하지 않습니다.
- 입력 계약 수정 당시 확인한 범위: 로컬 Qwen3 토크나이저로 실제 loader/encoder를 실행해 2쌍의 대화 이력 유지, 과거 assistant 마스킹, 현재 답변+EOS 학습 대상, 단일 턴·case 호환성을 확인했습니다. 모델 가중치 로딩이나 학습은 실행하지 않았습니다.

이는 학습 입력 유실 수정이며 새 학습 실행·데이터 승인·운영 대화 품질 개선을 뜻하지 않습니다.
학습할 기반 모델·검토된 데이터·평가 분리와 실행 승인은 별도로 확정해야 합니다.

### Packed-head 학습 엔진

`train_open_weight_lora.chunked_answer_loss`는 backbone의 문맥 계산을 유지하면서, 각 행에서 실제로 감독되는 토큰만 모아 vocabulary head를 계산합니다. 기존 호출 인터페이스와 손실의 행별 소유권은 유지하며, logits 작업 공간은 기존과 같은 `batch_size × LOGIT_CHUNK` 토큰 이내입니다. 모두 마스킹된 SFT 입력은 기존 head 계산 경로를 유지하여 trainable head에도 0 기울기가 전달됩니다.

- **캐시 안전성:** head 방식·라이브러리 버전·device/dtype·scorer/helper/Q8 소스 해시가 모두 같아야 참조 캐시를 수락합니다. 오래된 캐시를 자동 변환하거나 다시 계산하지 않습니다. 초기 점수 허용오차 `1e-3`도 바꾸지 않았습니다. 기존 원격 계획의 소스 pin은 과거 실행 기록이며 새 엔진 실행 승인으로 대체되지 않습니다.
- **실행 결박·평가 보존:** `engine.answer_head=packed_supervised_tokens_v1` 및 소스3개의 SHA를 새 캐시에 결박합니다. 예전 캐시는 덮어쓰거나 재결박하지 않고 현재 엔진에서 거부합니다. 고정 native 평가기 소스는 과거 도구 쓰기 기록에서 복구해 원래 SHA와 일치하며, 현재 run의 `runs/qwen-specialization-013/qwen013-packed-20261109070127/native-source-recovery/`에 보존했습니다. 평가 규칙은 바꾸지 않았습니다.
- **실행 검증:** 관련 두 테스트 파일의48개 사례를 검증했습니다. 처음 발생한 선택 환경의 선택적 의존성 누락과 임시 검증 감독기 문제는 기록에 남겼고, 로컬 패키지 캐시를 임시 경로에 설치한 뒤 해당19개 사례를 재검증했습니다. 추가로 과거 union-head 구현이 희소 감독 토큰 검사를 실패하고 새 구현은 통과하는 회귀를 확인했습니다.
- **합성 스모크:** 임의 생성한 hidden32·vocab64·2층 구성, Q8 head와 rank2 LoRA, 선호2쌍·replay5행으로 실제 누적 계산과 optimizer 한 단계를 실행했습니다. 네트워크·원 실험 데이터·기존 모델 가중치 읽기를 차단하고120초·프로세스 그룹 RSS1GiB 한도로 감독했습니다. 원 사전학습 모델을 학습하거나 DEV/final24를 평가한 결과가 아닙니다.
- **성능 근거의 한계:** 앞선 ARM 합성 실험에서는 같은 스레드 설정의 head forward+backward가 기존 대비2.33~6.94배 빨랐습니다. 이는 작은 합성 shape의 head 부분 측정이며 전체 학습 시간·배포 응답 속도·답변 품질 개선 배수가 아닙니다. Q8 direct-out 후보는 대부분 더 느려 채택하지 않았고 Q8 loader는 변경하지 않았습니다.
- **로컬 증거:** `runs/qwen-specialization-013/packed-head-engine-001/{verification.json,synthetic-smoke.json,baseline-sources.tar,implementation-sources.tar}`. ARM 진단 원본은 `runs/qwen-specialization-013/qwen013-kernel-20261109012340/`에 있습니다. 실행 자료는 별도 보관 대상이며 저장소에 포함된다고 가정하지 않습니다.

### Frozen-head VJP 실험과 CUDA 실행

`scripts/qwen_frozen_head_vjp.py`는 **CPU FP32·정확한 frozen Q8 출력 head 구현만 허용하는 별도 실험 경로**입니다. 제한된 logits 작업 공간이 살아 있을 때 토큰별 CE의 단위 hidden 미분을 계산하고, backward에는 hidden 폭의 미분만 보존해 checkpoint의 head 재투영을 피합니다. 기존 CE·Q8 계산과 선호 sequence SUM/replay sequence별 token MEAN을 유지합니다. **1차 미분 전용**이며 autocast·학습 가능한 head·GPU는 지원하지 않습니다.

- **실행 검증:** 관련 테스트 **72개 통과·실패0·skip0**. 임의 생성한 Qwen3의 floating backbone+Q8 head와 rank2 LoRA로 스레드1·8에서 각각 AdamW2회 갱신을 실행했습니다. 기준 대비 최대 LoRA 기울기 절대차는 **5.960464477539063e-08**, 갱신 파라미터 절대차는 **3.3760443329811096e-08**입니다. 두 스모크 모두120초·프로세스 그룹 RSS1GiB 한도에서3초 미만에 종료했고 자식 회수 완료·잔여 그룹 프로세스0을 확인했습니다.
- **Head-only 합성 시간:** 작은 제한된 shape의 Darwin CPU 측정이며 **전체1.7B·Linux 처리속도, 서빙 속도나 품질 개선 근거가 아닙니다. 유효한900초 본학습 예측도 아직 없습니다.** 아래 이전 실제 traced 실행의 실패를 뒤집는 결과가 아닙니다.

| 스레드 | 합성 입력 | 기존 forward+backward (ms) | VJP forward+backward (ms) |
|---:|---|---:|---:|
| 1 | sparse | 2.237625 | 1.428917 |
| 1 | full | 9.460375 | 6.103542 |
| 8 | sparse | 1.569750 | 0.996583 |
| 8 | full | 6.518791 | 3.921833 |

- **보존·적용 경계:** 원래 엔진 소스3개·Linux 캐시6개의 해시는 불변입니다. 이 CPU VJP 실험에서는 production 학습 경로 전환이나 캐시 identity 재라벨링을 하지 않았고, 사전학습 가중치·원 학습/평가 데이터·final24를 읽지 않았습니다. 이 로컬 실험 자체는 유료 실행이나 후보 게시가 아닙니다. **공개 선택지는 기존 원본과 v0.1 추출 독해기 두 개 그대로**이며, 두 기존 선택지가 새 특화 모델의 품질 기준을 통과했다는 뜻은 아닙니다.
- **로컬 근거:** [검증 요약·시간](runs/qwen-specialization-013/frozen-head-vjp-001/verification.json) · [72개 테스트](runs/qwen-specialization-013/frozen-head-vjp-001/affected-tests.log) · [스레드1 감독](runs/qwen-specialization-013/frozen-head-vjp-001/generated-smoke-threads1-supervisor.json) · [스레드8 감독](runs/qwen-specialization-013/frozen-head-vjp-001/generated-smoke-threads8-supervisor.json). 상세 수치·실행 소스는 같은 증거 디렉터리에 별도 보존합니다.
- **GPU 조사 근거:** 서울 `g6.xlarge`는 L4·4vCPU·호스트16GiB이며 관측한 Linux On-Demand compute 요율은 **$0.9896/시간**, G/VT quota는48vCPU입니다. 실행 전 pending/running/stopping/stopped를 보수적으로 포함한 사용량은16vCPU였습니다. 기존 중지 GPU3대는 재시작하거나 변경하지 않았습니다. [가격·사양·quota](runs/qwen-specialization-013/frozen-head-vjp-001/gpu-capacity-readonly.json) · [기존 GPU 소유·상태](runs/qwen-specialization-013/frozen-head-vjp-001/existing-gpu-work-readonly.json).
- **승인된 GPU 1회 실행 결과 — 초기 설치 실패:** `qwen013-cuda-20261110035134`는 승인된 `g6.xlarge`1대·1회·최대2시간, 추가$4·누적$13·자동 재시도0 조건으로 실행했지만 **학습 시작 전에 exit100으로 종료**됐습니다. Ubuntu24.04의 APT에 없는 `awscli` 패키지를 요청한 부트스트랩 오류입니다. 패키지 인덱스 다운로드는 성공했고, AMI에 이미 있던 AWS CLI로 실패 증거도 회수했습니다. **CUDA 연산 검증·새 캐시·2-update 스모크·본학습·DEV·새 후보는 실행되지 않았습니다.** 인스턴스 종료·잔여 EBS0·종료 예약 삭제를 확인했고, 해시 검증을 마친 S3객체5개와 소유 임시 경로만 정리했습니다. 부분 export의 `export_complete=false`도 그대로 보존합니다. 준비 당시 보수적 상한인 추가$3.228043289·누적$12.087935850은 실제 청구액이 아닙니다. **승인1회는 소비됐으며 유료 재실행에는 새 승인이 필요합니다.** [실제 결과](runs/qwen-specialization-013/qwen013-cuda-20261110035134/run-outcome.json) · [실패 로그](runs/qwen-specialization-013/qwen013-cuda-20261110035134/worker-output/bootstrap.log) · [회수 검증](runs/qwen-specialization-013/qwen013-cuda-20261110035134/output-recovery-verification.json) · [정리 증거](runs/qwen-specialization-013/qwen013-cuda-20261110035134/storage-cleanup-receipt.json).
- **CUDA 실행 계약·검증:** 명시적 CUDA 진입점과 별도 runtime/source/device 결박 캐시를 사용합니다. 원본 엔진3개·CPU 캐시는 보존하고 **CPU-only VJP는 적용하지 않습니다.** 호스트 RSS와 GPU 전체 사용 메모리는 각각6GiB 한도이며 실제 driver·GPU UUID를 서버에서 확인합니다. 로컬 계약 테스트 **257개**와 전송 묶음 전체 SHA 검증을 통과했습니다. 실제 AMI의 임시 디스크 매핑을 추가 EBS로 오인하던 사전 점검도 수정했고, 추가 EBS·중복 매핑은 계속 거부합니다. 생성 입력 GPU 검증→새 캐시6분할→2-update 스모크→900초 적합성→본학습·고정 DEV 순서로 **통과한 단계만** 진행합니다. 아직 본학습 완료·품질 수락·새 후보 배포 결과는 아닙니다. [검증 기록](runs/qwen-specialization-013/qwen013-cuda-20261110035134/verification.json) · [입력 묶음 검증](runs/qwen-specialization-013/qwen013-cuda-20261110035134/input-bundle-verification.json).
- **설치 오류 수정:** Ubuntu APT에서 `awscli`를 설치하려던 요구를 제거하고, AMI에 설치된 `aws`의 존재와10초 이내 `aws --version` 성공을 APT 작업 전에 요구하도록 수정했습니다. 설치기 fallback이나 APT 오류 무시는 추가하지 않았습니다. 위 소비된 실행의 승인·입력 pin은 다시 쓰지 않으며, 정확한 입력 소스31개와 실제 런처 감사 자료를 별도로 보존했습니다. 설치 수정 자체의 검증과 아래 새 실행의 실제 GPU 결과는 구분합니다.
- **앞선 무료 설치 검증의 범위:** CLI 사전 점검 두 줄은 실제 CloudShell에서 기존 CLI exit0, CLI가 없는 PATH exit1을 확인했습니다. 무료 Ubuntu 컨테이너의 전체 설치 검증은 CloudShell cgroup 오류로 시작 전에 차단됐고, 자원 제한을 해제해 우회하지 않았습니다. 이 무료 검증만으로 전체 설치나 GPU 성공을 주장하지 않았습니다. 해당 검증 자원과 임시 스크립트는 증거 보존 후 정리했습니다. [무료 검증 기록](runs/qwen-specialization-013/qwen013-cuda-20261110035134/bootstrap-fix-verification.json).
- **새 승인 GPU 실행 결과 — 장치 식별 단계 실패:** `qwen013-cuda-20261110135103`는 별도 승인된1회·최대2시간·추가$4·누적 보수적 상한$16 조건으로 실행했습니다. 승인 경계 테스트23개, bootstrap 문법, 실제 입력 조립·전체 묶음 인증과 AWS 사전 점검을 통과했습니다. 실제 GPU 서버에서는 **APT·Python venv·Torch2.9.1+cu128와 지정 의존성 설치, 입력·작업 허가 검증까지 성공**했지만 `GPU UUID/minor discovery mismatch`로 exit1 종료됐습니다. 원 조회의 stdout/stderr가 남지 않아 그 내부 실패 원인은 확정하지 않습니다. **CUDA 연산 검증·새 캐시·학습·DEV·새 후보는 없습니다.** 인스턴스 `i-0c8ad7d4df7d27cf0` 종료·전용 EBS0·종료 예약 삭제를 확인했고, S3객체6개의 실제 바이트 SHA를 검증한 뒤 조건부 삭제와 소유 임시 경로 정리를 완료했습니다. `export_complete=false`와 원본 입력·기존 캐시는 보존합니다. 누적 보수적 상한$15.315979140은 실제 청구액이 아니며, 승인1회는 소비됐습니다. [실제 결과](runs/qwen-specialization-013/qwen013-cuda-20261110135103/run-outcome.json) · [실패 로그](runs/qwen-specialization-013/qwen013-cuda-20261110135103/worker-output/bootstrap.log) · [정리 증거](runs/qwen-specialization-013/qwen013-cuda-20261110135103/storage-cleanup-receipt.json).
- **장치 조회 수정과 검증 한계:** 가정한 CSV 필드 대신 NVIDIA가 문서화한 `nvidia-smi -q -x` XML에서 단일 GPU·정확한 UUID·장치 minor를 검증하며, GPU index를 minor로 간주하거나 fallback하지 않습니다. 실패 시 제한된 stdout/stderr·종료 코드를 보존합니다. 관련 테스트29개를 통과했고, 공식 메타데이터 SHA로 검증한595.112.07 패키지의 실제 CLI와 수정 함수를 무료 CloudShell에서 실행해 driver 부재 exit9의 진단 보존을 확인했습니다. 일반 `--help`는 XML 옵션을 광고했으나 `--help-query-gpu`도 driver 부재로 실패했으므로, 이를 CSV 필드 미지원 증거로 해석하지 않습니다. **이 무료 검증 당시에는 실제 GPU 성공이 미검증이었으며, 아래 별도 승인 실행에서 장치 식별·CUDA 스모크를 확인했습니다.** [수정 검증](runs/qwen-specialization-013/qwen013-cuda-20261110135103/gpu-discovery-fix-verification.json) · [NVIDIA 공식 인터페이스](https://docs.nvidia.com/deploy/nvidia-smi/index.html).
- **이전 승인 GPU 실행 — 스모크 수락 후 본학습 시작 전 실패:** `qwen013-cuda-20261110152923`은 실제 L4 UUID·admission, 생성 입력 CUDA 검증·새 캐시6분할·2-update 스모크, native 변환·TRAIN4건 probe를 수락했습니다. 스모크 감독 시간38.81455544초·호스트 peak RSS1,838,542,848bytes·GPU peak 사용량3,852,468,224bytes를 관측했습니다. 본학습 예측245.136685387초는900초 한도 내지만 **실제 완료 시간이 아닙니다.** pipeline 메타데이터와 controller dispatch 기록이 `full-training-command.json`에서 충돌해 `Unit already dispatched; no retries`로 **본학습 자식 실행 전 종료**됐습니다. supervisor 영수증·본학습·native DEV·후보 게시는 없고 final24도 열지 않았습니다. 품질·성능 개선이나 배포 성공이 아닙니다. [최종 실행 결과](runs/qwen-specialization-013/qwen013-cuda-20261110152923/run-outcome.json) · [단계별 관측](runs/qwen-specialization-013/qwen013-cuda-20261110152923/monitor-002-summary.json).
- **Dispatch 경로 충돌의 로컬 수정:** pipeline 계획은 `full-training-plan.json`, controller의 독점 생성 dispatch 기록은 `full-training-command.json`으로 분리했습니다. CPU의 기존 덮어쓰기도 제거하고 CPU·CUDA 모두 중복 실행을 거부합니다. pipeline33·CPU24·CUDA29로 **86개 테스트 통과** 및 독립 검토를 마쳤습니다. 실제 무해한 자식 프로세스를 CPU·CUDA 경로에서 각각1회 실행해 의도한 exit17, 계획 보존·중복 거부·실패 pipeline 중단을 확인했습니다. **이는 systemd·GPU·수정 후 본학습 성공 검증이 아니며, 이 로컬 검증 당시에는 유료 재실행하지 않았습니다.** 과거 실행 소스·실패 증거는 변경하지 않았습니다. [수정 검증](runs/qwen-specialization-013/qwen013-cuda-20261110152923/dispatch-fix-verification.json) · [실제 자식 프로세스 검증](runs/qwen-specialization-013/qwen013-cuda-20261110152923/dispatch-real-child-smoke.json).
- **이전 실행 회수·정리·승인 경계:** worker98항목·controller 감사26항목의 SHA를 검증했고 `export_complete=true`입니다. 2026-11-10 16:02:38 UTC에 instance `i-04860731c903b8fe5` 종료·전용 EBS0·종료 예약 삭제를 확인했습니다. 16:10:04 UTC에는 **S3객체5개 전체 바이트 SHA 검증 후 조건부 삭제·소유 prefix2개 비움·소유 임시 경로 정리**를 확인했습니다. 검증한 실행·감사 archive는 보존하고 로컬 중복 입력1,961,574,400bytes와 임시 스크립트3개만 제거했으며, 원본 가중치·데이터·캐시는 유지했습니다. 실행 허용 전 독립 종료 예약과 controller 전환 계약29개 테스트를 확인했습니다. 해당 **1대·1회·최대2시간·추가$4·누적$19·자동 재시도0** 승인은 소비됐고 다음 실행은 아래 별도 승인으로 구분합니다. 당시 누적 보수적 상한 **$18.544022428889907**은 실제 청구액이 아닙니다. [종료 자원 상태](runs/qwen-specialization-013/qwen013-cuda-20261110152923/final-resource-state-002.json) · [저장소 정리](runs/qwen-specialization-013/qwen013-cuda-20261110152923/storage-cleanup-receipt.json) · [실행 허용·비용](runs/qwen-specialization-013/qwen013-cuda-20261110152923/release-observation.json) · [controller 계약 테스트](runs/qwen-specialization-013/qwen013-cuda-20261110152923/controller-contract-tests.log).
- **이전 별도 승인 실행 — 본학습13/23update 후 안전 중단:** `qwen013-cuda-20261111002536`은 실제 본학습을 시작했지만 host memory PSI full 누적값44,507→45,170µs(**+663µs=0.663ms**)를 관측해 엄격한 증가 금지 규칙대로 중단했습니다. 본학습 감독 시간131.311966초·peak RSS1,834,897,408bytes·GPU peak 사용량4,324,327,424bytes이며 중단 시 MemAvailable은13,262,176,256bytes였습니다. cgroup high/max/OOM·swap 증가는0으로, **원인 코드·설정 결함이 입증된 것은 아닙니다.** checkpoint8은 진단용으로만 보존하며16/23·native DEV·후보 선정·게시 결과는 없습니다. **본학습 완료·품질·배포 성공이 아닙니다.** [학습 로그](runs/qwen-specialization-013/qwen013-cuda-20261111002536/worker/output/full-training.log) · [감독·중단 증거](runs/qwen-specialization-013/qwen013-cuda-20261111002536/worker/output/full-training-supervisor.json).
- **이전 PSI 중단 실행 회수·정리·승인 경계:** worker106항목·controller27항목의 SHA를 검증했습니다. `export_complete=true`는 회수 완전성이고 **worker exit1인 실패 workflow의 성공 판정이 아닙니다.** 2026-11-11 01:01:52 UTC에 instance `i-03b19ddba7cc2bb1e` 종료·전용 EBS0·종료 예약 삭제를 확인했습니다. 01:09:11 UTC에는 **소유 S3객체5개 전체 바이트 SHA 검증 후 조건부 삭제·소유 prefix2개 비움·소유 임시 경로 정리**를 확인했습니다. 로컬 worker·controller·실행 소스 archive, checkpoint8 진단 자료와 durable HOME 증거는 보존했습니다. 새 controller 계약29개 테스트와 독립 종료 예약 확인 후 사용한 **GPU1대·1회·최대7200초·추가$4·누적$22·재시도0** 승인은 소비됐고 다음 실행은 아래 별도 승인으로 구분합니다. 당시 누적 보수적 상한 **$21.772065718278096**은 실제 청구액이 아닙니다. [worker 회수 검증](runs/qwen-specialization-013/qwen013-cuda-20261111002536/worker-recovery-verification.json) · [controller 회수 검증](runs/qwen-specialization-013/qwen013-cuda-20261111002536/controller-recovery-verification.json) · [종료 자원 상태](runs/qwen-specialization-013/qwen013-cuda-20261111002536/final-resource-state.json) · [저장소 정리](runs/qwen-specialization-013/qwen013-cuda-20261111002536/storage-cleanup-receipt.json) · [실행 허용·비용](runs/qwen-specialization-013/qwen013-cuda-20261111002536/release-observation-002.json).
- **중단 후 진단 보강 — 원인 수정 아님:** Linux guard와 테스트에만 제한된 압력 진단을 추가했고 학습 엔진·PSI 증가 금지·메모리 한도·재시도 금지 조건은 유지했습니다. 진단은 호출당 최대47개 파일×16,385bytes, 진입 시점과 최신2개 표본만 보존하며 비원자적 관측 구간·수집 불가 상태를 명시합니다. **guard79개 테스트 통과**, 실제 CloudShell Linux 수집기 exit0·host counter18개 관측과 독립 검토를 확인했습니다. 해당 환경의 root `memory.current` 부재는 수집 불가로 기록했으며, **종료된 GPU host의 원인 귀속·guard admission·모델 실행 검증은 아닙니다.** +663µs도 계속 중단하며 이 진단 검증 자체는 유료 실행·후보 게시가 아닙니다. 소스·검증 archive를 보존한 뒤 인증한 임시 스크립트4개만 제거했습니다. [진단 보강 검증·범위](runs/qwen-specialization-013/qwen013-cuda-20261111002536/pressure-diagnostics-verification.json).
- **최신 실행 — 계산 완료, 감독 수락 실패:** `qwen013-cuda-20261111030853`의 trainer는 **23/23update·선호91/91쌍·replay288/288행**을 소비하고 checkpoint8/16/23·최종 adapter·완료 보고서를 저장했습니다(trainer214.741727405초). 그러나 host memory PSI full48,200→49,218µs(**+1,018µs=1.018ms**)로 guard가 규칙대로 중단해 **`FAILED_NO_RETRY`**입니다. 감독218.697377887초·child exit−15·guard exit1이며 자식 회수·정리는 완료됐습니다. 완료 보고서 구성은 SIGTERM보다 약135.5ms 앞서지만 압력 증가 구간이 최종 저장·보고서 구성과 겹쳐 **학습 종료 뒤에만 발생했다고 단정할 수 없습니다.** 모든 adapter는 진단용이며 **수락된 본학습·native DEV·후보 선정·서빙·게시 결과는 없고 final24는 봉인**했습니다. [계산·중단 분석](runs/qwen-specialization-013/qwen013-cuda-20261111030853/pressure-stop-analysis.json) · [trainer 보고서](runs/qwen-specialization-013/qwen013-cuda-20261111030853/worker/output/full-training/training.json).
- **최신 압력 관측의 한계:** peak RSS1,851,273,216bytes·GPU peak4,326,424,576bytes·최소 관측 MemAvailable13,344,227,328bytes였고, 자기 cgroup high/max/OOM·swap 이벤트 증가는 관측되지 않았습니다. 같은 구간에 host kswapd scan27,940page·steal27,936page 증가가 있었지만 direct reclaim·allocstall·동기 compaction·swap·OOM 증가는 기록되지 않았습니다. 순차·비원자적 관측으로는 발생 주체를 귀속할 수 없고, 검출 후 자기 cgroup/system.slice PSI나 원인 추적 자료도 없어 **RAM 고갈·lifecycle 결함·근본 원인 수정으로 해석하지 않습니다.** 안전 기준을 완화하거나 재시도하지 않았습니다. [감독 원본](runs/qwen-specialization-013/qwen013-cuda-20261111030853/worker/output/full-training-supervisor.json).
- **최신 회수·정리·공개 상태:** worker117항목·controller32항목의 SHA를 검증했고 `export_complete=true`는 **worker exit1의 실패 실행에서 증거를 회수했다는 뜻**입니다. 2026-11-11 04:01:19 UTC에 instance `i-0c8ef90d36e0dc05b` 종료·전용 EBS0·종료 예약 삭제를 확인했습니다. 04:06:56 UTC에는 소유 S3객체5개 전체 SHA 검증 후 조건부 삭제·prefix2개 비움·소유 임시 파일18개 정리를 확인했고 HOME 증거는 보존했습니다. 공개 브라우저03:44:33 UTC 관측은 기존2개 선택지·기본 Qwen3 원본 그대로이며 추론 요청·서비스 변경은 없었습니다. **GPU1대·1회·최대7200초·추가$4·누적$26·자동 재시도0** 승인은 소비됐고 새 유료 실행에는 새 승인이 필요합니다. 누적 보수적 상한 **$25.000109007666286**은 실제 청구액이 아닙니다. [worker 회수](runs/qwen-specialization-013/qwen013-cuda-20261111030853/worker-recovery-verification.json) · [controller 회수](runs/qwen-specialization-013/qwen013-cuda-20261111030853/controller-recovery-verification.json) · [종료 자원](runs/qwen-specialization-013/qwen013-cuda-20261111030853/final-resource-state.json) · [저장소 정리](runs/qwen-specialization-013/qwen013-cuda-20261111030853/storage-cleanup-receipt.json) · [공개 선택기](runs/qwen-specialization-013/qwen013-cuda-20261111030853/public-model-status.json).
- **안전 기준 유지·회수 후 진단 보강:** 공식 PSI 의미와 독립 소비 경로를 검토하고 최신 실행의723개 표본을 실제 guard 판단 함수로 재생해 마지막 +1,018µs에서만 거절됨을 확인했습니다. 증가 금지 규칙은 유지합니다. 이제 PSI 중단 시 **종료 신호·자식 회수·정리 완료와 기존 영수증 저장 이후에만** 별도의 `post_cleanup_pressure_diagnostics`를1회 남깁니다. 회수·그룹 종료가 입증되지 않으면 파일을 읽지 않으며, 진단 오류나 누락이 원래 실패를 성공으로 바꾸지 않습니다.
- **추가 진단의 검증·한계:** guard·pipeline·native 회귀 **167개 통과**와 독립 검토를 마쳤습니다. 실제 CloudShell Linux에서 무해한 자식에 시험용 PSI 중단 사유를 주입해 SIGTERM→SIGKILL→회수·정리 후 추가 수집을 실행했습니다. child exit−15·guard 실패 판정1은 유지됐고 수집은 약0.497ms였습니다. **실제 압력을 유발한 실험·전용 cgroup admission·모델·GPU·품질 검증은 아닙니다.** 추가 수집은 최대48파일×16,385bytes이며 원래 deadline 안에서0.25초의 협조적 예산을 공유합니다. 진행 중 syscall을 강제로 끊는 시간 보장은 아니며 외부 종료 한도는 유지합니다. 관측은 비원자적이고 workload lock 해제 뒤이므로 원인 귀속 증거로 단정하지 않습니다. 원본 엔진3개·과거 실행 pin·불합격 결과를 보존했고 유료 재실행은 하지 않았습니다. 다음 실행에는 새 소스 결박과 승인이 필요합니다. [검증·범위](runs/qwen-specialization-013/qwen013-cuda-20261111030853/post-cleanup-diagnostics/verification.json) · [실제 Linux 종료·수집 기록](runs/qwen-specialization-013/qwen013-cuda-20261111030853/post-cleanup-diagnostics/linux-real-child-smoke.json).

### 최신 체크포인트의 진단 답변 비교

사용자 승인으로 **실행 수락에 실패한 step23 체크포인트의 독립 진단 평가만** 수행했습니다. 원래 실패 기록·배포 기준은 유지하며, 추가 GPU 학습이나 공개 모델 교체는 하지 않았습니다.

- **조건:** Mac mini CPU에서 동일한 연구용 Q8 기반 모델에 최신 F32 LoRA를 끈 경우(scale0)와 켠 경우(scale1)를 비교했습니다. 질문8개·모델별9턴, 총18답변이며 질문과 평가 기준은 출력 전에 고정했습니다. system prompt 없음·thinking 끔·greedy(seed42)·최대128토큰·검색/사이트 필터 없음입니다. 고정 DEV/final24는 사용하지 않았습니다. **현재 사이트의 원본 GGUF·검색·필터 경로와 동일한 비교는 아닙니다.**

| 진단 대상 | 8개 사례의 독립 평가 기준 충족 | 생성 답변9개 중 길이 제한으로 잘림 |
|---|---:|---:|
| 연구용 기반 모델, 어댑터 끔 | 3/8 | 4/9 |
| 최신 step23 학습 어댑터 켬 | 3/8 | 0/9 |

- **실제 오답:** “계약직은 제외”라고 제시했는데 둘 다 신청 가능하다고 답했습니다. 학습본은 실제 마감인11월20일을 **“11월12일18시까지”**로 잘못 답했고, 지원 금액이 없는 문서에는 **“1인당10만 원이 지원됩니다”**라고 지어냈습니다. 두 모델 모두 대화 속 프로젝트명·요일은 기억했지만, 세 사실을 두 항목으로 요약하는 질문에서는 각각 한 사실을 누락했습니다.
- **해석 제한:** 모델 이름을 가린 독립 평가이며, 작은 신규 사례 묶음의 일반 정확도·통계적 우열을 뜻하지 않습니다. 원본의 지원금 보류는 잘리기 전에 요건을 충족해 통과했으나 답변 전체는 미완결입니다. 학습본의 출장 규정 확인 항목도 선언된 기준은 충족했지만 근거 없는10만 원 언급이 있어 안전한 답변으로 보증하지 않습니다. **현재 결과로 품질 개선을 확인하지 못했으며, 개선된 기본 모델로 교체할 근거가 부족합니다.**
- **실행 근거:** 변환된392개 F32/rank16 LoRA tensor의 GGUF 구조와 입력 SHA를 확인했습니다. 실제 추론5청크 모두 압력1·정상 종료·회수 조건을 충족했고 최대 관측 그룹 RSS는2,487,582,720bytes(약2.32GiB)였습니다. 전체18요청의 프롬프트·토큰·대화 이력을 공식 tokenizer와 대조했습니다. 이는 단독 진단 실행이지 정식 서빙 수락은 아닙니다. 첫 청크 후 포트의TIME_WAIT 때문에 다음 청크가 추론 전에 차단된 기록도 보존했고, 나머지는 서로 다른 격리 포트로 실행했습니다. 평가 종료 후 소유 모델 프로세스가 없고 기존 native 서비스의 health 응답이200임을 확인했습니다.
- **별도 보존 자료:** [결과·한계](runs/qwen-specialization-013/diagnostic-answers-20261111/result.json) · [실제 답변과 실행 영수증](runs/qwen-specialization-013/diagnostic-answers-20261111/paired-execution-evidence.json) · [독립 평가](runs/qwen-specialization-013/diagnostic-answers-20261111/independent-judgment.json). 실행 자료와 가중치는 Git에 포함하지 않습니다.

### 서빙 수락과 배포 경계

`scripts/qwen_serving_qualification.py`는 과거 임시 디렉터리 대신 명시적 소스·계획·인터프리터·native 실행 파일·라이브러리 해시를 받습니다. `prepare`→`admit`→`run`→`qualify` 순서이며 준비 완료만으로 모델을 게시하지 않습니다.

- **선택 무결성:** 고정 DEV 시나리오 ID·그룹·rubric family를 원본 SHA와 대조한 뒤 기존 선택기를 재실행합니다. answerable control을 삭제하거나 분류를 바꿔 보존 기준을 우회할 수 없습니다. baseline 재생성·재채점과 final24 사용은 하지 않습니다.
- **실행 검증:** 관련 테스트 **28개 통과**. 실제 commit `8ca4060` 실행 파일의 `--version`과 native 라이브러리10개의 로더·정적 의존성을 확인하고, 보존된 원본 계획으로 실제 `prepare` CLI를 실행해 `PREPARED_NO_MODEL_EXECUTION`을 얻었습니다. **모델 가중치를 로드하거나 응답을 검증한 결과는 아닙니다.** [실행 증거](runs/qwen-specialization-013/qwen013-cuda-20261110035134/serving-verification.json).
- **이전 PSI 중단 실행의 서빙 준비:** `qwen013-cuda-20261111002536`용 격리된 Mac mini 경로에 소스·메타데이터13개만 준비했습니다. 계획한 CPU 전용 환경의 실제 `--version`은 exit0이며, DYLD trace의 비시스템 라이브러리11개 경로·SHA가 runtime 결박과 일치함을 확인했습니다. 실제 `prepare` CLI는 **`PREPARED_NO_MODEL_EXECUTION`**입니다. **모델 로드·서빙 적격성·통합 RSS·배포 검증은 아니며 운영 서비스는 변경하지 않았습니다.** 해당 GPU 본학습 중단으로 native DEV·후보 선정에는 이르지 못했습니다. [로더 환경 증거](runs/qwen-specialization-013/qwen013-cuda-20261111002536/site-loader-environment-proof.json) · [격리 준비 영수증](runs/qwen-specialization-013/qwen013-cuda-20261111002536/serving-remote-preparation.json).
- **남은 게시 조건:** 독립 DEV 선정 후 CPU-only180초×3청크에서 실제 native10요청·OpenAI 호환1요청, 대화 이력·새 세션·초기화 격리, 토큰 일치, 메모리·종료 회수를 검증하고 응답 내용을 독립 검토해야 합니다. 단독 서빙 합격도 공개 서비스의 통합 자원 수락이나 배포 완료를 뜻하지 않습니다. 이전 Mac mini 압력2 관측 때 추가 모델을 시작하지 않았으며, **2026-11-11 00:29:27 UTC의 읽기 전용 관측은 압력1**입니다. 모델 로드·서비스 변경은 없고 서빙 적격성이나 통합 메모리 수락 결과도 아니며, 모델 로드 직전에 다시 확인해야 합니다. [현재 압력 관측](runs/qwen-specialization-013/qwen013-cuda-20261111002536/macmini-pressure-readonly.json).

### 학습 단계별 중단 진단

`scripts/trace_qwen_preferences.py`는 기존 trainer를 한 번 호출하면서 단계별 시작·정상 반환·예외를 별도 JSONL에 즉시 flush합니다. 캐시 검증, 초기 점수 비교, 선호/replay forward·backward, optimizer, validation, adapter 저장을 구분합니다. 캐시를 결박하는 기존 엔진 소스3개와 회수한 Linux 참조 캐시6개·91쌍은 SHA가 그대로임을 확인했습니다.

- **호출:** `python -B scripts/trace_qwen_preferences.py --trace <새 JSONL 경로> -- <기존 train 인자 그대로>`. 기존 학습 의존성이 필요하며 `--mode train`만 허용합니다. trace의 부모 디렉터리는 이미 있어야 하고, trace는 trainer의 `--out` 밖에 있어야 합니다. 기존 trace는 덮어쓰지 않습니다. 인자·CWD를 보존하고 실제 런타임 버전·소스 해시를 기록하며, prompt·정답·토큰·텐서 값과 예외 메시지는 기록하지 않습니다.
- **해석:** `optimizer_update_returned`는 optimizer 구현이 정상 반환했다는 뜻일 뿐 scheduler·validation·checkpoint 완료가 아닙니다. `adapter_save_returned`도 직렬화 반환이며 checkpoint 장부 기록 완료가 아닙니다. SIGTERM 뒤에는 완료 이벤트를 만들어 붙이지 않습니다. 종료 이벤트가 없다고 실제 계산이 전혀 수행되지 않았다고 단정해서도 안 됩니다. flush는 fsync가 아니며, 계측·파일 I/O 비용이 있으므로 무계측 처리속도와 구분합니다.
- **실행 검증:** Darwin CPU의 Torch2.9.1·NumPy2.5.4·Transformers4.57.6·PEFT0.17.1 환경에서 기존 trainer 회귀24개와 최종 tracer 회귀13개를 통과했습니다. 실제 자식 프로세스를 validation 중 SIGTERM으로 종료해 앞선 optimizer 반환 기록이 남고 validation 성공·run 성공 기록은 생기지 않음을 확인했습니다. 파일 flush/close 오류가 원래 학습 예외를 가리지 않는 경로도 검증했습니다.
- **독립 합성 스모크:** 임의 초기화한 hidden32·2층·rank2 LoRA와 선호2쌍·replay3행으로 원래 학습 함수의 optimizer2회·validation2회를 실행했습니다. 계측 전후 손실·기울기·갱신 가중치·optimizer 상태가 정확히 일치했고 실제 adapter load/save와 패치 복구도 확인했습니다. 사전학습1.7B 가중치·원 TRAIN/DEV/final 데이터는 사용하지 않았습니다. 이는 실제 모델의 속도·품질 개선이나 Linux 캐시 재사용 적격 판정이 아닙니다.
- **원격 런처 코드 통합:** `qwen_cpu_smoke.py`와 `qwen_cpu_pipeline.py`가 guarded child를 tracer로 실행하고 정확한 argv를 검사하도록 연결했습니다. smoke와 본학습의 기록은 각각 `model-smoke-phases.jsonl`, `full-training-phases.jsonl`에 남기도록 구성했습니다. 기존 guard180초/systemd175+5초·RSS6GiB·본학습900초 적합 판정과 품질 기준은 유지합니다. trace 이벤트는 기존 성공 판정을 대체하지 않습니다. 아래 실제 원격 실행은 smoke에서 중단됐으며 본학습 trace나 성공을 뜻하지 않습니다.
- **로컬 증거:** `runs/qwen-specialization-013/phase-trace-001/{verification.json,test-results.json,synthetic-phases.jsonl,cli-invalid-input.jsonl}`. 실행 자료는 별도 보관 대상입니다.
- **캐시 보존 실행:** `qwen013-traced-20261109203337`은 이전 packed 실행에서 수락한 Linux 캐시6개·91쌍을 **재생성 없이 재사용**했습니다. 이전 수락·회수 영수증, 정확한 파일 집합·SHA·분할·실제 Linux 런타임·엔진·현재 입력 결박을 검사하는 계약과 불일치 시 중단 규칙을 유지했습니다. 최초4쌍 정책/참조 최대 오차는0(허용0.001)이었습니다. 원래 엔진·가중치·데이터·캐시와 과거 `export_complete=false` 기록은 바꾸지 않았고 final24도 열지 않았습니다.
- **준비 단계의 통합 검증:** 관련 회귀146개, bootstrap shell 문법과 Python heredoc4개 구문 검사 통과는 **준비 당시의 기록이며 이번 결과 정리에서 재실행하지 않았습니다.** 당시 실제 준비 파일에 production validator를 실행하고, 새 helper가 만든 CLI에서 원래 trainer의 입력 거부와 오류 trace를 관측했습니다. Darwin을 Linux 캐시 런타임으로 받아들이지 않는 것도 확인했습니다. 전송 묶음107항목(입력106개+manifest)의 개별 SHA를 검증했고, raw EC2 user-data 형태는15,976bytes로16KiB 이내였습니다. 이는 학습 성능·모델 품질 검증이 아닙니다.
- **실제 중단 결과:** 기존 Chrome의 인증된 세션으로 승인된1회 실행을 완료했으며 최종 결과는 **SMOKE_FAILED_NO_CANDIDATE**입니다. 로그에 optimizer1회(선호4쌍·replay12행) 완료가 남았습니다. update2는 두 번째 쌍 backward가 마지막 관측입니다. `owner_signal_SIGTERM`으로 중단됐으며 감독기 총 경과는175.368533초입니다. **2-update 스모크는 수락되지 않았고 완료된 checkpoint·유효한900초 본학습 예측·본학습·native 생성·DEV 평가·새 후보는 없습니다.**
- **계산 시간의 해석:** 첫 단계 gradient accumulation은121.732842초이며, 완료된 선호 forward/backward71.540398초·replay50.188096초로 관측됐습니다. optimizer 자체는0.061693초입니다. trainer의157.1초는 **누적 경과시간이지 update 한 번의 소요시간이 아닙니다.** 이 trace만으로 정확한 kernel 병목이나 정상 반복 구간의 처리속도를 확정하지 않습니다.
- **메모리 관측:** 678개 표본에서 peak RSS는4.194897GiB로6GiB 한도 미만, 최소 MemAvailable은25.8485GiB였습니다. OOM·swap·PSI 압박은 관측되지 않았습니다. 다음 작업은 단순 RAM 증설이나 유료 실행 반복이 아니라 **목적함수·안전 및 품질 기준을 보존하는 계산 프로파일링·최적화**입니다.
- **회수·정리와 승인 경계:** 보존35항목의 해시는 모두 검증했지만 export manifest가 없어 **`export_complete=false`는 여전히 false**입니다. instance `i-0215cbe013f8b3eae` 종료·EBS0·종료 예약 삭제·정확한 S3객체5개와 소유 CloudShell 임시 경로 삭제를 확인했고 durable HOME intent는 보존했습니다. 정확한 실행 소스18개를 `executed-sources.tar`에 보존·검증했고 현재 소스 pin18개와 원래 캐시6개의 SHA도 불변입니다([실행 소스 검증](runs/qwen-specialization-013/qwen013-traced-20261109203337/executed-source-verification.json)). 로컬에서는 중복 `inputs.tar`(1,961,267,200bytes)와 upload-response 임시 파일만 제거했습니다([로컬 전송 정리](runs/qwen-specialization-013/qwen013-traced-20261109203337/local-transfer-cleanup.json)). 현재 보수적 상한 **추가$1.999099837·누적$8.859892561**은 실제 청구액이 아닙니다. 추가$3·최대2시간·누적$10의 **승인1회는 소비됐고 자동 재시도나 배포를 허가하지 않습니다. 또 다른 유료 실행에는 새 승인이 필요합니다.**
- **실행·회수 근거:** [실행 결과](runs/qwen-specialization-013/qwen013-traced-20261109203337/run-outcome.json) · [trace 진단](runs/qwen-specialization-013/qwen013-traced-20261109203337/trace-diagnosis.json) · [회수 검증](runs/qwen-specialization-013/qwen013-traced-20261109203337/output-recovery-verification.json) · [controller 감사](runs/qwen-specialization-013/qwen013-traced-20261109203337/controller-audit-verification.json) · [저장소 정리](runs/qwen-specialization-013/qwen013-traced-20261109203337/storage-cleanup-receipt.json) · [공개 모델 상태](runs/qwen-specialization-013/qwen013-traced-20261109203337/public-model-status.json). 같은 run의 `authorization-intent.json`, `approval.json`, `execution-plan.json`, `local-test-results.json`, `local-smoke-verification.json`, `input-bundle-verification.json`은 승인·준비 이력이며 실행 자료는 별도 보관 대상입니다.

## 기존 자체 모델 실험

아래 구조·학습 명령과 그림은 **이전 자체 모델 연구 기록**입니다. 새 제품 기반 모델의 구조나 성능으로 읽으면 안 됩니다.

### 모델 구조

<p align="center"><img src="docs/figures/architecture.png" width="820" alt="trid-91M Transformer 구조: RoPE, RMSNorm, SwiGLU, 공유 임베딩"></p>

두 구성 모두 decoder-only Transformer에 RoPE, pre-norm RMSNorm, SwiGLU, 공유 입·출력 임베딩을 사용합니다.

| 항목 | trid-91M — 기존 실험 모델 | 350M급 — 이전 구조 후보 |
|---|---|---|
| 설정 | [korean-medium-91m.json](configs/korean-medium-91m.json) | [korean-350m.json](configs/korean-350m.json) |
| hidden / 층 수 | 768 / 12 | 1,024 / 28 |
| query head / KV head | 12 / 12 | 16 / 4 (GQA) |
| FFN hidden | 2,048 | 3,072 |
| QK-norm | 끔 | 켬 |
| 어휘 / 문맥 | 8,192 / 1,024 | 8,192 / 1,024 |
| 상태 | 사전학습·SFT·추출 독해 평가 기록 있음 | 설정 추가 및 작은 학습 실행 확인 기록; 독해 성능 미확인 |

위 구조 그림은 **91M** 기준입니다. 350M급 후보는 크기뿐 아니라 GQA와 QK-norm도 바뀌므로 파라미터 수만의 효과로 해석할 수 없습니다. 이 설정 표는 별도로 실행한 390M MHA 비교 실험과도 구분합니다. 새 제품 개발을 위한 추가 자체 사전학습은 보류 상태입니다.

### 질문에 답하는 방법

근거 문단을 `[D1]`, `[D2]`처럼 표시해 질문과 함께 입력합니다. 문서 QA의 제한 출력은 다음과 같습니다. 아래는 출력 형식 예시이지 실제 모델의 시연 결과는 아닙니다.

```text
8만 원 [D3]      근거에서 고른 값과 출처
확인 불가         근거 부족
근거 충돌         근거 사이의 모순
```

- **생성형 독해**: 다음 토큰을 예측하되, 근거 제한 디코딩에서는 인용 문단의 부분 문자열과 유효한 출처 번호만 허용합니다.
- **구간 점수화**: 근거 안의 후보 구간을 `값 [Dn]`으로 만들고 전체 로그우도로 비교합니다.
- **추출 독해기**: 몸통에 답 있음·없음·충돌 분류와 시작·끝 위치 헤드를 붙여, 문서 구간을 직접 고릅니다. 출처는 해당 구간의 문단에서 정합니다.

**복사 제한은 정확성 보장이 아닙니다.** 근거 안의 잘못된 값이나 악성 지시문도 선택할 수 있습니다. 이 제약은 일반 자유 생성·설명 답변에 자동 적용되지 않으며, 개인정보 마스킹·권한 필터·보안 평가가 별도로 필요합니다.

### 기존 학습 방식

<p align="center"><img src="docs/figures/pipeline.png" width="960" alt="공개 자료 정제, 토크나이저 학습, 사전학습, SFT와 근거 기반 추론 흐름"></p>

그림은 자체 학습과 문서 QA의 전체 흐름입니다. 초기 Mac 실험에서 시작했으며, 이후 이어 사전학습과 SFT에는 CUDA GPU도 사용했습니다.

<p align="center"><img src="docs/figures/training-data.png" width="960" alt="91M 사전학습 단계와 준비된 혼합 말뭉치 구성"></p>

- **사전학습**: 한국어 위키백과, 국가법령정보센터 자료, 정책브리핑 보도자료를 사용합니다. 정제·개인정보 치환·중복 제거를 거치고 출처와 라이선스를 기록합니다.
- **지도학습**: 합성 기관 문서 QA, KLUE MRC, TyDi QA 한국어와 실제 보도자료 기반 QA를 사용합니다. 같은 문서에서 질문을 바꿨을 때 답도 올바르게 바뀌는지 확인하는 실험을 진행했습니다.
- **질문 작성 보조**: 일부 실제 문서 QA는 공개 모델이 질문 초안을 작성하고 다른 모델이 검증합니다. 해당 QA의 답은 원문에서 복사하고 작성·검증 모델을 기록합니다. 이는 외부 가중치를 이어받는 학습과 다릅니다.

혼합 자료 막대는 **준비된 말뭉치의 측정값**입니다. 측정용 토크나이저가 실제 CPT 토크나이저와 달라 학습 중 토큰 노출 비율로 읽으면 안 됩니다. 위키 검증 손실과 혼합 검증 손실도 데이터가 달라 직접 비교할 수 없습니다.

### 현재 우선순위

1. **공개 가중치 기준선과 특화 효과** — 원본 모델과 추가 학습 모델을 같은 조건에서 평가합니다. 기존 자체 모델은 비교 자료로 보존합니다.
2. **정확한 근거 활용** — 조건·예외·적용 대상·시점·표·단위를 검증하고, 근거 부족·충돌·조회 필요를 구분합니다. 답변을 많이 하는 것과 정확하게 답하는 것을 구분합니다.
3. **자연스러운 설명과 실험 배포** — 짧은 값 추출 성능으로 설명 품질을 대신하지 않습니다. 검증 결과와 제한사항을 표시한 버전을 테스트 사이트에서 확인합니다.

학습 계보와 실패 기록은 [사전학습 기록](docs/experiments/PRETRAINING.md), [RAG 실험 기록](docs/experiments/KOREAN_RAG.md), [연구 기록](docs/RESEARCH_LOG.md)에 보존합니다.

## 평가

### 평가 범위와 지표

| 평가 | 무엇을 측정하나 | 판정 |
|---|---|---|
| RAG310 전체·답 있음·답 없음 | 근거를 제공했을 때 값과 출처를 맞히는가 | 엄격 정답: 값·출처·출력 형식 모두 일치 |
| RAG310 과제별 분석 | 표, 개정, 방해 문서, 다중 근거에 대응하는가 | 동일 체크포인트·동일 디코딩의 과제별 엄격 정답률 |
| 잘못된 보류 | 답이 있는데도 회피하는가 | 답 있음 문항에서 보류한 비율; 낮을수록 좋음 |
| 기관 문서 probe | 실제 보도자료의 값을 추출하는가 | 추출 독해기는 허용 답 문자열과 exact 일치 |
| 구조 그래프 검색 | 답에 필요한 여러 문단을 모두 찾는가 | 필요한 근거 전체 회수율; 답변 정답률과 별개 |

아래 그림은 새로운 모델 추론이 아니라 **기존 자체 모델 연구에서 저장한 원본 응답·집계의 시각화**입니다. 새 제품 기반 모델의 선정 결과나 특화 성능이 아닙니다. MMLU, 수학, 코드 생성 같은 범용 벤치마크 점수로 대체해서 읽으면 안 됩니다.

### 공개 모델과의 기준선 비교

<p align="center"><img src="docs/figures/benchmark.png" width="960" alt="기존 91M 생성 모델과 별도 추출 독해기의 기록된 기준선 비교"></p>

이 그림은 **기존 체크포인트의 기록된 기준선**입니다. 350M 후보의 결과나 현재 배포 모델의 통합 성능이 아닙니다. 새 실험을 가장 좋은 결과로 자동 교체하지 않습니다.

- **RAG310**: 자체 작성 160문항 + KLUE dev 150문항. 질문과 근거를 이미 제공한 독해 평가로, 검색 포함 서비스 성능은 아닙니다. trid는 `public-dialogue-gpu-05-mix04-s3000` 생성 모델입니다.
- **기관 probe**: 실제 보도자료 기반 40문항. trid는 별도 `cpt-v1/head-v2/data-v4` 추출 독해기의 `exact`입니다. 생성 모델의 값·출처·형식 채점과 동일한 지표가 아닙니다.
- **외부 기준선**: Qwen3·Kanana와 그 LoRA 결과입니다. 공개 모델에도 근거 제한 문법을 적용한 조건이며, 각 모델의 범용 능력 순위를 뜻하지 않습니다. 외부 모델의 사전학습 데이터 노출 여부는 확인할 수 없습니다.
- **평가 한계**: 기관 probe와 개발 중 반복 확인한 평가는 새 독립 최종 평가가 아닙니다. 합성 문항과 실제 기관 문서 성능도 구분합니다.
- **새 최종 평가:** 공공기관 업무 독립 평가 2,000문항을 목표로 준비 중이며 완성되지 않았습니다. 예약 문서 1,000계열 중 서명을 확보한 148계열만 중복 검사가 가능하고 852계열은 미확인입니다. 상위 법령의 공유 문구와 기관 고유 문서의 노출도 구분해야 합니다. 이 상태에서 학습·최종 평가가 완전히 독립이라고 주장하지 않습니다.

[생성 모델 원본 보고서](reports/open-weight-baseline/) · [공개 모델 비교](reports/open-weight-lora/) · [추출 독해 결과](reports/reader-factorial-01-two-complete.json)

### 과제별 능력: 레이더

<p align="center"><img src="docs/figures/task-radar.png" width="960" alt="RAG310의 8개 과제별 엄격 정답률 레이더: trid-91M과 공개 LoRA 기준선"></p>

모든 축은 **0–100% 엄격 정답률**로 방향과 척도가 같습니다. 각 축에 문항 수를 표시하며, 축별 표본 수가 다르므로 다각형의 면적을 종합 점수로 사용하지 않습니다. 이 레이더는 RAG310 중 답 없음 과제를 제외한 8개 유형을 보여주며, 전체 평가를 대신하지 않습니다.

trid의 표 읽기·두 근거 결합·충돌 판정은 이 체크포인트에서 정답을 내지 못했습니다. 원문 문자열만 출력하도록 제한하는 것과, 질문에 맞는 문자열을 선택하는 것은 다른 능력임을 보여줍니다. 공개 모델과의 차이는 파라미터 수만의 효과가 아니라 사전학습량·자료·후학습의 차이도 포함합니다.

### 답변 품질: 보류와 오답 분리

<p align="center"><img src="docs/figures/answer-errors.png" width="960" alt="답이 있는 228문항을 엄격 정답, 잘못된 보류, 나머지 비정답으로 나눈 누적 막대"></p>

전체 정답률만 보면 답이 없는 문항을 자주 보류하는 모델이 유리해질 수 있습니다. 따라서 **답이 있는 문항만** 떼어 정답·보류·나머지 비정답으로 분해합니다. 각 막대는 같은 문항 수를 분모로 하며 세 범주의 합은 100%입니다.

여기서 나머지 비정답은 모두 환각이라고 단정할 수 없습니다. 값 선택, 출처, 형식 등 여러 실패가 섞여 있습니다. 반대로 근거가 없는 문항에서의 올바른 보류는 앞의 전체 비교에 별도로 포함합니다.

### 학습 데이터 실험 비교

<p align="center"><img src="docs/figures/experiment-comparison.png" width="960" alt="자체 91M의 네 가지 3000단계 데이터 실험 비교"></p>

네 실험 모두 자체 91M의 생성 모델이며 RAG310 답 있음 문항을 **같은 엄격 지표**로 비교합니다. 질문을 같은 문맥에 여러 개 묶은 구성이 자동으로 독해력 개선을 만들지는 않았습니다. 문서별 teacher QA도 이 엄격 지표에서는 기준선과 같았습니다.

이는 데이터만의 깨끗한 인과 비교를 의미하지 않습니다. 공유 문맥 구성은 일부 지문을 제외하고 입력 길이·방해 근거 수도 바꿉니다. 또한 이 그림의 엄격 정답과 실험 판정 보고서의 느슨한 `출처 일치 + 정답 언급` 지표를 혼용하지 않습니다.

[teacher QA 실험 판정](reports/public-dialogue-gpu-06-assessment.json) · [공유 문맥 실험 판정](reports/public-dialogue-gpu-07-assessment.json) · [모든 원본 점수](reports/open-weight-baseline/summary.md)

### 검색 품질과 지연

실제 법령·행정규칙·자치법규 **3,000문서 / 109,380문단**의 검색 결과입니다. 다중 근거 개발 문항 80개에서 필요한 근거를 **모두** 찾았는지 측정했습니다.

| 검색 방식 | 반환 상한 | 전체 근거 회수율 | 실제 평균 반환 문단 | 검색 p50 | 검색 p95 |
|---|---:|---:|---:|---:|---:|
| BM25 | 12 | 6.25% | 12.00 | 41.08 ms | 61.98 ms |
| BM25 상위 3개 + 2홉 그래프 | 12 | 16.25% | 4.70 | 41.94 ms | 62.24 ms |
| BM25 | 20 | 17.50% | 20.00 | 41.17 ms | 62.46 ms |
| BM25 상위 5개 + 2홉 그래프 | 20 | 22.50% | 7.92 | 41.32 ms | 62.80 ms |

반환 **상한**은 같아도 실제 반환 개수는 다릅니다. 파서의 참조 경로에서 자동 구성한 문항이라 그래프에 유리할 수 있고, 사람이 만든 독립 정답셋이 아닙니다. 시간은 단일 프로세스의 검색 호출 기준으로, 모델 추론·HTTP·동시 요청을 포함하지 않습니다. [원본 측정과 재현 조건](reports/graph-retrieval-v1.md)

<details>
<summary><b>차트 데이터와 재현</b></summary>

`scripts/plot_readme_figures.py`는 모델을 실행하지 않고 저장된 보고서를 읽습니다.

- 비교·레이더·보류 분석의 자체 모델: `reports/open-weight-baseline/rag310-trid-public-dialogue-gpu-05-mix04-s3000-evidence.json`
- 공개 비교 모델: `reports/open-weight-lora/rag310-{kanana,qwen3}-lora.json`의 `evidence_constrained`
- 데이터 실험: `reports/open-weight-baseline/rag310-trid-public-dialogue-{gpu-05-mix04,gpu-06-mix05,gpu-07-shared-ctx-01,gpu-07-mix06}-s3000-evidence.json`
- 기관 추출 결과: `reports/reader-factorial-01-two-complete.json`

차트 생성은 Python 표준 라이브러리, PNG 변환은 `rsvg-convert`를 사용합니다. 아키텍처 그림과 동일한 `Nanum Pen Script` 폰트가 필요합니다. 전체 생성의 학습 데이터 그림은 별도 로컬 `runs/`·`data/` 자료를 요구하지만, 새 레이더·오류·실험 비교 그림은 커밋된 보고서만으로 생성됩니다.

</details>

## 폐쇄망 서비스

<p align="center"><img src="docs/figures/system.png" width="960" alt="문서 수집, 조문 구조 검색, 권한 필터와 추출 독해를 연결하는 목표 시스템"></p>

HWP·HWPX·PDF·텍스트에서 문단을 추출하고, Kiwi 형태소 분석과 BM25로 검색합니다. 법·장·조·항·호 및 명시 참조를 파싱한 구조 그래프로 관련 조문을 확장합니다. 권한 필터와 감사 로그는 모델 밖 서비스 계층에서 처리합니다.

그림은 **목표 구조**를 포함합니다. 특히 risk–coverage 기반 보류 확률 보정과 CPU 지연 목표를 이미 달성했다고 주장하지 않습니다. 모델 점수, 서비스 전체 지연, 보안성은 별도로 검증합니다.

## 실행

가중치는 아직 공개 배포하지 않습니다. 아래 명령은 의존성과 **호환되는 자체 모델의 로컬 run 폴더**를 준비한 개발 환경용입니다. 공개 모델용 HF 체크포인트를 그대로 넣는 명령이 아닙니다.

```sh
uv sync --frozen --python 3.12 --extra service

# <documents>와 <run>을 실제 경로로 바꿉니다.
.venv/bin/trid ingest <documents> --index index/
.venv/bin/trid ask --index index/ --run runs/<run> --user kim '숙박비 상한은 얼마인가?'
.venv/bin/trid serve --index index/ --run runs/<run>
```

HWP는 [rhwp](https://github.com/edwardkim/rhwp), PDF는 poppler의 `pdftotext`가 필요합니다. 오프라인 설치, 사용자 권한, API 인증과 실행 조건은 [서비스 안내](docs/SERVICE.md)를 참고하세요. GGUF 내보내기는 지원되는 생성 모델 경로이며 추출 헤드를 포함한 서비스 전체 변환을 뜻하지 않습니다.

## 개발과 Git 관리

- **A — 학습·배포:** 기반 모델 선정, 특화 학습, 모델 아티팩트, 서빙과 테스트 사이트 배포를 담당합니다.
- **B — 수집·학습 데이터:** 출처별 병렬 수집, 권리·개인정보·품질·시점·중복 검사, 학습 사례와 고정 데이터 manifest를 담당합니다. 같은 호스트의 요청 제한은 모든 작업자가 공유합니다.
- **C — 테스트:** 고정 평가 실행, 원본·특화 모델의 블라인드 답변 검토, 회귀 분석과 배포 사이트 검증을 담당합니다. 평가 정답을 학습 데이터에 넣지 않습니다.
- 진행 중인 C의 학습 사례 작업은 배치를 고정한 뒤 B에 인계합니다. 이 역할 분리 자체가 기존 평가의 독립성이나 사람 검증을 보증하지는 않습니다.
- 공유 작업 트리에서는 담당 파일만 명시적으로 스테이징합니다. `git add .` 또는 `git add -A`로 다른 세션의 작업을 함께 커밋하지 않습니다.
- 검증한 작업 단위로 커밋하고, 다른 세션의 변경을 임의로 초기화·되돌리거나 강제로 푸시하지 않습니다. 공유 파일 변경은 먼저 소유자를 조율합니다.
- 가중치·체크포인트·대량 원문·인증 정보는 커밋하지 않습니다. 코드와 재현 설정, 필요한 작은 평가 결과 및 데이터·모델 해시로 실행을 추적합니다.
- 배포 버전에는 기반 모델과 revision, 토크나이저·추론 설정, 학습 데이터 manifest, 코드 커밋, 평가 결과와 제한사항을 연결합니다. 실행 여부와 성능은 관측한 근거에 따라 기록하고 계획을 완료 상태로 표시하지 않습니다.

## 라이선스

코드는 [Apache License 2.0](LICENSE)입니다. 데이터에는 위키백과·KLUE의 CC BY-SA, 정책브리핑의 공공누리 제1유형, TyDi QA의 Apache 2.0 등 자료별 조건이 적용됩니다. 법령 등 비보호 저작물과 사이트의 다른 콘텐츠를 구분하며 각 데이터의 출처·이용 조건은 `provenance.json`과 원 출처를 확인합니다. 코드 라이선스가 모든 데이터나 향후 가중치의 이용 조건을 대신하지 않습니다.
