# CHAT-ROOT-A-001: 일반 대화 운영 경로 정적 원인 분리

작성: A, 2026-11-02. 운영 메타 확인 05:53–05:56 UTC.

## 0. 결론과 정정

**의미 보존 실패는 확인됐지만 단일 근본 원인은 아직 특정하지 못했다. 모델 크기·가중치 문제로 단정하지 않는다.**

- 부모 공개 UI 관측을 사실로 수용했다. 새 일반 대화의 메일 다듬기 요청에서 확인 요청을 참석 확정으로 바꾸고 새 질문을 덧붙였다. 동일 실패 재실행 0.
- 이번 소스 추적에서 UI나 후처리가 그 문장을 새로 써 넣는 경로는 발견되지 않았다. 이 오류는 **생성 결과의 의미 충실성 문제로 위치를 좁힐 수 있지만**, 생성에 영향을 준 요청·sampling·런타임·가중치 중 무엇이 원인인지는 별개다.
- 현재 GGUF 내장 template과 pinned HF template은 원문이 다르지만, **부모 공개 입력에 대한 로컬 Jinja 렌더 문자열은 일치**했다. 실제 llama renderer/tokenizer의 이번 입력 토큰 동등성을 새로 시험한 것은 아니다.
- 일반 대화 설정은 Qwen 권장 non-thinking의 temperature/top_p/top_k/min_p와 일치한다. 다만 presence penalty 1.5, 고유 system, 첫 턴 문구, 출력 384, history 절단, Q8_0/llama CPU 경로라는 별도 조건이 있다. 권장값 준수는 정답 보증이 아니다.
- 이전 A034 HANDOFF의 “모델 능력 문제”, “모델 선택으로만 개선 가능”을 철회했다. 프롬프트 3종 × decoding 3종 근거는 **GENERAL-CHAT-003**이며 `C-030.md`(검색 회귀) 인용도 정정했다.

**왜 배포·인프라가 진행됐는데 대화는 그대로인가:** GENERAL-CHAT-003에 이미 남아 있던 일반 대화 실패가 해결됐다는 근거 없이, 이후 A024–A032는 검색·기관 전환·표 조건 경로를 수정했다. A032는 일반 대화 생성 계약과 가중치를 바꾸지 않았다. AWS는 setup 실패로 GPU 추론 0이다. 따라서 이 작업들의 완료를 기본 대화 개선으로 연결할 근거가 없었다. 이 보고도 개선 완료 보고가 아니다.

## 1. 범위와 증거

- 코드 변경·새 학습·새 모델 생성·CPU/GPU 인스턴스 기동·운영 재시작 0. 기존 보고 및 README의 상태/단정만 정정했다.
- 비밀 파일·실제 사용자 대화 로그·C 비공개 문항/gold·P/Q 매핑 열람 없음. 부모가 공개한 문장과 기존 공개 진단만 사용했다.
- 메타 확인: `/props` GET, 프로세스 인자, LaunchAgent에서 CPU 관련 두 환경변수만 추출, 지정된 소스 파일 해시. `/completion`, `/api/ask`, 모델 생성 요청은 하지 않았다.
- 로컬 static check: `shared_demo.py` AST에서 system 문구를 추출해 현재 `/props.chat_template` 및 pinned HF template을 Jinja로 렌더. 서비스 모듈이나 모델을 import하지 않았다.
- 신규 원자료:
  - `CHAT-ROOT-A-001/static-contract.json`: allowlist `/props`, canonical 출처, 공개 messages와 렌더, 해시, 비교 범위.
  - `CHAT-ROOT-A-001/runtime-source.json`: 배포 식별자, 파일 해시 일치, 프로세스 인자, 당시 AWS 조회.
- 소스 참조는 `../trid-model-030`의 배포 코드 기준이다. shared_demo SHA `83ba188a…1b45`는 운영과 같다. helper `evaluate_external_chat.py`, UI `app.js`, versions 설정도 운영과 로컬 SHA 일치.

## 2. 운영 모델·실행 환경

| 항목 | 관측 / 근거 |
|---|---|
| 배포 | `DEPLOYED_FROM_COMMIT = 2b8ba92 (branch model-resume-030-followup)` |
| 모델 | `Qwen3-1.7B-Q8_0.gguf`; 기본 ID `trid-assistant-qwen3-1.7b`, reader와 2개만 제공 |
| 가중치 식별 | versions 기대 SHA `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`; 공식 GGUF revision `90862c4b9d2787eaed51d12237eafdfe7c5f6077` |
| 해시 검증 범위 | `State.start_llama`(2020–2048)는 기동 전 파일 digest를 대조하고 다른 모델/이미 열린 서버 채택을 거부함. 이번 진단에서 대용량 GGUF를 다시 해싱하지 않았으므로 현재 파일 재해시 실측이라고 주장하지 않음 |
| 서버 | `/props.build_info = b1-8ca4060`; PID 92545; model_path가 versions와 일치 |
| 실행 인자 | `-m …Qwen3-1.7B-Q8_0.gguf --host 127.0.0.1 --port 51912 -c 4096 -np 1 -t 4 -tb 4 --jinja` |
| CPU | LaunchAgent `LLAMA_ARG_DEVICE=none`, `LLAMA_ARG_N_GPU_LAYERS=0`; 새 offload/모델 교체 없음 |
| tokenizer | HF tokenizer를 매 요청 로드하는 것이 아님. GGUF/llama의 `/apply-template`와 `/tokenize` 사용. HF pinned config의 `tokenizer_class=Qwen2Tokenizer` 명칭은 Qwen3 배포 오류의 증거가 아님 |
| EOS/BOS | live EOS `<\|im_end\|>`; HF EOS 동일, `add_bos_token=false`, `bos_token=null`. live props의 bos 표시는 `<\|endoftext\|>`로 다름. 표시 차이와 실제 BOS 추가 여부는 구별해야 함 |
| adapter | live 기본 generation params `lora=[]`; 코드에 이 일반 대화 요청용 adapter 지정 없음 |

`versions.json`의 `decoding=case_system_256`은 일반 대화의 실제 출력 한도가 아니다. 일반 대화는 별도 `chat_reply(..., max_tokens=384)`를 사용한다.

## 3. 실제 요청 구성: 공개 입력의 첫 턴

### 3.1 UI → 분류 → 생성

1. `trid/shared_ui/app.js:176–182`: `{question, scope:'workspace', version:선택값}`을 JSON 전송. UI가 system/history를 받아서 임의 조립하지 않는다.
2. `shared_demo.py:2294–2344`: question 문자열(500자 이하)과 version을 검증하고 `session.history`를 복사한다. `request_type` → 필요시 `route`로 chat/search를 정한다.
3. 부모가 **일반 대화 경로**를 확인했다. 이 건을 검색 실패로 재분류하지 않는다. 일반 chat 생성에는 검색 passage, 기관별 규정, A030의 inherited 조건을 넣지 않는다.
4. `route`의 분류 호출은 별도 messages에 `ROUTER_SYSTEM`과 “분류할 입력”을 넣는다(417–439). 최종 chat messages는 별도로 `_messages`에서 구성한다. 분류 system을 최종 chat system으로 전달하는 소스 경로는 없다.
5. 일반 chat은 `CHAT_SYSTEM`을 사용한다. `EXPLAIN_SYSTEM`, 규정 답변 CASE_SYSTEM과 별개다.

### 3.2 정확한 system / history / user

`shared_demo.py:328–330,412–414,464–480`에 따라 새 대화일 때 system은 다음이다.

> 당신은 TRID입니다. 공공기관 직원과 이야기하는 한국어 AI 비서입니다. 사용자의 말에 공감하고, 앞선 대화를 이어서 자연스럽고 구체적으로 답하세요. 대화 도중에 인사나 자기소개를 반복하지 마세요. 한국어로 답하세요. 이 대화는 지금 막 시작되어 앞선 내용이 없습니다. 사용자가 앞에서 한 말을 물으면 아직 들은 적이 없다고 답하세요.

- history: 부모의 새 대화 관측을 전제로 빈 배열. 실제 사용자 세션 메모리를 들여다본 것은 아니다.
- user: 부모가 공개한 메일 다듬기 문장을 그대로 하나의 user content에 둔다. “아래 내용”을 별도 role로 분리하거나 확인 요청을 확정으로 치환하지 않는다.
- role 순서: `system → (user, assistant)* → user`; 매 history 항목에서 role/content만 복사.
- 이후 턴: search history는 일반 chat에서 제외(회상 요청은 예외), chat assistant는 **후처리된 화면 답변의 앞 1500자**를 저장, user는 원문 저장, 전체 최대 24 messages 유지(456–461,2572–2578).
- 컨텍스트 초과 시 오래된 messages 두 개씩 제거하고 재렌더(475–480). 첫 턴 빈 history의 의미 변형을 “옛 대화 누락”으로 설명할 수는 없다. 다만 1500자/24개/컨텍스트 절단은 별도의 다중 턴 정보 손실 가능성으로 남는다.
- `/api/clear`는 document/pin/offered/history를 제거(2286–2293). UI도 clear 성공 후 화면을 비움(`app.js:204–210`).

### 3.3 template → token IDs → completion

`shared_demo.py:442–453`과 `evaluate_external_chat.py:159–163`:

```text
POST /apply-template
  messages = 위 system/history/user
  chat_template_kwargs.enable_thinking = false
→ prompt가 '<think>\n\n</think>\n\n'로 끝나는지 검사(Qwen3 아니면 별도)
POST /tokenize
  content = prompt; add_special = true; parse_special = true
→ token_count + 384 <= 4096인지 검사
POST /completion
  prompt = token ID 배열; n_predict = 384; cache_prompt = false
  seed = secrets.randbelow(2**31)
  temperature = 0.7; top_p = 0.8; top_k = 20; min_p = 0
  presence_penalty = 1.5
```

- `/v1/chat/completions`에 messages를 넘기는 경로가 아니다. template은 생성 요청 전에 렌더하고, 생성에는 token IDs를 넘긴다. 따라서 `/completion`에서 chat template을 다시 적용하는 요청 형태가 아니다.
- service는 `add_generation_prompt` 값을 직접 보내지 않는다. 기존 실측의 assistant/no-thinking 접두와 현재 suffix guard가 기본 동작을 확인하는 근거다. 이번 local 비교에서는 canonical 계약대로 `add_generation_prompt=True`를 명시했다.
- `enable_thinking=False`는 실제 template 렌더 단계에 전달한다. 단순 `/no_think` 사용자 문구나 출력에서 `<think>`를 잘라 내는 방식이 아니다.
- 일반 chat에는 grammar/stop 문자열/ignore_eos를 지정하지 않는다. live 기본 `ignore_eos=false`, EOS는 위와 같음. `repeat_penalty=1`, `frequency_penalty=0`, `repeat_last_n=64`, `dry_multiplier=0`, `mirostat=0`도 기본값에 의존한다.
- live sampler 순서는 `penalties → dry → top_n_sigma → top_k → typ_p → top_p → min_p → xtc → temperature`. 일반 chat은 sampler 목록을 덮어쓰지 않는다. 반면 분류용 `complete`는 temperature=0, seed=42, sampler=[temperature], n_predict=4 및 grammar를 사용한다. **분류의 greedy 설정과 일반 대화 설정을 혼동하면 안 된다.**
- 임의 seed와 생성 token IDs/실효 params/원시 completion을 chat 응답에 남기지 않는다. 따라서 현재 자료만으로 부모 실패 한 건을 seed까지 동결한 재현 입력으로 복원할 수 없다. 실제 로그를 열어 메우지 않았다.

## 4. canonical Qwen 계약 비교

공식 고정 revision:
- [tokenizer_config.json](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/tokenizer_config.json)
- [모델 카드: non-thinking / Best Practices](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/README.md)

| 비교 항목 | 일치 / 차이 / 판단 범위 |
|---|---|
| role 경계 | canonical은 `<\|im_start\|>role\ncontent<\|im_end\|>\n`. 현재 live template도 일반 string message에서 같은 구조. 이번 공개 문장에는 role 특수 문자열·tool_response가 없어 해당 분기 차이가 발동하지 않음 |
| template 원문 | 불일치: live는 역순 index range / message.content 직접 사용, HF는 역순 slice와 string guard 등 사용. 원문 SHA live `57f1fd00…`, HF `a55ee1b1…` |
| 현재 공개 입력의 local render | 두 template 문자열 일치, SHA `5714e9411d6b73af4f7d8cf68003fa730767cfa42442c6aa4d14e9e628fb8e4a`. **로컬 Jinja 비교이며 llama 서버 렌더/토큰 재검증 아님** |
| 기존 실제 render/token 비교 | GENERAL-CHAT-003: 당시 격리 서버, 같은 GGUF/build의 공개 S3 5턴 표본에서 문자열 및 634 token IDs 일치. raw `template_check.server_vs_hf_render_identical=true`, `token_ids_identical=true`. 모든 입력·현재 CPU runtime의 동등성 증명은 아님 |
| thinking | canonical hard-off는 assistant 접두 다음 빈 think 블록. 서비스가 해당 suffix를 검사. 기존 450 생성에서 특수 토큰 누출 0은 과거 표본 한정 |
| 기본 sampling 네 값 | non-thinking 권장 0.7/0.8/20/0과 일치 |
| presence penalty | 공식은 반복 억제를 위해 0–2 조정 가능, 높은 값이 언어 혼합/성능 저하를 낳을 수 있다고 명시. 서비스는 항상 1.5. 권장 범위 안이라는 사실과 이 업무 의미 보존에 적합하다는 주장은 별개 |
| length | 서비스 384/4096은 공식 예제의 넉넉한 출력 한도와 다름. 기존 진단의 해당 CHAT_SYSTEM 조건은 잘림 0이나, 이번 실패의 finish_reason/token 수는 미보유. 짧은 화면 답만으로 잘림을 확정 배제하지 않음 |
| system | 공식 generic 예제와 달리 공감·자연스러운 대화 + 첫 턴 기억 안내가 있음. 역할 몰입이 요청 이행에 영향을 줄 수 있다는 것은 **[INFERENCE] 가설**일 뿐, 이번 확인 요청→확정의 인과 증거 없음 |
| 가중치/런타임 | 공식 Q8_0 GGUF + llama CPU와 HF 고정밀 reference는 같은 실행 조건이 아님. 이번 조사 및 과거 render parity는 logits/수치 연산 동등성까지 검증하지 않음 |

BOS 표시는 live/HF 사이 다르지만 과거 token parity 표본은 실제 중복 BOS가 없었다는 근거다. 이번 입력의 live token IDs를 얻지 않았으므로 이 차이를 전역 오류라고 단정하거나 전역 배제하지 않는다.

## 5. 출력 후처리와 화면

`shared_demo.py:481–512`, `app.js:9–22,113–121`:

- completion content를 strip한 뒤 문장/줄로 분리한다.
- 규정 관련 표현(RULE_CLAIM)과 숫자+단위(UNSOURCED_FACT)가 함께 있고 사용자 제공/계산 가능 수치가 아니면 그 **문장 전체를 제거**한다. 남은 문장은 순서대로 결합하고 연속 빈 줄만 정리한다.
- 모두 제거되면 고정 보류 답, 길이 종료이면 notes를 추가한다. 문장을 새로 바꿔 쓰는 2차 LLM 호출·재생성은 없다.
- 부모의 공개 출력에는 해당 규정 주장 조건이 없다. 특히 `10시`의 `시`는 UNSOURCED_FACT 단위 목록에도 없다. 이 필터가 확인 요청을 “참석 가능합니다”로 치환하거나 새 질문을 생성하는 설명은 소스와 맞지 않는다.
- UI는 `result.answer`를 `textContent`로 출력한다. 일반 대화 답을 번역/요약/완성하는 UI 로직은 없다.
- 단, 이 관측의 원시 completion과 실제 API response 쌍은 보유하지 않는다. 후처리가 어떤 다른 문장을 숨겼는지까지 실측했다고 주장하지 않는다. 삭제가 의미를 훼손할 가능성 일반은 남지만 **새 술어/질문의 생성원**과는 구분한다.

## 6. 배제 범위와 남은 원인

| 가설 | 현재 판정 |
|---|---|
| 검색 corpus/기관 승계가 이 메일 답을 만들었음 | 부모 chat 관측 + chat messages 소스로 이 건의 직접 원인에서 제외. 검색 개선은 해결책 근거가 아님 |
| UI가 확인 요청을 확정으로 다시 씀 | 추적한 app.js에는 그런 변환 없음; 운영 파일 SHA 동일 |
| 후처리가 해당 확정 문장·추가 질문을 생성 | 생성/치환 코드 없음. 제거·보류 경로와 구분 |
| 역할이 한 문자열로 평탄화되거나 system이 user로 잘못 들어감 | 현재 소스·local render는 올바른 role 구조. 실제 실패 요청 trace는 없으므로 모든 runtime 오염을 전역 배제하지 않음 |
| template이 공식과 달라 모든 입력이 잘못됨 | 원문 차이 자체만으로는 성립 안 함. 이번 local render 일치 + 과거 live token 일치. 새 live parser/token parity는 남음 |
| thinking이 켜져서 실패 | false 전달·suffix guard·기존 실측으로 우선순위 낮음. 이번 raw token trace 없는 범위는 명시 |
| history 잘림 | 새 첫 턴이면 해당 원인 아님. 다중 턴의 1500자/24 messages/4096 절단 위험은 남음 |
| sampling/presence/system 영향 | 일부 기존 실패는 영향 있었음. 모든 실패를 해결하지 못함. 새 문구/9조건 sweep 반복 근거 없음 |
| 모델 크기·학습·Q8 양자화·CPU 수치 경로 | 분리 실험 없음. 단정 불가 |
| 메모리 압력 때문에 의미를 바꿈 | 압력 2는 운영 위험이나 의미 오류의 인과 근거 없음. 지연·실패와 의미 변형을 혼동하지 않음 |

GENERAL-CHAT-003의 세부: 처음 버전 재현·번역 핵심 누락은 시험한 모든 설정에서 실패가 남았다. 축약·반복·견과 추천은 설정/표본에 따라 달랐다. “9조건 전부 모든 문항 실패”라는 해석은 잘못이다. 이전 보고의 “template/EOS는 문제가 아니다” 역시 해당 표본 범위로 제한한다.

**추가 확인된 운영 설명 불일치:** 현재 공개 versions 설명은 “최근 6개 메시지”, “greedy”라고 표시하지만 일반 chat 코드는 24개 messages 및 sampling이다. 숫자 필터 설명도 실제 규정 주장 조건보다 넓다. 이는 진단·재현 조건을 오인하게 하는 문서 결함이다. 최신 변경 금지 지시에 따라 운영 설정은 고치지 않았으며, 이 표시 오류가 의미 변형을 직접 일으켰다는 증거도 없다.

## 7. 부모/C에 제안하는 최소 절제 실험 — 미실행

이 제안은 실행 승인이 아니다. 새 인스턴스·학습·모델 교체·프롬프트 sweep 없이, 먼저 남은 서빙 동등성만 검사한다.

1. **생성 0의 계약 검사부터:** 부모 공개 입력의 현재 messages를 그대로 사용해 운영과 동일한 별도 안전 환경의 `/apply-template` + `/tokenize` 결과를 pinned HF와 비교. role delimiters, BOS/EOS, no-thinking prefix, token IDs, 384 여유만 기록. 이번 local Jinja 검사의 남은 llama parser/tokenizer 차이를 분리하는 것이며 실패 답 재생성이 아니다. 압력 2 운영에 새 부하를 주기 전에 부모가 실행 위치/시점을 결정한다.
2. **1이 같고 생성 실험이 별도 승인될 때만:** C가 새 공개 의미보존 문항 2개(확인 요청/확정 진술의 최소 대조)를 정하고 정답 조건을 먼저 고정한다. 동일한 고정 seed·messages·GGUF·sampling으로 서비스 경로와 직접 completion 경로를 비교하되, raw content→API answer→화면 문자열 및 effective settings/stop/token 수를 한 번씩 보존한다. 기존 실패 문구 재현이나 “좋은 seed” 선택을 하지 않는다.
   - direct는 맞고 service만 다르면 요청/history/후처리 차이를 추적한다.
   - 양쪽이 같은 의미 오류를 내면 해당 wrapper 가설만 낮춘다. 그 결과만으로 모델 크기 원인이라고 하지 않는다.
3. **고정밀 reference 비교는 그 다음 별도 승인 사항:** 같은 canonical input의 HF reference와 GGUF runtime 차이를 분리해야 양자화/엔진 가설을 평가할 수 있다. 모델도 함께 바꾸면 원인이 섞인다. 지금 GPU를 기동할 근거로 사용하지 않는다.

실제 부모 실패의 seed/raw trace가 없다는 제한을 인정한다. 기존 사용자 로그를 열거나 알려진 실패를 다시 실행해 빈칸을 메우지 않는다.

## 8. 안전 상태 / 인계

- A032 `2b8ba92` 운영 유지. 배포 후 smoke 근거는 `MODEL-RESUME-A-032.md`에 분리했다. 05:46Z 압력 1 → 05:53Z 압력 2; 공개 페이지는 응답했다. 지속 안정/대화 개선 선언 없음. 자동 rollback·추가 restart 없음.
- A033 fail-fast 수정, CPU 환경 검증, 3차 smoke 모두 보류. 새 유료 자원 0. 기존 smoke **GPU 추론 0 / 인스턴스 종료 2 / 장부 40/230분**.
- 05:52Z operator 조회에서 smoke 2대 terminated. watchdog 종료는 기존 A031 실측. 두 Scheduler는 당시 05:59:01Z / 06:18:23Z, ActionAfterCompletion=DELETE였다. **06:00Z 이후 첫 일정 GetSchedule은 ResourceNotFoundException으로 삭제 확인.** 두 번째는 아직 예정 시각 전이므로 삭제 확인 미완료다.
- A031 수치·로그 보존. A034 HANDOFF의 단정과 인용을 정정했다. 비공개 모델 매핑 인계 없음.

SUPERVISION_REPORT_READY CHAT-ROOT-A-001
