# CHAT-ROOT-C-004 — 기존 로컬 후보 비교와 버전별 대화 계약 연결

## 판정

- **원본 Qwen의 프롬프트만 조금 더 고치는 안은 채택하지 않았다.** 고정한 두 예시를 넣어도 검토 완료·공유 승인 사실을 모두 보존하지 못했다. 추가 문구 튜닝은 중단했다.
- 이미 로컬에 있던 **Gemma-4-E4B Q4_0 + 고정된 CHAT-CAPABILITY-006 계약**은 이번 공개 의미보존 과업 4개를 직접 검토한 범위에서 모두 통과했다. 모델·시스템·샘플링이 함께 다른 후보 묶음의 결과이며, 모델 크기나 특정 요소의 인과 효과가 아니다.
- 실제 `shared_demo` 일반 대화 함수 경로에서도 검토 완료·공유 승인 두 사실을 유지했고, 다음 턴의 두 문장 분리 요청에서도 유지했다. 이후 전역 변수 덮어쓰기 없이 같은 계약을 연결하는 **실제 코드 변경과 재검증까지 완료**했다.
- 코드: `chat-contract-profile-004`, **`78d4c3b`**, 기준 `2b8ba92`. 변경은 `scripts/shared_demo.py`, `docs/SHARING.md` 두 파일이다. 원격 브랜치에 push했다. 운영 versions/기본 모델/사이트는 수정·배포하지 않았다.
- **일반 대화 전체 합격이나 운영 교체 승인은 아니다.** 기존 공개 20문항 평가의 `55/63 FAIL` 판정은 그대로다. E1을 새로 실행하지 않았다. 4개 과업의 결과를 통계적 개선으로 표현하지 않는다.
- **후속 UI 확인:** 기존 두 턴을 실제 로컬 HTTPS/HTTP/화면 경로로 연결했다. 후속 답변은 검토·승인·두 문장을 유지했지만 `팀 외부` 대상을 명시하지 않았다. 연결 성공과 독립 문안의 완전한 의미 보존을 분리한다(6절).
- **사용자 지시에 따른 기준 정정:** 사실 몇 개의 보존만으로 답변 품질을 통과시키지 않는다. 기존 실제 Astra API 참조 20건을 원문과 대조했고, 18건은 개발 참조로 사용 가능, 2건은 문체 보류다. 이는 TRID 점수가 아니다. 의미·대상/조건·자연스러움·요청 이행·히스토리를 함께 판단한다(7절).
- **추가 개발 결과:** native 생각 모드/최종 답변 분리 코드 `8844708`과 실제 UI 연결을 검증했다. 수정 계약의 일부 문안은 개선됐지만 미정 상태에 지원 없는 계획을 붙이는 문제가 남았다. 기존 7B의 안전 실행과 공식 생성 설정 비교도 완료했으나 합격 후보는 아니다. 상세는 8절.

## 1. 문제와 이번 변경의 경계

기존 감사 [C001](CHAT-ROOT-C-001.md), [C002](CHAT-ROOT-C-002.md), [C003](CHAT-ROOT-C-003.md), [A001](CHAT-ROOT-A-001.md)을 이어갔다. template/HF 토큰·role·EOS 검증을 반복하지 않았다. 현재 서비스의 nonthinking 설정, GGUF/runtime, 원모델 실행 차이는 원인 미확정이다. 동일 GGUF와 nonthinking을 공유한 이전 여러 프롬프트/디코딩 실패를 모델 자체 결함의 독립 증명으로 취급하지 않는다.

후보 계약을 검증해도 서비스의 전역 `CHAT_SYSTEM`/`CHAT_SAMPLING`만 쓰면 그 계약 그대로 연결할 수 없었다. 이번 코드는 버전별 선택 필드를 전달하는 작은 연결 변경이다. 원본 모델의 가중치나 생성 능력을 고쳤다는 뜻이 아니다. 다른 버전의 전역 설정을 바꾸거나 업무 검색 수정이 일반 대화 개선인 것처럼 표시하지 않는다.

## 2. 이미 가진 공개 자산만 사용한 비교

| 실행 | 실제 답변 생성 | 관찰과 결정 |
|---|---:|---|
| 기존 Gemma3-4B Q4_K_M, 고정 greedy | 2 | 요청은 유지했으나 확정문에서 검토 완료·외부 공유 정보를 누락하고 확인 요청을 덧붙였다. 채택하지 않음 |
| 원본 Qwen, 고정 두 예시 편집 지시의 새 전이 과업: baseline/candidate | 4 | 수리 완료·부품 보관, 손상 검사 후 조건부 이동은 양쪽 모두 의미 유지. 후보 우위 증거 아님 |
| 같은 편집 지시, 기존 공개 요청/확정 쌍의 변경 조건만 실행 | 2 | 확정문에서 검토 완료·승인 사실 보존 실패. 저장된 baseline 재사용, 동일 baseline 재실행 없음. 후보 폐기 |
| Gemma-4-E4B Q4_0, 기존 006 계약 그대로 | 4 | 요청/완료·승인/수리·보관/조건부 이동 4개 모두 의미 유지 |
| 배포 기준 소스의 실제 함수 경로, 평가 프로세스 안에서만 계약 교체 | 2 | 첫 턴의 검토 완료·승인과 다음 턴의 두 문장 분리 모두 유지 |
| 새 코드의 명시적 버전 프로필 인자, 전역 계약 변경 없음 | 2 | 같은 두 턴 의미 유지. 각 턴에서 전역 기본값 불변 확인 |

답변 생성 **16회**, 마지막 두 실행의 실제 분류 호출 **각 2회**를 더해 모델 completion **총 20회**다. 마지막 두 실행은 같은 2턴의 연결 검증이지 새로운 독립 과업 4개가 아니다. 이전 공개 20문항은 열거나 실행·튜닝하지 않았다. 새로운 문항을 계속 늘리지 않았다.

### 후보 provenance와 계약

- 최신 후보 파일: `runs/base-resume-004/gemma-4-E4B_q4_0-it.gguf`
- SHA-256: `676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee`
- 공개 provenance: [BASE-RESUME-004-RESULT](BASE-RESUME-004-RESULT.md). 기존 보고의 공식 Google QAT GGUF revision `4b4a2c1d584be7264f87aac328a1bc739ce81b6c`, Apache-2.0 자산이다. 파일을 새로 받지 않았다.
- 고정 시스템: [CHAT-CAPABILITY-006/chat-system-candidate.txt](CHAT-CAPABILITY-006/chat-system-candidate.txt). 이번 결과에 맞춰 수정하지 않았다.
- 생성: temperature `1`, top_p `.95`, top_k `64`, min_p `0`, presence_penalty `0`; 평가 seed `1`, nonthinking, 기본 출력 한도 `384`. repeat penalty는 실행 서버 기본값 `1`이다. 실제 서비스의 random seed 정책은 코드에서 변경하지 않았다.
- 원본 Qwen 비교와 모델 외 시스템·샘플링 조건도 다르므로 크기·양자화·system·sampling 개별 원인 분리는 하지 못한다.
- 이전 Gemma3 자산은 [GEMMA-LICENSE-001](GEMMA-LICENSE-001.md)의 내부 비교 범위만 사용했다. 배포·학습·증류에 사용하지 않았다.

## 3. 코드 변경과 실제 검증

### 변경

- `chat_complete(..., sampling=None)`: 미지정이면 기존 `CHAT_SAMPLING`, 지정하면 해당 버전 객체를 사용한다. 일부 키를 기존 객체에 합치지 않는다.
- `chat_reply(..., system_prompt=None, sampling=None)`: 미지정 system은 기존 `CHAT_SYSTEM`. 명시적 일반 설명의 `EXPLAIN_SYSTEM`은 유지한다.
- HTTP handler의 일반 대화 및 검색 실패 후 대화 전환 두 callsite가 관리자 버전 설정의 `chat_system`, `chat_sampling`을 전달한다. 사용자 HTTP 요청으로 system/sampling을 받지 않는다.
- 라우팅, 검색·분류 프롬프트/디코딩, 첫 턴 안내, history 포함·절삭, 근거 없는 업무 숫자 문장 후처리, 출력 한도, 캐시·seed 정책을 바꾸지 않았다. 새 로그·재시도·답변 교체 규칙도 없다.
- 재현 가능한 설정 객체: [chat-profile.json](CHAT-ROOT-C-004/chat-profile.json). 이는 검증 프로필이며 운영 버전 등록/모델 선택 변경이 아니다.

### 관측한 경로

실제 로컬 CPU 모델 서버와 변경 모듈을 실행했다. `request_type` → `route` → `conversation_turns` → `chat_reply` → 후처리 → 표시 응답의 history 저장/절삭 경로를 사용했다. 첫 턴 안내를 유지한 상태에서도 완료·승인 사실이 보존됐다. 다음 턴에서는 그 표시 응답을 history로 넣어 두 문장 분리 요청을 수행했다. 변경 코드 실행은 `CHAT_SYSTEM`/`CHAT_SAMPLING`을 덮어쓰지 않고 새 인자를 전달했다.

**최초 함수 검증에서 제외한 범위:** HTTP 서버·브라우저 UI·CSRF·검색 fallback 실추론·clear/다중 세션·운영 mini. 후속 6절에서 로컬 문서 모드의 정상 HTTP/UI 경로만 추가 확인했다. 공유 검색 모드·CSRF 거절 경계·fallback·clear/다중 세션·운영 mini까지 통과한 것으로 확대하지 않는다. 기존 기관 범위/종료 테스트/알려진 F4 실패는 재실행하지 않았다.

기존 `ChatBackendTests` 중 변경 함수의 기본 안전 동작을 다루는 다음 **5개만 실행, 5개 통과**했다.

- `test_chat_reply_withholds_unsourced_rule_facts`
- `test_chat_keeps_everyday_figures_but_not_rule_claims`
- `test_chat_drops_the_oldest_turns_when_the_conversation_does_not_fit`
- `test_chat_keeps_numbers_the_visitor_gave`
- `test_chat_keeps_arithmetic_of_visitor_figures_but_not_other_numbers`

처음 throwaway runner가 잘못된 테스트 클래스를 선택해 테스트 실행 전에 중단됐다. 클래스 선택을 고친 뒤 위 5개를 한 번 실행했다. 이를 모델/서비스 회귀로 세지 않는다. 전달 인자를 그대로 확인하는 새 mock 테스트는 추가하지 않았다.

## 4. 의미 판정과 독립성 한계

이번 판정은 C의 직접 의미 검토다. 요청을 확정으로 바꾸지 않는지, 이미 완료된 검토·승인을 누락하지 않는지, 조건과 보관 사실을 유지하는지, 없는 요청을 보태지 않는지를 보았다. 단순 문자열 일치/빈 응답 여부로 통과시키지 않았다.

별도 작업자와 label-masked reviewer를 요청했지만 공급자 `429`로 실행 전에 모두 실패했다. 독립 검토 완료라고 주장하지 않는다. 성공한 리뷰 결과가 없으며 재시도하지 않았다. 최신 후보의 4개 통과는 작고 과업이 제한된 관찰이다. 이전 전체 gate를 대체하거나 모델 크기/학습량 부족을 원인으로 확정할 수 없다.

## 5. 자원·보존·인계

- 기존 로컬 llama-server와 CPU만 사용했다. 각 실행 뒤 해당 프로세스를 종료하고 `51992` 포트가 해제된 것을 확인했다. 메모리 pressure는 각 실행 전후 정상 `1`이었다.
- AWS/새 CPU·GPU 인스턴스/운영 mini 접근·배포/다운로드/학습은 하지 않았다. 유효한 GPU smoke 인계나 전송도 수행하지 않았다.
- 비공개 문항·정답·결과·매핑·봉인 자료에 접근하지 않았다. 서비스 영구 로그를 추가하지 않았다. 모델 답변 원문은 `/private/tmp/chat-root-c-004/`의 일시적 평가 결과에만 보관했다. repo에는 system 설정과 원문을 뺀 실행 메타데이터만 남긴다.
- 정리된 증거: [evidence.json](CHAT-ROOT-C-004/evidence.json). 실제 raw/변경 코드 실행은 각각 `latest-results.json`, `service-results.json`, `profile-results.json`; 다른 실행의 파일명은 임시 디렉터리에 보존했다.
- **A/총괄 인계:** `78d4c3b`는 버전별 계약 연결 코드로 검토·통합 가능한 상태다. 운영 설정이 없으면 기존 계약이 유지된다. 최신 후보로의 운영 기본값 전환은 포함하지 않았고, 전체 gate 미통과를 무시한 자동 교체를 하지 않는다. 추가 대규모 평가나 20문항 맞춤 튜닝 대신, 이번 관찰과 기존 E1/전체 gate의 차이를 승인된 개선 판단에 사용해야 한다.

## 6. 후속: 기존 두 턴의 실제 로컬 HTTP/UI 연결

새 문항 대신 3절의 두 턴을 그대로 사용했다. 실제 `State`가 기존 로컬 추출 독해기와 후보 llama-server를 로드하고, 변경 모듈의 HTTP handler와 원래 UI 자산을 실행했다. 임시 loopback HTTPS proxy로 원래 proxy 인증 헤더를 전달했으며 브라우저는 실제 세션 cookie·CSRF token으로 요청했다. 공개 안내 한 문장을 UI에서 문서로 붙여넣어 **문서 모드의 일반 대화** 경로를 사용했다. 공유 검색 index는 로드하지 않았으므로 공유 검색 모드 실증은 아니다.

- seed를 1로 고정하지 않고 **운영 코드의 무작위 seed 정책**을 그대로 두었다. 개별 seed 값은 종료 때 최종 기록이 중단되어 보존하지 못했다.
- 첫 응답: 검토 완료·팀 외부 공유 승인 유지. 두 번째 응답: 검토 완료·공유 승인·두 문장 유지, **팀 외부라는 명시적 대상 표현은 생략**. 승인을 뒤집거나 새 확인 요청을 만든 것은 아니지만 독립적으로 사용할 편집 문안의 모든 수식 조건을 보존했다고 판정할 수 없다.
- 두 API 응답 모두 `intent/status=chat`, `checked_chunks=0`, 후보 인용 없음, 숨김 메모 없음. 화면의 두 답변 모두 `일반 대화 · 문서 근거 없음 · 검증 대상 아님`으로 표시됐다. JavaScript 오류 목록은 비어 있었다.
- 실제 답변 2회와 분류 2회. 임시 llama 로그에 완료된 generation task 4개가 남았다. **이번 보고의 로컬 completion 누계는 24회(답변 18, 분류 6)**이며, 서로 다른 독립 문항 24개라는 뜻은 아니다.
- 새 의미 튜닝이나 재생성은 하지 않았다. 전체 문항 평가·운영 배포·학습도 없다.

증거: [http-ui-evidence.json](CHAT-ROOT-C-004/http-ui-evidence.json). 원문 두 응답과 UI 텍스트는 `/private/tmp/chat-root-c-004/http-ui/`, 화면 캡처는 해당 evidence의 임시 경로에만 보관했다.

**실행 한계/정리:** 처음 CLI 탐색은 sparse worktree의 import 경로 누락으로, 첫 임시 runner 시작은 저장된 metadata의 JSON 앞 설명 문자열 때문에 추론 전에 중단됐다. 환경 경로·snapshot 파싱만 고친 뒤 실제 두 턴을 한 번 수행했다. 종료 시 반복 interrupt가 임시 runner의 `backend.shutdown`을 끊어 종합 snapshot(history/template/seed)이 생성되지 않았다. 이를 얻으려고 다시 실행하지 않았다. 이미 저장된 응답·화면·runtime task 로그로 위 범위만 판정했다. 별도 확인에서 `51992/51993/51994` listener가 모두 없고 메모리 pressure는 `1`이었다. 임시 TLS private key를 삭제했다. 종료 suite를 재실행한 것이 아니라 이번 실행 자원을 정리한 것이다.

## 7. 사용자 우선 지시: Astra 참조로 개발 기준 정정

기존 [ASTRA-REFERENCE-001/teacher-responses.json](ASTRA-REFERENCE-001/teacher-responses.json)은 실제 API 응답이며 `status=200`, 요청·반환 모델 모두 `codex/gpt-6-astra`다. 개발 20건의 reference가 실제 batch 응답의 같은 ID/답변과 일치하는지 모두 확인했다. 이 세트는 기존 공개 20문항 관문과 다른 **공개 합성 개발 문항 D01–D20**이며 비공개 평가셋이 아니다.

원문·히스토리와 직접 대조한 정리: [astra-reference-review.json](CHAT-ROOT-C-004/astra-reference-review.json).

- 18건: 의미와 요청 형식, 문체를 개발 참조로 사용할 수 있다.
- D05: 발신·수신 주체는 맞지만 자기 직함을 굳이 설명하는 표현이 어색해 문체 보류.
- D14: 요청한 장소 표시는 맞지만 연결 어미 뒤에 명사형 표지를 붙인 문장 마무리가 부자연스러워 문체 보류.
- Astra가 작성했다는 이유만으로 20건 모두 완성된 골드로 채택하지 않았다. 모델 참조도 원문 충실도와 자연스러움 검토가 필요하다.
- 참조 생성에 invariants가 함께 제공됐으므로 비보조 자연 대화 실력이나 독립 benchmark 점수로 해석하지 않는다. 학습 승인은 0이며 SFT 입력으로 넘기지 않았다.

### 실제 개발 판정에 반영한 항목

1. **의미:** 화자·수신자·대상 범위·행동·완료/미확정·부정·조건·날짜·수량을 유지해야 한다. `검토`, `승인` 같은 키워드 존재만으로 통과시키지 않는다.
2. **요청 이행:** 바꾸라고 한 부분만 바꾸고 나머지 사실을 유지한다. D15의 두 문장 분리, D17–D18의 부분 수정처럼 후속 편집에서도 남겨야 할 조건을 확인한다.
3. **자연스러움:** 사용자가 바로 사용할 수 있는 문장인지 별도 판정한다. 어색한 자기소개, 끊긴 문장 마무리, 불필요한 설명을 의미 정확성과 구분한다.
4. **지원 없는 추가:** 원문에 없는 확정·기한·행동·새 요청을 금지한다. 관례적 공손 표현과 새로운 사실 주장은 구분한다.
5. **히스토리:** 앞선 문안을 참조하되 독립 문안에 필요한 대상 수식을 잃으면 완전 통과가 아니다. 따라서 6절 후속 답변은 사실 일부 보존과 대상 표현 누락을 분리해 **부분 미충족**으로 기록했다.

이는 Astra 문장과의 글자 일치 검사가 아니다. 의미가 같은 자연스러운 대안은 허용하며 참조 대비 차이를 원문·요청에 근거해 설명해야 한다. 기존 비교의 4건 의미 통과를 Astra 수준의 품질 합격으로 승격하지 않는다.

**동일 UI 문항의 별도 참조 요청:** 일반 task의 명시 모델 지정은 도구가 거절해 실행되지 않았다. `completion(model=default)` 요청은 응답을 반환했지만 반환 모델 ID를 제공하지 않아 이를 확인된 Astra 골드로 승격하지 않았다. 출력은 일시적 artifact에만 보관했다. 위 20건의 확인된 Astra provenance와 섞지 않는다.

**개발 인계:** `78d4c3b`의 버전별 계약 연결은 실제 UI까지 연결됐지만 후보의 전체 품질 합격은 아니다. 이번에 확정한 Astra 참조 기반 판정과 두 문체 보류 항목을 다음 후보 판단에 사용한다. 특정 실패 문구를 service prompt에 추가하거나 기존 관문 20문항에 맞춰 튜닝하지 않았으며, 모델 크기·학습량 원인 단정과 자동 배포도 하지 않았다.

## 8. Astra 참조 기반 후속 개발과 실제 연결

### 코드와 검증

- 별도 브랜치 `chat-native-reasoning-005`, commit **`8844708`**, parent `78d4c3b`를 push했다. `scripts/shared_demo.py`, `tests/test_shared_demo.py`, `docs/SHARING.md`만 변경했다. 운영 버전·기본 모델·사이트는 그대로다.
- 버전별 `chat_thinking`, `chat_max_tokens`, `chat_reasoning_budget`를 실제 대화 경로에 연결했다. 생각 모드에서는 native chat API의 최종 `message.content`만 반환하며, 최종 답변이 없으면 reasoning을 대체 답변으로 노출하지 않는다. 기존 nonthinking 기본 경로는 유지한다.
- 관련 행동 테스트 **7개 통과**: 새 추론 비노출/최종 답변 부재 처리 2개와 기존 수치·계산·히스토리 등 회귀 5개. 전체 suite 통과를 주장하지 않는다. 별도 코드 reviewer는 provider429로 실행되지 않았다.
- 실제 `State`·HTTP handler·로컬 HTTPS·브라우저 화면까지 실행했다. 각 2턴 UI 실행에는 답변 2회 외 분류 2회가 있어 서버 generation은 각각 4회다. wrapper 누락분은 저장된 서버 로그로 확인했으며 집계를 위한 재실행은 하지 않았다.

### 후보별 판정 — 서로 다른 계약의 점수를 합치지 않음

| 후보/단계 | 관찰 범위 | 판정 |
|---|---|---|
| 기존 계약에 보존 지시를 추가하고 native 생각 모드 사용 | 채택된 Astra 개발 참조 18건; 완료된 답변 재사용 포함 | 의미·형식·자연스러움 모두 통과 10건, 의미 통과 13건. D03 원문 재검토에 따른 집계 정정이며 모델 개선 수치가 아니다. 지원 없는 절차·운영 계획 등이 남아 미채택 |
| 실제 행동 제한과 문안 속 화자를 분리한 수정 계약 | 공개 D03/D06/D12/D20 | 3건 완전 통과. D20은 미정인 마감에 원문 밖 향후 안내 계획 등을 붙여 미충족 |
| 수정 계약의 실제 UI | D06 작성과 두 문장 분리 후속 | 관찰한 두 턴의 주체·행동·형식 유지. random seed 정책은 그대로. 전체 대화 또는 독립 평가 합격 아님 |
| 원본 Qwen3 Q8의 native 생각 모드 비교 | 동일 공개 2건 × on/off; weights/system/history/sampling/seed/API/총 한도 공유 | 두 설정 모두 채택할 문안 품질에 못 미침. 크기·가중치·학습량·양자화의 단독 원인 증명 아님 |
| 기존 Qwen2.5 7B, 안전 실행 후 greedy 및 공식 생성 수치 적용 | 동일 공개 4건 × 두 설정 | 역할 누락/방향, 작성 설명 덧붙임, 알려진 담당 부서까지 미정으로 읽히는 오류가 남음. 조건부 나머지 14건은 실행하지 않음 |

수정 계약은 [writing-contract-candidate.json](CHAT-ROOT-C-004/writing-contract-candidate.json), 완료된 판단·메타데이터는 [native-thinking-development-evidence.json](CHAT-ROOT-C-004/native-thinking-development-evidence.json)에 보관한다. 이 파일은 **미배포 개발 후보**이며 자동 적용 설정이 아니다.

### 폐기한 접근과 원인 해석

- 일반 교정 추가, greedy 전환, 같은 모델의 원문 차이 검사, 불확실성 보존 문구 추가는 채택하지 않았다. 새 문항이나 실패 문구 전용 규칙으로 점수를 맞추지 않았다.
- 구조화 parser는 동일한 공개 문안 4건과 기존 계산 대조 1건에서 nonthinking/생각 모드를 비교했다. 생각 모드에서 미정 상태 추출은 나아졌다. 초기에는 D03의 부정 설명을 작성 지시 유입으로 판정해 후속 생성을 보류했지만, 해당 설명은 실제로 원문의 '다음 사실' 뒤에 포함되어 있다. **그 포함만으로 의미 오류라는 판정을 정정한다.** 초기 단계의 후속 답변 생성은 0회, 서비스 통합은 0회다. 내부 분류나 무시되는 필드만으로 실행하지 않은 최종 답변의 회귀를 주장하지 않는다.
- 수정 계약 D20의 저장된 native 응답은 completion 23 tokens, 정상 stop, reasoning 없음이다. 이 실패를 reasoning cap 384 소진 탓으로 돌릴 근거가 없어, 더 큰 budget으로 같은 사례를 실행하지 않았다. 생각 모드를 허용하는 것과 모든 턴에 실제 reasoning이 생성되는 것은 다르다.
- D03 후보의 의미 판정을 partial → pass로 정정했다. 자연스러움 partial은 유지하여 3축 모두 통과한 수는 10/18로 변하지 않는다. 모델 재실행이나 참조 답변 수정 없이 실제 원문에 맞춘 정정이다. 원문 밖 향후 계획을 붙인 D20과 행동 방향을 바꾼 문안은 여전히 미충족이며, Astra 문장을 그대로 재현해야 한다는 기준을 적용하지 않는다.
- 기존 7B의 첫 세 시작은 pressure 2에서 추론 없이 종료됐다. 이후 stdlib driver·context 2048·batch 128에 `--no-repack`을 적용해 pressure 1에서 실제 비교를 완료했다. 당시 모델 RSS는 4,821,760 KiB였다. 이는 실행 가능한 구성을 찾은 것이지 호스트 부하를 통제한 메모리 단독 원인 증명은 아니다.
- 공식 생성 수치는 [Qwen의 고정 revision 설정](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/a09a35458c702b33eeacc393d103063234e8bc28/generation_config.json)을 llama.cpp 설정으로 옮겼다. HF 원본 정밀도/런타임과 동등하다는 뜻은 아니다. 서버 설정 float32를 정확한 소수와 비교한 준비 단계 오류는 추론 0회였고, 허용 오차를 수정한 뒤 처음 실제 비교했다.

### 배포 경계

**현재 운영 배포 불가 판정을 유지한다.** 연결 성공, 일부 문안 통과, 메모리 문제 해소는 전체 품질 합격을 대신하지 않는다. 기존 공개 관문 `55/63`, `14/20`, FAIL도 이 개발 결과로 뒤집지 않는다. 최선 수정 계약에 앞선 다른 계약의 18건 점수를 붙이지 않는다.

모델 최종 해설/답변 원문은 일시적 평가 artifact에만 보관했다. 내부 reasoning 원문은 저장·노출하지 않았다. UI의 일반 대화/근거 없음 표시는 유지했고 `verified` 표시는 추가하지 않았다. 생성 서버·브라우저·임시 TLS 키는 각 실행 종료 절차로 정리했다. 이 절의 실행에서는 비공개 문항·골드·매핑 접근, 기존 기관/종료/F4 반복, 기존 20 맞춤 튜닝, 신규 학습·모델 다운로드·유료 인스턴스·운영 mini 접근을 하지 않았다. 이 작은 공개 개발 세트로 통계적 모델 개선이나 모델 크기 한계를 주장하지 않는다. 이후 공식 원본 다운로드는 13절에서 별도로 기록한다.

## 9. 이미 있던 원본 가중치로 공유 실행 조건 분리

원본 `runs/open-weight-base/qwen3-1.7b-hf-rev70d244cc` 가중치가 로컬에 있었다. 새 다운로드나 학습 없이 원본 BF16/Transformers와 같은 원본에서 변환한 Q8/llama.cpp를 비교했다. [MODEL-REPAIR-002](MODEL-REPAIR-002.md)의 변환 텐서 일치 기록을 재사용했으며 기존 template/HF 634-token 검증은 반복하지 않았다. 이 Q8의 SHA-256은 `f2ba848c65c2e135495656fc91774b82a10d1f9267671609fad54684c9db1c38`이며, 운영 파일 `061b54da…`와 혼동하지 않는다.

### 같은 token ID를 공급한 직접 비교

원본은 CPU bfloat16·SDPA, torch `2.14.1`·transformers `5.21.0`으로 실행했다. 누락/예상 밖/불일치 가중치 및 로드 오류는 없었다. 두 backend에 같은 HF-rendered token ID를 공급하고 nonthinking·greedy·출력 한도 384를 고정했다. 짧은 한국어 도우미 system과 수정 서비스 계약을 각각 사용했다. 이는 최적 품질 설정 평가가 아니라 입력/실행 비교다.

| 공개 입력 | 원본 BF16 | 동일 원본 Q8 |
|---|---|---|
| 최소 system / D06 | 보완 주체 누락, 수신자용 본문보다 간접 요청 표현 | 같은 최종 답변 |
| 최소 system / D20 | 재무팀과 미정 상태는 있으나 신청 담당 부서라는 역할이 명확하지 않음 | 같은 최종 답변 |
| 수정 계약 / D06 | 주체·행동은 유지하지만 문안 밖 설명을 붙임 | 주체·행동은 유지하지만 설명과 두 문장 문안을 출력 |
| 수정 계약 / D20 | 미정 상태는 유지, 담당 부서 누락 | 같은 최종 답변 |

총 8회 모두 EOS로 끝났고, 대응 4쌍의 입력 hash/입력 토큰 수가 일치했다. 최종 답변이 같은 쌍은 3개다. **이 공개 변형의 누락·형식 문제는 GGUF나 서비스 후처리 없이 원본 직접 생성에서도 발생한다.** 나머지 한 쌍은 문장이 달라졌으므로 양자화/런타임의 영향이 전혀 없다는 결론은 아니다. 정밀도와 runtime을 각각 독립 분리한 실험도 아니다. 값 일부 보존·역할 불명확·실제 모순·형식 미충족을 구분하며, 단순 누락을 주체 반전으로 과장하지 않는다.

### 원본의 생각 모드 추가 분리

같은 최소 입력·원본 BF16·backend에서 sampling `.6/.95/top_k20/min_p0`, request별 seed `1729`, 총 한도 2048을 고정하고 생각 모드만 off/on으로 비교했다. [원본 모델 지침](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc/README.md)이 thinking에서 greedy를 금지하므로 이를 쓰지 않았다. nonthinking 최적 수치를 찾는 실험은 아니다.

동일 공개 2건 × 모드 2개, 4회 모두 EOS 종료다. 켠 두 응답 모두 reasoning을 실제 생성했지만, 각 모드 쌍의 최종 답변은 같았고 위의 문안 품질 문제가 남았다. reasoning은 메모리에서 버리고 길이와 존재 여부만 기록했다. 어느 모드도 선택 기준을 충족하지 않아 조건부 GGUF 추가 실행은 0회다.

두 입력·한 seed의 개발 진단일 뿐이다. 운영에서 관측한 원래 메일 오류의 개별 원인, 모델 크기나 학습량 한계, 다른 설정/모델 전체의 능력은 확정하지 않는다. 다만 이 결과를 무시하고 RAG 수집량, GGUF 또는 생각 모드 스위치 하나만 고치면 해결된다고 약속할 근거는 없다.

CPU loader 준비 중 optional `accelerate` 요구와 set형 진단 메타데이터 JSON 처리 오류가 각각 있었다. 모두 생성 0회였으며 기본 CPU 경로와 메타데이터 직렬화를 수정했다. 패키지는 설치하지 않았다. 완료 실행은 pressure 1을 유지했고 프로세스/포트는 종료됐다.

## 10. 추가 지시 대신 검증한 구조와 실제 최종 출력

- 같은 간결한 작성 계약을 기존 원본과 Gemma에서 각각 공개 4건으로 확인했다. 원본은 알려진 담당 부서를 누락했고, Gemma는 요청한 완성 문안 대신 여러 대안·설명을 붙였다. 두 후보 모두 채택하지 않았다.
- 저장된 초안의 원문 차이 검사에도 native 생각 모드를 적용했다. 메모리 가드를 그대로 유지하고 `--no-repack`, batch 128로 4건을 완료했다. 모두 실제 reasoning이 있었지만, 원문 밖 향후 계획을 포함한 D20을 통과시켰다. 자동 수정이나 서비스 검증기로 채택하지 않았다.
- D03 원문 해석을 정정한 뒤, **기존 parser 결과를 고치거나 다시 생성하지 않고 실제 최종 출력까지 검증했다.** 소비되는 source-text 3건과 계산 bypass 1건만 새로 생성했다. 변경 없는 D06 bypass는 반복하지 않았다.
- D12의 원칙·예외·한 문장과 계산 대조의 `8 + 2 = 10만원`은 유지됐다. D03은 제공된 부정 설명을 포함한 점이 아니라 이를 불확실한 해석으로 바꾸고 요청 자체에 답한 점이 미흡했다. D20은 향후 안내 계획을 없앴지만 담당 부서를 새 문의 창구 지시로 바꿨다. 따라서 내부 JSON이 정돈됐다는 이유로 통과시키지 않았고 서비스 통합도 하지 않았다.
- 이 소비 경로 4건은 모두 정상 stop, 종료 후 pressure 1·포트 해제였다. 결과는 `native-thinking-development-evidence.json`의 `source_plan_consumer_probe`와 `source_checker_thinking`에 기록했다.

## 11. 기존 한국어 후보의 종료 경계 분리

기존 `open-weight-lora-kanana-01.q8_0.gguf`도 확인했다. SHA-256은 `82779b84ea7eab86579c13e8a3c7ec962b69fac8be014d9fa7eca2ea7adbf31e`다. 이는 Kanana 1.5 2.1B 원본 자체가 아니라 **이미 있던 grounded-QA 250-step LoRA 산출물**이다. 신규 학습·다운로드·복원·배포는 하지 않았다. [고정 원본 revision](https://huggingface.co/kakaocorp/kanana-1.5-2.1b-instruct-2505/tree/7df4bc35ccd610e451809d7106e1c3cf82bfd44c)의 공개 라이선스 확인은 전체 후속 학습 데이터 권리 감사와 구별한다.

초기 native 4건은 모두 본문 끝에 `<|eot_id|>`를 노출했다. 종료 문자열과 별개로 D06의 항목 안내 방향도 잘못됐다. GGUF의 EOS ID는 `128009`, 해당 토큰은 CONTROL이며 template도 같은 종료 기호를 쓴다. EOS 설정을 잘못된 값으로 덮어쓰지 않았다.

새 후보의 프로토콜 경계를 별도로 확인했다. EOS `128009` 하나만 강제 생성한 native 응답은 `stop_type=eos`, `chat_format=peg-native`인데도 해당 기호를 본문에 반환했다. `preserved_tokens`에도 그 ID가 있었다. **모델이 일반 문자 조각으로 기호를 지어낸 문제가 아니라, 시험한 native 응답 경계의 노출 문제로 분리됐다.** 특정 PEG 내부 분기까지 추적했다고 주장하지 않는다. 관련 고정 소스는 [특수 토큰 보존](https://github.com/ggml-org/llama.cpp/blob/8ca4060/tools/server/server-context.cpp#L4812-L4815)과 [chat 응답 직렬화](https://github.com/ggml-org/llama.cpp/blob/8ca4060/tools/server/server-task.cpp#L412-L457)다.

반면 실제 `chat_reply(thinking=False)` → `/completion`에서는 D03 입력 649 tokens, 생성 11 tokens, 마지막 ID `128009`, 정상 EOS로 끝났으며 raw 응답과 표시 문안 모두 기호가 없었다. **문자열 사후 삭제·가중치/메타데이터 수정 없이 기존 실제 경로가 정상 처리했다.** 새 경계 진단이지 기존 기관/종료/F4 관문 반복이 아니다. 다른 모델에 native 모드를 자동 적용하지 않도록 격리 브랜치의 `docs/SHARING.md`에 범위를 명시했다.

초기 임시 계획에는 부모 계획에서 복사된 미사용 Qwen family/metadata/manifest 설명이 남아 있었다. 이는 Kanana 출처 근거로 무효이며 정정했다. 실제 모델 경로·파일 SHA·server log·GGUF header·해당 학습 메타데이터만 근거로 사용한다. 원본 대비 미세학습 효과나 모델 크기의 인과 비교로 해석하지 않는다.

## 12. 간결 계약과 실제 Astra 예시의 전이 확인

같은 간결 계약을 기존 Kanana 산출물의 실제 nonthinking 경로에서도 4건 실행했다. D03은 두 문장과 새로운 재확인 불필요 지시, D06은 보완 주체 누락, D12는 신청 원칙을 검토 행위로 변경, D20은 요청한 완성 문장 대신 라벨을 출력했다. 남은 참조와 UI로 승격하지 않았다.

이어 실제 Astra 응답 D01·D08·D10·D17을 네 개의 문체 예시로 고정했다. 예시 문항은 평가에서 제외했고, 대상 D03·D06·D12·D20의 참조 답변이나 invariants는 모델에 주지 않았다. **같은 corrected writing contract와 예시를 기존 Kanana·원본 Qwen·Gemma에 각각 적용했으며, 가중치 학습은 하지 않았다.**

| 기존 모델 | 실제 4건에서 남은 문제 | 판정 |
|---|---|---|
| Kanana 학습 산출물 | 재확인 불필요 지시 추가, 보완 주체를 확인 요청으로 변경, D20 라벨 형식 | 미채택 |
| 원본 Qwen | D06의 행동 방향 변경·간접 설명, D20의 담당 역할 불명확·새 확인 요청 | 미채택 |
| Gemma | D03의 새 불필요 지시, D20의 원문 밖 향후 공지 계획 | 미채택 |

세 후보 모두 1차 네 문항을 통과하지 못해 조건부 잔여 10건은 실행하지 않았다. 예시 네 건을 포함한 18건 합격률이나 독립 평가 점수를 만들지 않았다. 위 신규 16회 생성은 실제 `chat_reply` 경로에서 완료했으며 각 실행 후 pressure 1·포트 해제를 확인했다. 참조 문장을 그대로 복사해야 한다는 기준이 아니라 사실·역할·상태·요청 형식·자연스러움으로 판정했다. 이 결과로 prompt 예시 추가가 근본적인 사실 충실성을 해결했다고 주장하지 않는다.

## 13. 공식 한국어 원본 확보

기존 한국어 후보는 원본이 아닌 과거 미세학습 산출물이므로 원본 능력의 대용으로 판정하지 않는다. 로컬 cache에는 해당 모델의 `refs/main`만 있고 가중치가 없음을 확인했다. 운영 배포 가능한 수준까지 계속 개발하라는 최신 요청 범위에서, 유료 자원이나 GPU를 만들지 않고 [공식 공개 원본](https://huggingface.co/kakaocorp/kanana-1.5-2.1b-instruct-2505/tree/7df4bc35ccd610e451809d7106e1c3cf82bfd44c)을 고정 revision으로 받았다.

- 원본 revision: `7df4bc35ccd610e451809d7106e1c3cf82bfd44c`, 공개 Apache-2.0.
- 가중치 SHA-256: `9a9d12e01656388e49114191ab235b8bec3b2214264341732798c698f0dd039f`. 가중치와 tokenizer의 실제 hash·크기를 공식 LFS 메타데이터와 대조해 일치했다.
- 다운로드 8개 파일, 합계 `4,650,978,030` bytes. 인증 토큰을 사용하지 않았다.
- 원본 확보만으로 품질 합격을 주장하지 않는다. 새 학습·운영 접근·배포는 하지 않았다. 이 단계까지 모델 다운로드는 1회이며 앞 절의 기존 파일만 사용한 실행과 구분한다.

### 변환과 실제 대화 결과

고정 converter의 Q8_0 직접 변환을 CPU·임시 파일 모드로 완료했다. 결과는 `2,469,819,168` bytes, SHA-256 `ebba2cdcbebd6a88ab418d243c8ee0b427f690020d9042776623241efbdfb427`이다. 준비 단계에서 `sentencepiece` 누락과 Transformers 5.21의 명시적 head dimension 설정 거부가 발생했다. 두 번 모두 추론은 0회다. converter의 명시된 호환 tokenizer 의존성을 임시 `PYTHONPATH`에 격리해 해결했으며, 원본 config·검증 코드·프로젝트 runtime을 수정하지 않았다.

같은 수정 계약과 실제 `chat_reply(thinking=False)`로 4건을 완료했다. D03은 접수 완료와 한 문장을 지켰다. 그러나 D06은 안내 방향을 바꾸고 새로운 문의 지시를 넣었으며, D20도 요청한 신청 안내 대신 문의·상세 안내를 받으라는 지시를 생성했다. **원본 역시 미채택**이며 조건부 잔여 14건은 실행하지 않았다. 모두 EOS 종료, 제어 기호 없는 최종 답, pressure 1·포트 해제였다. 과거 미세학습 산출물만의 문제라고 단정할 수 없지만, 미세학습·양자화·모델 크기 각각의 인과 효과를 분리했다고 주장하지 않는다.

## 14. 최신 소형 후보의 권리·실행 조건 확인

Kanana-2-3B도 확인했지만 [라이선스 3.1·4.1](https://huggingface.co/kakaocorp/kanana-2-3b-instruct/blob/main/LICENSE)은 표시 의무와 API/cloud·SI/on-premise 등 명시된 제3자 제공 유형의 별도 상업 라이선스를 요구한다. 이를 Apache-2.0과 같다고 취급하거나 사용자 대신 상업 사용 승인을 가정하지 않았다. 해당 가중치는 받지 않고, [공식 Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)의 Apache-2.0 후보를 선택했다. 공개 benchmark는 이번 문안 품질의 증거로 사용하지 않는다.

[Unsloth 고정 GGUF release](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/e87f176479d0855a907a41277aca2f8ee7a09523)의 Q8_0 파일을 받았다. `4,482,403,488` bytes, SHA-256 `10cc391b403021dd11c614679d2fd92f611c3681d29e29651b717316965d61e1`이며 실제 bytes를 발행자의 LFS hash와 대조했다. 공식 원본과의 tensor parity까지 확인했다고 주장하지 않는다. vision projector·새 학습·운영 배포 없이 text 경로만 사용했다. 모델 다운로드는 이 단계까지 총 2회다. 고정 [llama.cpp architecture 표](https://github.com/ggml-org/llama.cpp/blob/8ca4060/src/llama-arch.cpp)에도 `qwen35` 지원이 있다.

같은 수정 계약·같은 네 공개 입력으로 공식 mode별 sampling 수치를 사용한 두 구성을 먼저 고정했다. GGUF와 공식 template의 기본 thinking 설정은 다를 수 있으므로 실제 `/apply-template`의 명시적 mode별 접미부 일치를 확인한 뒤에만 생성했다. 이는 모델·sampling·출력 예산을 포함한 실제 구성 비교이지 thinking 하나만의 인과 실험이 아니다.

| 관측 구성 | CPU 단건 최소 초 | CPU 단건 최대 초 | 실제 최종 출력의 판정 |
|---|---:|---:|---|
| nonthinking / presence penalty 1.5 | 5.022 | 6.365 | D06의 안내 방향 변경, D20의 새 직접 신청 지시로 미채택 |
| thinking / reasoning budget 384 | 28.994 | 29.730 | 핵심 역할·미정 상태는 일부 보존하나, 메일의 주어 연결 오류와 원칙을 가능 여부로 바꾼 문안으로 미채택 |
| nonthinking / presence penalty만 0으로 변경 | 5.036 | 6.010 | D03·D12·D20은 통과. D06은 항목 안내를 항목 제출로 바꿔 미채택 |

각 4개 단일 seed 관측일 뿐 percentile이나 운영 GPU 지연으로 일반화하지 않는다. 이 세 구성의 12회는 정상 종료·pressure 1·포트 해제를 확인했다. 내부 reasoning은 별도 field 존재만 확인하고 원문은 버렸다. 잔여 14건이나 독립 평가 점수로 승격하지 않았다.

이후 precision-oriented sampling과 reasoning 강제 종료 제거를 묶어 한정된 실제 예산에서 확인했다. D03 한 건이 총 출력 한도 2,048 tokens를 모두 사용하고 `finish_reason=length`, 최종 본문 없음으로 끝났다. 실제 앱은 `Native chat returned no final answer` 오류를 반환했으며 내부 추론을 답변으로 대신하지 않았다. 나머지 세 건은 실행하지 않았고 pressure 1·포트 해제를 확인했다. 이 구성도 미채택이며, 출력 예산 확대만으로 해결됐다고 주장하지 않는다.

## 15. 같은 메모리 가드에서 확인한 9B 후보

[공식 Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a)의 Apache-2.0 모델에 대해 [고정 Unsloth Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/3885219b6810b007914f3a7950a8d1b469d598a5)을 확보했다. 실제 파일은 `5,680,522,464` bytes, SHA-256 `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8`로 발행자 LFS 값과 일치했다. 이 단계까지 모델 다운로드는 3회다. 원본 tensor parity 검증이나 파라미터 수만의 인과 비교가 아니라 모델·양자화 패키지의 비교다.

GGUF의 embedding 4096, 32 layers, vocabulary 248320, 저장 tensor elements 8953803264를 확인했다. 처음 EOS 검사에서는 model config의 endoftext `248044`와 chat tokenizer의 im_end `248046`을 혼동했다. [공식 tokenizer metadata](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/tokenizer_config.json)를 확인해 검사를 고쳤으며 모델 EOS를 덮어쓰지 않았다. 진단 JSON의 uint64 직렬화와 명령 경로 오타도 추론 전에 바로잡았다.

| 실제 확인한 구성 | 네 진단 문항에서의 잔여 문제 |
|---|---|
| 기존 수정 계약·nonthinking·greedy | 세 건 통과. D06은 내부 담당자를 항목 전달 수신자로만 적고 보완 책임을 누락 |
| 기존 간결 계약 | D06의 첨부·재제출 추가, 서문·인사말을 포함한 형식 위반 |
| 기존 실제 Astra 네 예시 포함 계약 | D06의 연결이 어색하며, D20은 담당 부서를 신청 경유 창구로 변경 |
| 새 compact 계약 | D06의 안내 방향 변경, D20의 신청 경유 창구 추가 |
| 기존 수정 계약·precision thinking·budget 768 | D03·D06·D12는 통과하나 D20에 원문 밖 추후 공지 확인 지시 추가 |

첫 세 구성은 stateless 보조 검토도 받았다. 검토는 D06의 실제 보완 책임과 단순 수신자 표기를 구별했고, 초기 계약의 D20 담당 업무 서술은 허용 가능한 바꿔 쓰기로 판단했다. 반면 예시 계약의 신청 경유 창구 추가는 구별해 지적했다. 이 보조 검토를 새로운 Astra provenance나 독립 release 평가라고 부르지 않는다.

위 20회는 모두 실제 `chat_reply`에서 완료했고 정상 종료·pressure 1·포트 해제를 확인했다. thinking 구성의 네 CPU 관측은 63.836–65.605초였으며 운영 GPU 성능을 측정한 것은 아니다. **서로 다른 구성에서 잘된 답만 골라 합격 후보를 만들지 않는다.** 아직 모든 진단을 통과한 단일 구성은 없으며 잔여 참조·실 UI·운영 승인으로 승격하지 않았다.

구조화된 원문 계약의 이전 parser도 같은 9B에서 실제로 비교했다. 기존 JSON 입력과 native role 입력의 nonthinking 각 5건은 문안 요청을 일반 응답으로 오분류했다. native role 입력에 reasoning 384를 적용한 5건은 문안/계산 구분은 맞췄지만, D06에서 언급하지 않은 수신자·제목·자료 내용을 미정이라고 만들어 넣었다. 이 15회는 생산자 진단이며 최종 문안의 end-to-end 통과 수에 넣지 않았다. 계약을 수동으로 고쳐 소비자 입력으로 승인하지도 않았다.

중복된 unknown/change slot을 없앤 계약도 5건 생성했다. 미언급 정보를 미정으로 채우던 문제는 이 관측에서 사라졌지만, D06의 작성 지시 두 개가 사실 목록에 들어가 생산자는 여전히 미승인이다. 이 결함을 사전에 명시하고, 계약을 고치지 않은 진단용 실제 소비자 5건까지 실행했다. 네 문안의 의미와 형식, 계산 대조의 합계·단위는 보존됐으나 D03·D20의 사무 문구는 어색했다. 별도 stateless 보조 검토도 같은 두 자연스러움 partial을 판정했다. JSON만 보고 합격시키지 않았고, 최종 문안까지 확인한 뒤에도 기능을 승격하지 않았다.

[HyperCLOVA X SEED Omni 8B의 라이선스](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B/blob/main/LICENSE)도 확인했다. 조항 4의 사용자 규모·NAVER 서비스와의 경쟁 조건과 별도 라이선스 요구를 확인하지 않은 채 상용 권리를 가정하지 않았다. 이 모델의 가중치는 받지 않았고, Apache-2.0 대안의 검토를 유지했다.

## 16. 한국어 8B 후보와 운영 메모리 경계

[공식 Kanana 1.5 8B](https://huggingface.co/kakaocorp/kanana-1.5-8b-instruct-2505/tree/c963a5f4f6496c749f94064a20b33028b0db9f19)의 고정 Apache-2.0 라이선스를 읽고, [DevQuasar 고정 Q4_K_M 패키지](https://huggingface.co/DevQuasar/kakaocorp.kanana-1.5-8b-instruct-2505-GGUF/tree/4ac8eab32701b37555f225ee2c34dfa4edcdc99d)를 확보했다. `4,920,765,472` bytes, SHA-256 `f7ae0cc1de2396647dce4a82fb4943136bcc785339ee8944c6818184c983d287`로 발행자 LFS와 일치했다. 소스 config의 embedding 4096·32 layers·vocabulary 128259·BOS 128000·EOS 128009와 GGUF가 일치했고, 저장 tensor elements는 8030285824였다. chat template도 고정 공식 tokenizer와 같았다. 이는 tensor parity 검증이나 품질 승인이 아니다. 이 단계까지 새 모델 다운로드는 4회다.

이미 확인한 Kanana native EOS 문제를 피하기 위해 문자열을 지우는 후처리가 아니라 기존 `/completion` 경로를 사용했다. 기존 수정 계약의 첫 4건은 D20이 두 문장이라 실패했다. 기존 간결 계약은 D03에 추가 질문을 붙이고 두 문장을 출력했다. 기존 compact 계약도 D03의 한 문장 요구를 어겼다. D03의 부연 내용 자체를 무근거로 다시 판정한 것은 아니다. 이 12건은 실제 `chat_reply`·정상 종료·pressure 1·포트 해제를 확인했으며, 서로 다른 계약에서 잘된 문항만 합치지 않았다.

같은 실제 Astra 네 예시를 재사용한 추가 4건도 D03의 한 문장 요구를 어겼다. 예시 D01·D08·D10·D17은 평가에서 제외했으며, 나머지 전이 10건으로 승격하지 않았다. 이 구성도 정상 완료·pressure 1·포트 해제를 확인했지만 품질은 미달이다.

개발기 실측 물리 메모리는 32 GiB지만, [기존 운영 문서](../../docs/SHARING.md)의 맥미니는 **16 GB이며 다른 서비스가 함께 돈다**. 개발기에서 압력이 정상이라는 사실을 운영 여유의 증거로 대체하지 않는다. 운영 맥미니 접속·부하 시험·배포는 하지 않았다.

SUPERVISION_REPORT_READY CHAT-ROOT-C-004
