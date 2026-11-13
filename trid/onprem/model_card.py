"""Model information shown in the on-prem UI (spec section 6).

status, base_model, license and evaluation are copied verbatim from the
production entry trid-assistant-qwen3-1.7b of deploy/macmini-site/versions.json.
"""

MODEL_CARD = {
    'label': (
        'TRID · Qwen3-1.7B 원본 (업무 특화 학습 없음)'
    ),
    'version': (
        '설치판 · Qwen3-1.7B 원본 가중치 그대로(튜닝 없음) · 올린 PDF에서만 근거 검색 · 질문별 독립 처리 · llama.cpp 8ca4060 리눅스 CPU 빌드 · greedy · 사고 모드 끔'
    ),
    'status': (
        '실험 서비스 · 원본 모델(미특화) · 품질 기준 미달'
    ),
    'base_model': (
        'Qwen/Qwen3-1.7B-GGUF Q8_0, revision 90862c4b9d2787eaed51d12237eafdfe7c5f6077 (Alibaba Qwen)'
    ),
    'license': (
        'Apache-2.0'
    ),
    'evaluation': (
        '이 서비스 조건 그대로의 측정은 아직 없음. 참고(조건이 다름): 같은 원본 파일의 RAG310(값 형식 지시·고정 근거) 답 있는 228문항 중 104, 보류해야 할 82문항 중 바르게 보류 2. C-023 pilot_dev(설명형 지시, 같은 가중치의 다른 변환 파일) 결론 정확 9/18.'
    ),
    'limits': (
        "올린 PDF에서 찾은 원문과 글자 그대로 일치하는 문장만 '확인된 인용'이며 해설은 검증하지 않았습니다. 관련도 상위 구간만 확인하므로 문서의 다른 부분에 정보가 없다고 보장하지 않습니다. 근거를 찾지 못한 질문에 기억으로 답하지 않습니다. 이미지·스캔 PDF는 읽지 못합니다."
    ),
}
