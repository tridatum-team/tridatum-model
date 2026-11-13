"""Client for a local llama.cpp `llama-server`: native chat template, greedy decoding, thinking disabled."""
import hashlib
import json
import urllib.request

QWEN3_NO_THINK = '<think>\n\n</think>\n\n'


def request(base, endpoint, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(base+endpoint, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def complete(base, messages, family, max_tokens, context, grammar=None):
    """Render with the model's native template, then greedy-complete; returns result fields."""
    prompt = request(base, '/apply-template', {'messages': messages, 'chat_template_kwargs': {'enable_thinking': False}})['prompt']
    result = dict(rendered_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
    # Qwen3's native no-thinking generation prefix must actually be rendered.
    if family == 'qwen3' and not prompt.endswith(QWEN3_NO_THINK):
        raise ValueError('Native Qwen3 no-thinking prefix was not rendered')
    tokens = request(base, '/tokenize', {'content': prompt, 'add_special': True, 'parse_special': True})['tokens']
    result['prompt_tokens'] = len(tokens)
    if len(tokens) + max_tokens > context:
        result['skipped_context'] = True
        return result
    payload = dict(prompt=tokens, n_predict=max_tokens, temperature=0, seed=42, samplers=['temperature'],
                   repeat_penalty=1, cache_prompt=False, return_tokens=True)
    if grammar is not None:
        payload['grammar'] = grammar
        result['grammar_sha256'] = hashlib.sha256(grammar.encode()).hexdigest()
    completion = request(base, '/completion', payload)
    result.update(generated=completion['content'].strip(),
        output_tokens=len(completion['tokens']), finish_reason=completion['stop_type'],
        truncated_output=completion['stop_type']=='limit',
        context_truncated=completion['truncated'], timings=completion['timings'])
    return result


def count_prompt_tokens(base, messages, timeout=180):
    """Tokens of the prompt as the serving model renders and tokenizes it (same /apply-template + /tokenize calls as
    `complete`), or None when the server cannot be asked."""
    try:
        prompt = request(base, '/apply-template', {'messages': messages, 'chat_template_kwargs': {'enable_thinking': False}}, timeout=timeout)['prompt']
        return len(request(base, '/tokenize', {'content': prompt, 'add_special': True, 'parse_special': True}, timeout=timeout)['tokens'])
    except OSError:
        return None
