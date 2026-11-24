"""Strict local native MLX inference; no prompt repair, fusion, or remote code."""
import json
import math
import os
from pathlib import Path
import socket

PROFILE_FIELDS = {'name', 'thinking', 'temperature', 'top_p', 'top_k', 'min_p',
                  'presence_penalty', 'repetition_penalty', 'penalty_context_size',
                  'seed', 'max_new_tokens', 'context_tokens', 'prefill_step_size'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_profile(profile):
    require(isinstance(profile, dict) and set(profile) == PROFILE_FIELDS, 'Profile requires exact fields')
    require(isinstance(profile['name'], str) and bool(profile['name'].strip()), 'Invalid profile name')
    require(type(profile['thinking']) is bool, 'thinking must be boolean')
    for key in ('temperature', 'top_p', 'min_p', 'presence_penalty', 'repetition_penalty'):
        value = profile[key]
        require(type(value) in (int, float) and math.isfinite(value), f'{key} must be finite numeric')
    require(profile['temperature'] >= 0, 'temperature must be nonnegative')
    require(0 < profile['top_p'] <= 1, 'top_p must be in (0, 1]')
    require(0 <= profile['min_p'] <= 1, 'min_p must be in [0, 1]')
    require(-2 <= profile['presence_penalty'] <= 2, 'presence_penalty must be in [-2, 2]')
    require(profile['repetition_penalty'] > 0, 'repetition_penalty must be positive')
    for key in ('top_k', 'penalty_context_size', 'seed', 'max_new_tokens', 'context_tokens', 'prefill_step_size'):
        require(type(profile[key]) is int and profile[key] >= 0, f'{key} must be a nonnegative integer')
    require(profile['seed'] <= 2 ** 32 - 1, 'seed must fit uint32')
    require(0 < profile['max_new_tokens'] < profile['context_tokens'], 'Invalid generation/context budget')
    require(0 < profile['prefill_step_size'] <= profile['context_tokens'], 'Invalid prefill budget')
    return dict(profile)


def historical_profile(max_new_tokens=768, context_tokens=3072):
    return validate_profile(dict(name='historical-greedy-no-thinking', thinking=False,
                                 temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
                                 presence_penalty=0.0, repetition_penalty=1.0, penalty_context_size=0,
                                 seed=14, max_new_tokens=max_new_tokens, context_tokens=context_tokens,
                                 prefill_step_size=min(128, context_tokens)))


def native_thinking_token(tokenizer, marker):
    """Authenticate an atomic added token declared by the native thinking protocol."""
    token = tokenizer.convert_tokens_to_ids(marker)
    template = tokenizer.get_chat_template()
    declared = (type(token) is int and tokenizer.get_added_vocab().get(marker) == token
                and (token in tokenizer.all_special_ids or marker in template))
    label = 'closing' if marker == '</think>' else 'opening'
    require(declared and tokenizer.convert_ids_to_tokens(token) == marker
            and tokenizer.encode(marker, add_special_tokens=False) == [token],
            f'Tokenizer has no verified native thinking {label} token')
    return token


def render_prompt(tokenizer, messages, profile):
    profile = validate_profile(profile)
    require(isinstance(messages, list) and bool(messages), 'Nonempty messages required')
    for message in messages:
        require(isinstance(message, dict) and set(message) == {'role', 'content'}
                and message['role'] in ('system', 'user', 'assistant')
                and isinstance(message['content'], str) and bool(message['content'].strip()),
                'Invalid text chat message')
    require(messages[-1]['role'] == 'user', 'Generation must follow a user message')
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=profile['thinking'])
    require(isinstance(text, str) and bool(text), 'Empty native prompt')
    ids = tokenizer.encode(text, add_special_tokens=False)
    require(bool(ids) and all(type(token) is int and token >= 0 for token in ids), 'Invalid native prompt tokens')
    if profile['thinking']:
        template = tokenizer.get_chat_template()
        require('enable_thinking' in template and '<think>' in template,
                'Native template did not open thinking')
        native_thinking_token(tokenizer, '<think>')
        closing = native_thinking_token(tokenizer, '</think>')
        without_generation = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=True)
        require(text.startswith(without_generation) and len(text) > len(without_generation),
                'Native thinking generation prefix is unavailable')
        prefix_ids = tokenizer.encode(text[len(without_generation):], add_special_tokens=False)
        # Original Qwen3 starts with a plain assistant prefix; Qwen3.5 preopens
        # <think>. Neither may preclose reasoning while thinking is enabled.
        require(closing not in prefix_ids, 'Native template did not leave thinking open')
    require(len(ids) + profile['max_new_tokens'] <= profile['context_tokens'],
            'Prompt plus generation exceeds context; no truncation')
    return text, ids


def model_context(model):
    args = model.args
    text = getattr(args, 'text_config', None)
    limit = (text.get('max_position_embeddings') if isinstance(text, dict)
             else getattr(text if text is not None else args, 'max_position_embeddings', None))
    require(type(limit) is int and limit > 0, 'Actual model context is unavailable')
    return limit


def make_generation_processors(profile, prompt_length):
    """Native penalties over generated tokens only; zero window means all history."""
    from mlx_lm.sample_utils import make_logits_processors
    profile = validate_profile(profile)
    context = profile['penalty_context_size']
    processors = make_logits_processors(
        repetition_penalty=None if profile['repetition_penalty'] == 1 else profile['repetition_penalty'],
        repetition_context_size=context,
        presence_penalty=profile['presence_penalty'], presence_context_size=context)
    return [lambda tokens, logits, processor=processor: processor(tokens[prompt_length:], logits)
            for processor in processors]


def collect_generation(model, prompt_ids, stops, profile, check, generated_ids):
    """Collect exact native token IDs including EOS; never infer termination from text."""
    profile = validate_profile(profile)
    require(bool(prompt_ids) and all(type(token) is int and token >= 0 for token in prompt_ids),
            'Invalid prompt tokens')
    require(not generated_ids, 'Generation output must start empty')
    require(profile['context_tokens'] <= model_context(model), 'Profile exceeds actual model context')
    require(len(prompt_ids) + profile['max_new_tokens'] <= profile['context_tokens'],
            'Prompt plus generation exceeds context; no truncation')
    require(bool(stops) and all(type(token) is int and token >= 0 for token in stops), 'Invalid native stop tokens')
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    mx.random.seed(profile['seed'])
    sampler = make_sampler(temp=profile['temperature'], top_p=profile['top_p'],
                           top_k=profile['top_k'], min_p=profile['min_p'])
    iterator = generate_step(mx.array(prompt_ids, dtype=mx.int32), model,
                             max_tokens=profile['max_new_tokens'], sampler=sampler,
                             logits_processors=make_generation_processors(profile, len(prompt_ids)),
                             prompt_cache=make_prompt_cache(model), prefill_step_size=profile['prefill_step_size'],
                             prompt_progress_callback=lambda n, total: check('prefill', n=n, total=total))
    try:
        for token, _logprobs in iterator:
            generated_ids.append(int(token))
            check('token', generated_tokens=len(generated_ids))
            if token in stops:
                break
    finally:
        try:
            iterator.close()
        finally:
            mx.synchronize()
    require(bool(generated_ids), 'Generation yielded no tokens')


def decode_generation(tokenizer, generated_ids, stops, max_new_tokens, thinking=False):
    stops = {stops} if type(stops) is int else set(stops or [])
    require(type(max_new_tokens) is int and max_new_tokens > 0 and type(thinking) is bool,
            'Invalid decoding budget or thinking mode')
    require(0 < len(generated_ids) <= max_new_tokens
            and all(type(token) is int and token >= 0 for token in generated_ids),
            'Generated IDs violate registered envelope')
    terminal = generated_ids[-1] if generated_ids[-1] in stops else None
    require(not any(token in stops for token in generated_ids[:-1]), 'Stop token before final generated token')
    require(terminal is not None or len(generated_ids) == max_new_tokens, 'Unexplained generation termination')
    kwargs = dict(skip_special_tokens=False, clean_up_tokenization_spaces=False)
    answer_ids = generated_ids[:-1] if terminal is not None else generated_ids
    complete = not thinking
    if thinking:
        closing = native_thinking_token(tokenizer, '</think>')
        if closing in answer_ids:
            answer_ids = answer_ids[answer_ids.index(closing) + 1:]
            complete = True
        else:
            answer_ids = []
    return dict(raw_output=tokenizer.decode(generated_ids, **kwargs),
                response_text=tokenizer.decode(answer_ids, **kwargs),
                generated_tokens=len(generated_ids), terminal_stop_token_id=terminal,
                termination_reason='eos' if terminal is not None else 'length',
                framing_status='complete' if complete else 'incomplete_reasoning',
                reasoning_complete=complete)


def configure_offline():
    for key, value in {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'USE_TORCH': '0',
                       'USE_TF': '0', 'TOKENIZERS_PARALLELISM': 'false'}.items():
        os.environ[key] = value
    def denied(*args, **kwargs):
        raise RuntimeError('Network access forbidden in offline inference')
    for name in ('create_connection', 'getaddrinfo', 'gethostbyname', 'gethostbyname_ex', 'gethostbyaddr'):
        setattr(socket, name, denied)
    for name in ('connect', 'connect_ex', 'sendto', 'sendmsg'):
        if hasattr(socket.socket, name):
            setattr(socket.socket, name, denied)


def local_model_config(model_path):
    root = Path(model_path)
    require(root.is_absolute() and root.is_dir(), 'Existing absolute local model directory required')
    for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        require((root / name).is_file(), f'Missing local model file: {name}')
    require(any(root.glob('model*.safetensors')), 'Missing local model weights')
    config = json.loads((root / 'config.json').read_text())
    require(config.get('model_type') in ('qwen3', 'qwen3_5')
            and not config.get('model_file') and not config.get('auto_map'), 'Unsupported native model architecture')
    text = config.get('text_config', config)
    require(isinstance(text, dict) and not text.get('model_file') and not text.get('auto_map'),
            'Remote text architecture forbidden')
    if config['model_type'] == 'qwen3_5':
        require(text.get('model_type') == 'qwen3_5_text', 'Unsupported Qwen3.5 text architecture')
    require(type(text.get('max_position_embeddings')) is int and text['max_position_embeddings'] > 0,
            'Actual model context is unavailable')
    tokenizer_config = json.loads((root / 'tokenizer_config.json').read_text())
    require(not tokenizer_config.get('auto_map'), 'Remote tokenizer code forbidden')
    return config


def validate_adapter_tensors(model, weights):
    from mlx.utils import tree_flatten
    expected = dict(tree_flatten(model.trainable_parameters()))
    require(bool(expected) and isinstance(weights, dict) and set(weights) == set(expected),
            'Adapter tensor keys require exact coverage')
    require(all(tuple(weights[key].shape) == tuple(value.shape) for key, value in expected.items()),
            'Adapter tensor shape mismatch')


def load_local(model_path, adapter_path=None):
    local_model_config(model_path)
    adapter = None
    if adapter_path is not None:
        adapter = Path(adapter_path)
        require(adapter.is_absolute() and adapter.is_dir(), 'Existing absolute local adapter directory required')
        for name in ('adapter_config.json', 'adapters.safetensors'):
            require((adapter / name).is_file(), f'Missing local adapter file: {name}')
    configure_offline()
    from mlx_lm.utils import load
    model, tokenizer, config = load(str(model_path), tokenizer_config={'local_files_only': True,
                                    'trust_remote_code': False}, return_config=True, trust_remote_code=False)
    if adapter is not None:
        import mlx.core as mx
        from mlx_lm.tuner.utils import linear_to_lora_layers
        declaration = json.loads((adapter / 'adapter_config.json').read_text())
        require(declaration.get('fine_tune_type') == 'lora', 'Only explicit native LoRA adapters supported')
        require(type(declaration.get('num_layers')) is int and 0 < declaration['num_layers'] <= len(model.layers),
                'Invalid adapter layer count')
        model.freeze()
        linear_to_lora_layers(model, declaration['num_layers'], declaration['lora_parameters'])
        weights = mx.load(str(adapter / 'adapters.safetensors'))
        validate_adapter_tensors(model, weights)
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())
    model.eval()
    return model, tokenizer, config
