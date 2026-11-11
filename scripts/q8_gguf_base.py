"""Frozen ggml Q8_0 base weights for LoRA training, loaded tensor-by-tensor from the serving GGUF.

Representation (explicit, NOT the BF16 base): every 2-D weight keeps the GGUF's Q8_0 blocks as two frozen
Parameters, `weight` int8 [out, in] and `scales` float16 [out, in/32] (block size 32, one fp16 scale per block,
exactly the bytes ggml serves); value = int8 * fp16, computed in float32 and rounded to `compute_dtype` (bfloat16 on
MPS, float32 on CPU, where the rounding is exact). 1-D norms are F32 in the GGUF and are cast to `compute_dtype`.
The head shares the embedding's `weight` AND `scales` Parameter objects (tied, as the HF base).

Q8Linear / Q8Embedding subclass nn.Linear / nn.Embedding without ever allocating a dense float weight; PEFT's
default LoRA dispatcher wraps Q8Linear as an ordinary Linear (int8 base weight: adapters stay float32 on the base's
device). Q8Matmul dequantises bounded output-row chunks on the fly in forward and in backward; autograd saves only
the quantised references, never a dequantised weight. The input gradient is accumulated over row chunks in float32
and cast back, so it is not bit-identical to one full-width matmul in `compute_dtype`. The embedding dequantises
only the gathered rows. Merging LoRA into an int8 base is unsupported.

Loading: the GGUF header (names, types, ggml shapes, offsets) is validated against the meta-built Qwen3ForCausalLM
and the expected file sha256 BEFORE any tensor bytes are read: architecture, file type, block count, every expected
tensor present with the right shape/type (Q8_0 for 2-D, F32 for 1-D, in % 32 == 0), nothing unexpected; a present
`output.weight` (tied head) is shape-checked only and never read. Data is read with os.pread per bounded row chunk
into a CPU numpy buffer, split into int8/fp16 and copied into the PyTorch-allocated device tensors (the MPS memory
fraction cap applies); no mmap-backed weight storage is retained (GGUFReader's header memmap is dropped after the
header pass). Qwen3 needs no Q/K permutation (llama.cpp's Qwen3 converter only renames). No memory fit is claimed.
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

Q8_BLOCK = 32                      # weights per Q8_0 block
Q8_BLOCK_BYTES = 34                # 2-byte fp16 scale + 32 int8
Q8_DEQUANT_CHUNK_BYTES = 16 << 20  # float32 row workspace; dtype conversion additionally needs a compute-dtype copy
Q8_LOAD_CHUNK_BYTES = 16 << 20     # pread CPU staging bound per row chunk
REPRESENTATION = 'q8_0'


def sha256_file(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dequantize_rows(weight, scales, compute_dtype):
    """Float rows of a Q8_0 slice: int8 [r, in] * fp16 [r, in/32] in float32, rounded to compute_dtype."""
    rows, in_features = weight.shape
    blocks = weight.view(rows, in_features // Q8_BLOCK, Q8_BLOCK).to(torch.float32)
    blocks.mul_(scales.to(torch.float32).unsqueeze(-1))
    return blocks.view(rows, in_features).to(compute_dtype)


def chunk_rows(in_features, chunk_bytes):
    """Output rows per dequantised chunk so that the float32 transient stays under chunk_bytes."""
    return max(1, chunk_bytes // (in_features * 4))


class Q8Matmul(torch.autograd.Function):
    """y = x @ dequant(weight, scales).T, rows dequantised per chunk; backward re-dequantises for grad_x only."""

    @staticmethod
    def forward(ctx, x, weight, scales, compute_dtype, rows):
        ctx.save_for_backward(weight, scales)
        ctx.compute_dtype, ctx.rows, ctx.x_dtype = compute_dtype, rows, x.dtype
        out_features = weight.shape[0]
        x = x.to(compute_dtype)
        out = torch.empty(*x.shape[:-1], out_features, dtype=compute_dtype, device=x.device)
        for r0 in range(0, out_features, rows):
            r1 = min(r0 + rows, out_features)
            out[..., r0:r1] = torch.matmul(x, dequantize_rows(weight[r0:r1], scales[r0:r1], compute_dtype).t())
        return out

    @staticmethod
    def backward(ctx, grad_out):
        weight, scales = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        out_features, in_features = weight.shape
        grad_out = grad_out.to(ctx.compute_dtype)
        grad_x = torch.zeros(*grad_out.shape[:-1], in_features, dtype=torch.float32, device=grad_out.device)
        for r0 in range(0, out_features, ctx.rows):
            r1 = min(r0 + ctx.rows, out_features)
            grad_x += torch.matmul(grad_out[..., r0:r1], dequantize_rows(weight[r0:r1], scales[r0:r1], ctx.compute_dtype))
        return grad_x.to(ctx.x_dtype), None, None, None, None


class Q8Linear(nn.Linear):
    """nn.Linear whose weight is frozen Q8_0 (int8 weight + fp16 block scales); no dense float weight ever exists."""

    def __init__(self, in_features, out_features, compute_dtype, device):
        if in_features % Q8_BLOCK:
            raise ValueError(f'in_features {in_features} is not a multiple of the Q8_0 block size {Q8_BLOCK}')
        nn.Module.__init__(self)
        self.in_features, self.out_features, self.compute_dtype = in_features, out_features, compute_dtype
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.int8, device=device), requires_grad=False)
        self.scales = nn.Parameter(torch.empty((out_features, in_features // Q8_BLOCK), dtype=torch.float16, device=device), requires_grad=False)
        self.register_parameter('bias', None)
        self.chunk_rows = chunk_rows(in_features, Q8_DEQUANT_CHUNK_BYTES)

    def reset_parameters(self):
        raise RuntimeError('Q8Linear weights come from a GGUF; there is no initialisation')

    def forward(self, x):
        return Q8Matmul.apply(x, self.weight, self.scales, self.compute_dtype, self.chunk_rows)

    def dequantize(self):
        """Full dequantised weight [out, in] in compute_dtype (reference/test use only; never called in training)."""
        return dequantize_rows(self.weight, self.scales, self.compute_dtype)


class Q8Embedding(nn.Embedding):
    """nn.Embedding over frozen Q8_0 rows; only the gathered rows are dequantised."""

    def __init__(self, num_embeddings, embedding_dim, compute_dtype, device, padding_idx=None):
        if embedding_dim % Q8_BLOCK:
            raise ValueError(f'embedding_dim {embedding_dim} is not a multiple of the Q8_0 block size {Q8_BLOCK}')
        nn.Module.__init__(self)
        self.num_embeddings, self.embedding_dim, self.compute_dtype = num_embeddings, embedding_dim, compute_dtype
        self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse = padding_idx, None, 2.0, False, False
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), dtype=torch.int8, device=device), requires_grad=False)
        self.scales = nn.Parameter(torch.empty((num_embeddings, embedding_dim // Q8_BLOCK), dtype=torch.float16, device=device), requires_grad=False)

    def reset_parameters(self):
        raise RuntimeError('Q8Embedding weights come from a GGUF; there is no initialisation')

    def forward(self, ids):
        rows = dequantize_rows(self.weight[ids.reshape(-1)], self.scales[ids.reshape(-1)], self.compute_dtype)
        return rows.view(*ids.shape, self.embedding_dim)

    def dequantize(self):
        return dequantize_rows(self.weight, self.scales, self.compute_dtype)


def tied_q8_head(embedding):
    """Q8Linear head sharing the embedding's weight AND scales Parameter objects (tie_weights() keeps the weight tie;
    the scales tie is ours and is asserted after loading). Built on meta: no storage of its own."""
    head = Q8Linear(embedding.embedding_dim, embedding.num_embeddings, embedding.compute_dtype, 'meta')
    head.weight, head.scales = embedding.weight, embedding.scales
    return head


def _field(reader, key):
    field = reader.get_field(key)
    return None if field is None else field.contents()


def _read_q8_rows(fd, module, data_offset, in_features, out_features, report):
    """Fill module.weight/module.scales from Q8_0 blocks, one bounded row chunk per pread(2)."""
    n_blocks = in_features // Q8_BLOCK
    row_bytes = n_blocks * Q8_BLOCK_BYTES
    rows = max(1, Q8_LOAD_CHUNK_BYTES // row_bytes)
    for r0 in range(0, out_features, rows):
        r1 = min(r0 + rows, out_features)
        n = (r1 - r0) * row_bytes
        buf = os.pread(fd, n, data_offset + r0 * row_bytes)
        if len(buf) != n:
            raise IOError(f'short read: wanted {n} bytes at {data_offset + r0 * row_bytes}, got {len(buf)}')
        blocks = np.frombuffer(buf, dtype=np.uint8).reshape(r1 - r0, n_blocks, Q8_BLOCK_BYTES)
        scales = np.ascontiguousarray(blocks[:, :, :2]).view(np.float16).reshape(r1 - r0, n_blocks)
        qs = np.ascontiguousarray(blocks[:, :, 2:]).view(np.int8).reshape(r1 - r0, in_features)
        module.weight.data[r0:r1].copy_(torch.from_numpy(qs))      # CPU chunk -> PyTorch-allocated device rows
        module.scales.data[r0:r1].copy_(torch.from_numpy(scales))
        del buf, blocks, scales, qs
        report['bytes_loaded'] += n
        report['chunks'] += 1


@torch.no_grad()
def load_qwen3_q8_gguf(model_dir, gguf_path, expected_sha256, compute_dtype, device):
    """Qwen3ForCausalLM whose 2-D weights are the GGUF's frozen Q8_0 blocks (Q8Linear/Q8Embedding, tied Q8 head)
    and whose norms are the GGUF's F32 tensors cast to compute_dtype. See the module docstring. Returns (model, report)."""
    import gguf
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

    model_dir, gguf_path = Path(model_dir), Path(gguf_path)
    actual = sha256_file(gguf_path)
    if actual != expected_sha256:
        raise ValueError(f'{gguf_path}: sha256 {actual} != expected {expected_sha256}')
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if config.model_type != 'qwen3':
        raise ValueError(f'load_qwen3_q8_gguf supports model_type qwen3 only, got {config.model_type!r}')
    config.name_or_path = model_dir  # PEFT records it as base_model_name_or_path (adapter conversion contract)
    config.dtype = compute_dtype
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config, dtype=compute_dtype)
    model.tie_weights()
    embed, head = model.get_input_embeddings(), model.get_output_embeddings()
    if head is None or head.weight is not embed.weight:
        raise ValueError('load_qwen3_q8_gguf expects a tied output embedding (tie_word_embeddings)')
    head_key = next(n for n, m in model.named_modules() if m is head) + '.weight'
    embed_key = next(n for n, m in model.named_modules() if m is embed) + '.weight'
    expected = {name: tuple(t.shape) for name, t in model.state_dict().items() if name != head_key}

    reader = gguf.GGUFReader(str(gguf_path))
    header = dict(architecture=_field(reader, 'general.architecture'), file_type=_field(reader, 'general.file_type'),
                  block_count=_field(reader, f'{config.model_type}.block_count'))
    if reader.byte_order != 'I':
        raise ValueError(f'{gguf_path}: byte-swapped GGUF is not supported')
    tensors = {t.name: (t.tensor_type, tuple(reversed(t.shape.tolist())), int(t.data_offset), int(t.n_bytes)) for t in reader.tensors}
    del reader  # drop the header memmap; data is read with pread only
    problems = []
    if header['architecture'] != config.model_type:
        problems.append(f"general.architecture {header['architecture']!r} != {config.model_type!r}")
    if header['file_type'] != int(gguf.LlamaFileType.MOSTLY_Q8_0):
        problems.append(f"general.file_type {header['file_type']!r} != MOSTLY_Q8_0 ({int(gguf.LlamaFileType.MOSTLY_Q8_0)})")
    if header['block_count'] != config.num_hidden_layers:
        problems.append(f"block_count {header['block_count']!r} != config.num_hidden_layers {config.num_hidden_layers}")
    name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN3, config.num_hidden_layers)
    plan, used = [], set()
    for key, shape in expected.items():
        gguf_name = name_map.get_name(key, try_suffixes=('.weight',))
        if gguf_name is None:
            problems.append(f'{key}: no GGUF name mapping')
            continue
        entry = tensors.get(gguf_name)
        if entry is None:
            problems.append(f'{key}: missing GGUF tensor {gguf_name}')
            continue
        kind, got, offset, nbytes = entry
        used.add(gguf_name)
        if got != shape:
            problems.append(f'{gguf_name}: GGUF shape {list(got)} != model shape {list(shape)}')
            continue
        if len(shape) == 2:
            if kind != gguf.GGMLQuantizationType.Q8_0:
                problems.append(f'{gguf_name}: type {kind.name}, expected Q8_0')
            elif shape[1] % Q8_BLOCK:
                problems.append(f'{gguf_name}: in_features {shape[1]} not a multiple of {Q8_BLOCK}')
        elif kind != gguf.GGMLQuantizationType.F32:
            problems.append(f'{gguf_name}: type {kind.name}, expected F32')
        plan.append((key, gguf_name, kind, shape, offset, nbytes))
    tied_name = name_map.get_name(head_key, try_suffixes=('.weight',))
    skipped_tied, bytes_skipped = [], 0
    if tied_name in tensors:  # bytes tie_weights() would discard: shape-checked, never read
        kind, got, _, nbytes = tensors[tied_name]
        used.add(tied_name)
        if got != expected[embed_key]:
            problems.append(f'{tied_name}: GGUF shape {list(got)} != model shape {list(expected[embed_key])}')
        skipped_tied.append(tied_name)
        bytes_skipped += nbytes
    unexpected = sorted(set(tensors) - used)
    if unexpected:
        problems.append(f'unexpected GGUF tensors: {unexpected}')
    if problems:
        raise ValueError(f'{gguf_path} does not match {type(model).__name__} from {model_dir}: ' + '; '.join(problems))

    report = dict(backend='gguf-pread', staging='cpu-row-chunk', representation=REPRESENTATION, device=str(device),
                  compute_dtype=str(compute_dtype), block_size=Q8_BLOCK, scale_dtype='float16',
                  gguf=dict(path=str(gguf_path), sha256=actual, **header), tensors=dict(q8_0=0, f32=0),
                  skipped_tied=skipped_tied, input_key=embed_key, parameters=0, bytes_loaded=0, bytes_skipped=bytes_skipped,
                  chunks=0, load_chunk_bytes=Q8_LOAD_CHUNK_BYTES, dequant_chunk_bytes=Q8_DEQUANT_CHUNK_BYTES,
                  note='training base = serving GGUF bytes; PyTorch dequant matmul in compute_dtype, not the ggml Q8 kernel')
    fd = os.open(gguf_path, os.O_RDONLY)
    try:
        for key, gguf_name, kind, shape, offset, nbytes in plan:
            module_name, _, attr = key.rpartition('.')
            parent_name, _, child = module_name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            old = getattr(parent, child)
            if kind == gguf.GGMLQuantizationType.Q8_0:
                if isinstance(old, nn.Embedding):
                    new = Q8Embedding(shape[0], shape[1], compute_dtype, device, padding_idx=old.padding_idx)
                elif isinstance(old, nn.Linear) and old.bias is None and attr == 'weight':
                    new = Q8Linear(shape[1], shape[0], compute_dtype, device)
                else:
                    raise ValueError(f'{key}: Q8_0 tensor for unsupported module {type(old).__name__}')
                setattr(parent, child, new)
                _read_q8_rows(fd, new, offset, shape[1], shape[0], report)
                report['tensors']['q8_0'] += 1
            else:
                buf = os.pread(fd, nbytes, offset)
                if len(buf) != nbytes:
                    raise IOError(f'short read for {gguf_name}')
                staged = torch.frombuffer(bytearray(buf), dtype=torch.float32).reshape(shape)
                tensor = staged.to(device=device, dtype=compute_dtype)  # PyTorch-allocated device copy
                del staged, buf
                old.load_state_dict({attr: tensor}, strict=False, assign=True)
                del tensor
                report['tensors']['f32'] += 1
                report['bytes_loaded'] += nbytes
            report['parameters'] += 1
    finally:
        os.close(fd)
    embed = model.get_input_embeddings()
    head_parent_name, _, head_child = head_key[:-len('.weight')].rpartition('.')
    setattr(model.get_submodule(head_parent_name) if head_parent_name else model, head_child, tied_q8_head(embed))
    model.tie_weights()
    head = model.get_output_embeddings()
    still_meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type == 'meta']
    if still_meta:
        raise RuntimeError(f'parameters left uninitialised after loading: {still_meta}')
    if head.weight is not embed.weight or head.scales is not embed.scales:
        raise RuntimeError('tied Q8 head lost its shared weight/scales storage')
    model.to(device)  # int8/fp16 parameters are already on `device`; this moves the CPU-initialised buffers
    model.requires_grad_(False)  # Freeze the F32-derived norms too; only subsequently attached LoRA parameters train.
    model.eval()
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_dir, local_files_only=True)
    except OSError:
        pass
    return model, report


def base_weights_record(report):
    """training.json / checkpoints.json `base_weights` block distinguishing this run from BF16 base runs."""
    return dict(representation=REPRESENTATION, gguf=report['gguf']['path'], gguf_sha256=report['gguf']['sha256'],
                block_size=report['block_size'], scale_dtype=report['scale_dtype'], compute_dtype=report['compute_dtype'],
                quantized_tensors=report['tensors']['q8_0'], f32_tensors=report['tensors']['f32'],
                merge_supported=False, note=report['note'])
