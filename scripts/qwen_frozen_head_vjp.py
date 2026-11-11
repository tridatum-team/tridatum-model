"""Experimental CPU FP32 frozen-Q8 answer head; not routed into production training.

Each token's CE is independent. Compute its unit hidden derivative while the bounded
logits workspace is live, then retain only that hidden-width derivative for backward.
This trades checkpoint reprojection for an early Q8 backward; it is first-order only.
The original CE and Q8Matmul implement all numerical work, including row dequantisation.
"""
import torch

from scripts.train_qwen_preferences import T
from scripts import q8_gguf_base as q8
import q8_gguf_base as command_q8


def _validate(hidden, head):
    # The trainer's script import and the tests' package import create distinct classes.
    # Accept those exact implementations, not subclasses or PEFT-wrapped/trainable heads.
    if type(head) not in (q8.Q8Linear, command_q8.Q8Linear):
        raise TypeError('frozen head VJP requires an exact Q8Linear head')
    if head.bias is not None or any(p.requires_grad for p in head.parameters()):
        raise ValueError('frozen head VJP requires a frozen, bias-free head')
    if hidden.device.type != 'cpu' or head.weight.device.type != 'cpu' or head.scales.device.type != 'cpu':
        raise ValueError('frozen head VJP supports CPU only')
    if hidden.dtype != torch.float32 or head.compute_dtype != torch.float32:
        raise ValueError('frozen head VJP requires FP32 hidden states and compute dtype')
    if head.weight.dtype != torch.int8 or head.scales.dtype != torch.float16:
        raise ValueError('frozen head VJP requires int8 weights and FP16 scales')
    if torch.is_autocast_enabled('cpu'):
        raise ValueError('frozen head VJP does not support autocast')
    if (hidden.ndim != 3 or hidden.shape[-1] != head.in_features
            or tuple(head.weight.shape) != (head.out_features, head.in_features)
            or tuple(head.scales.shape) != (head.out_features, head.in_features // q8.Q8_BLOCK)):
        raise ValueError('frozen head VJP requires compatible [batch, tokens, hidden] Q8 shapes')
    if (type(head.chunk_rows) is not int or head.chunk_rows < 1
            or head.chunk_rows > q8.chunk_rows(head.in_features, q8.Q8_DEQUANT_CHUNK_BYTES)):
        raise ValueError('frozen head VJP requires the original bounded Q8 dequantisation workspace')


class _FrozenHeadCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, head, target):
        with torch.enable_grad():
            local_hidden = hidden.detach().requires_grad_(True)
            losses = T.chunk_cross_entropy(local_hidden, head, target)
            derivative, = torch.autograd.grad(losses.sum(), local_hidden)
        ctx.save_for_backward(derivative)
        return losses.detach()

    @staticmethod
    def backward(ctx, grad_losses):
        if torch.is_grad_enabled():
            raise RuntimeError('frozen head VJP does not support higher-order derivatives (create_graph=True)')
        derivative, = ctx.saved_tensors
        return derivative * grad_losses.unsqueeze(-1), None, None


def frozen_head_cross_entropy(hidden, head, target):
    """Original per-token CE with first-order VJP storage instead of head checkpointing.

    Hidden states are CPU FP32 [batch, tokens, hidden]; targets follow the original
    CE's [batch, tokens] / -100 ignore convention. No-grad scoring computes no VJP.
    Frozen heads are required even for no-grad or all-ignored calls.
    """
    _validate(hidden, head)
    if not torch.is_grad_enabled() or not hidden.requires_grad:
        return T.chunk_cross_entropy(hidden, head, target)
    return _FrozenHeadCrossEntropy.apply(hidden, head, target)


def chunked_answer_loss(model, ids, mask, keep, target):
    """Original supervised packing, sequence ownership and LOGIT_CHUNK capacity."""
    inner = T.causal_lm(model)
    hidden = inner.model(input_ids=ids, attention_mask=mask).last_hidden_state
    head = inner.lm_head
    _validate(hidden, head)
    rows, columns = (target != -100).nonzero(as_tuple=True)
    if rows.numel() == 0:
        # Keep the original head/CE calculation and connected exact zero gradients.
        selected = hidden[:, keep]
        return torch.cat([frozen_head_cross_entropy(selected[:, s:s + T.LOGIT_CHUNK], head,
                                                    target[:, s:s + T.LOGIT_CHUNK])
                          for s in range(0, selected.shape[1], T.LOGIT_CHUNK)], dim=1)
    packed_hidden = hidden[rows, keep[columns]].unsqueeze(0)
    packed_target = target[rows, columns].unsqueeze(0)
    capacity = target.shape[0] * T.LOGIT_CHUNK
    losses = torch.cat([frozen_head_cross_entropy(packed_hidden[:, s:s + capacity], head,
                                                packed_target[:, s:s + capacity])
                        for s in range(0, rows.numel(), capacity)], dim=1)
    return losses.new_zeros(target.shape).index_put((rows, columns), losses.reshape(-1))
