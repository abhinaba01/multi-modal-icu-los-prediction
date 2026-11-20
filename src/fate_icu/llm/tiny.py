"""A tiny random chat model and tokenizer, built offline, for tests and smoke runs (Stage 6).

It proves that the training and generation code runs end to end (label masking, LoRA module selection,
resume, the pass-rate callback) without downloading anything. Its weights are random, so it learns only
to imitate the training text: it can show that the loss goes down, never that explanations are good.

The tokenizer is byte-level BPE trained on the given texts, so decoding is exact, with a Gemma-style chat
template (``<start_of_turn>user ... <end_of_turn>``; the assistant role is called ``model``).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ '<start_of_turn>' + ('model' if m['role'] == 'assistant' else 'user') + '\\n' + m['content'] | trim"
    " + '<end_of_turn>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}"
)
SPECIALS = ["<pad>", "<eos>", "<bos>", "<start_of_turn>", "<end_of_turn>"]


def make_tiny_tokenizer(texts: Iterable[str], vocab_size: int = 400):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(list(texts), trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
        additional_special_tokens=["<start_of_turn>", "<end_of_turn>"],
    )
    fast.chat_template = CHAT_TEMPLATE
    return fast


def make_tiny_model(path: str | Path, texts: Iterable[str], *, vocab_size: int = 400, seed: int = 0) -> Path:
    """Write a tiny random Llama-style model and its tokenizer to ``path`` and return ``path``."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tokenizer = make_tiny_tokenizer(texts, vocab_size)
    tokenizer.save_pretrained(path)
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<end_of_turn>"),
        bos_token_id=tokenizer.bos_token_id,
    )
    LlamaForCausalLM(config).save_pretrained(path)
    return path
