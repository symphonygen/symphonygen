import warnings
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from transformers import Cache, DynamicCache, EncoderDecoderCache, GPT2Config, GPT2Model, GPT2PreTrainedModel, __version__
from transformers.masking_utils import create_causal_mask
from arch.model.cross_attn_provider import CrossAttnProvider, prepare_masks
from arch.vocab import *

class EventDecoder(GPT2Model):
    def __init__(self, config: GPT2Config):
        super().__init__(config)

        if __version__ not in ("5.0.0", "5.1.0"):
            warnings.warn(f"{__version__=} is not tested")

    def forward(self,
        inputs_embeds: torch.FloatTensor,
        cross_attn_provider: CrossAttnProvider | None = None,
        attention_mask: torch.BoolTensor | None = None,
        output_hidden_states: bool = False,

        # KV cache
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        use_cache: bool = False,
    ):
        if use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds + self.wpe(position_ids)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_ids=position_ids
        )

        if self.config.add_cross_attention:
            prepare_masks(cross_attn_provider, self._attn_implementation, inputs_embeds.dtype, inputs_embeds.shape[1])

        hidden_states = self.drop(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        for layer_idx, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.config.add_cross_attention:
                encoder_hidden_states, encoder_attention_mask = cross_attn_provider.forward(hidden_states, layer_idx)
            else:
                encoder_hidden_states, encoder_attention_mask = None, None

            outputs = block(
                hidden_states,
                past_key_values=past_key_values,
                cache_position=cache_position,
                attention_mask=causal_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
            )
            hidden_states = outputs[0]

        hidden_states = self.ln_f(hidden_states)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, past_key_values, all_hidden_states

class EventDecoderWithLMHead(GPT2PreTrainedModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.transformer = EventDecoder(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # XXX: Different from GPT2LMHeadModel, we do not tie the weights here
        # self.lm_head.weight = self.transformer.wte.weight

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs_embeds,
        cross_attn_provider=None,
        attention_mask=None,
        labels: torch.LongTensor = None,
        output_hidden_states=False,
        **kv_cache_kargs,
    ):
        hidden_states, past_key_values, all_hidden_states = self.transformer.forward(
            inputs_embeds,
            cross_attn_provider=cross_attn_provider,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            **kv_cache_kargs,
        )

        logits = self.lm_head.forward(hidden_states)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.flatten(0, -2), labels.flatten(), ignore_index=self.config.pad_token_id)
            loss.nan_to_num_() # May happen when data is empty

        return EventDecoderOutput(
            logits=logits,
            loss=loss,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )

@dataclass
class EventDecoderOutput:
    logits: torch.FloatTensor | None = None
    loss: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: list[torch.FloatTensor] | None = None
