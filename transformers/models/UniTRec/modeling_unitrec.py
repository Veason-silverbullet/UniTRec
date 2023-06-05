import os
from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.functional import log_softmax
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from .configuration_unitrec import UniTRecConfig


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class UniTRecLearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor):
        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(0, seq_len, dtype=torch.long, device=self.weight.device).expand(bsz, -1)
        return super().forward(positions + self.offset)


class UniTRecAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads}).')
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f'Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}')

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(f'Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}')
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}')

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class UniTRecEncoderLayer(nn.Module):
    def __init__(self, config: UniTRecConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = UniTRecAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)
        return outputs


class UniTRecDecoderLayer(nn.Module):
    def __init__(self, config: UniTRecConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = UniTRecAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = UniTRecAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        if encoder_hidden_states is not None:
            residual = hidden_states
            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            hidden_states = self.encoder_attn(hidden_states=hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask)
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states,)
        return outputs


class UniTRecPretrainedModel(PreTrainedModel):
    config_class = UniTRecConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = [r'encoder.version', r'decoder.version']
    _no_split_modules = [r'UniTRecEncoderLayer', r'UniTRecDecoderLayer']

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (UniTRecDecoder, UniTRecEncoder)):
            module.gradient_checkpointing = value


class UniTRecPretrainedModel(PreTrainedModel):
    config_class = UniTRecConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = [r'encoder.version', r'decoder.version']
    _no_split_modules = [r'UniTRecEncoderLayer', r'UniTRecDecoderLayer']

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (UniTRecDecoder, UniTRecEncoder)):
            module.gradient_checkpointing = value


class UniTRecEncoder(UniTRecPretrainedModel):
    def __init__(self, config: UniTRecConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight
        self.local_embed_positions = nn.Embedding(config.max_position_embeddings, embed_dim)
        self.embed_positions = UniTRecLearnedPositionalEmbedding(config.max_position_embeddings, embed_dim)
        self.layers = nn.ModuleList([UniTRecEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        self.gradient_checkpointing = False
        self.post_init()
        self.encoder_local_attention_layers = config.encoder_local_attention_layers

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids, local_position_ids, global_position_ids, local_attention_mask, global_attention_mask):
        inputs_embeds = self.embed_tokens(input_ids)
        local_embed_pos = self.local_embed_positions(local_position_ids)
        global_embed_pos = self.embed_positions(global_position_ids)
        hidden_states = inputs_embeds + local_embed_pos + global_embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        dtype = inputs_embeds.dtype
        attention_mask = torch.zeros_like(local_attention_mask, dtype=dtype, device=local_attention_mask.device)
        attention_mask.masked_fill_(~local_attention_mask, torch.finfo(dtype).min)
        global_attention_mask = _expand_mask(global_attention_mask, dtype)

        for idx, encoder_layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                if idx < self.encoder_local_attention_layers:
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask
                    )
                else:
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        global_attention_mask
                    )
            else:
                if idx < self.encoder_local_attention_layers:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        global_attention_mask
                    )
            hidden_states = layer_outputs[0]

        return hidden_states


class UniTRecDecoder(UniTRecPretrainedModel):
    def __init__(self, config: UniTRecConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight
        self.embed_positions = UniTRecLearnedPositionalEmbedding(config.max_position_embeddings, config.d_model)
        self.layers = nn.ModuleList([UniTRecDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids, encoder_hidden_states, encoder_attention_mask):
        input_shape = input_ids.shape
        inputs_embeds = self.embed_tokens(input_ids)
        attention_mask = _make_causal_mask(input_shape, inputs_embeds.dtype, device=inputs_embeds.device)
        encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_ids)
        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask
                )
            hidden_states = layer_outputs[0]

        return hidden_states


class UniTRecModel(UniTRecPretrainedModel):
    _keys_to_ignore_on_load_missing = ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight']
    def __init__(self, config: UniTRecConfig, dis_scoring=True, ppl_scoring=True):
        super().__init__(config)
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.encoder = UniTRecEncoder(config, self.shared)
        self.decoder = UniTRecDecoder(config, self.shared)
        self.lm_head = nn.Linear(config.d_model, self.shared.num_embeddings, bias=False)
        self.post_init()
        nn.init.zeros_(self.encoder.local_embed_positions.weight)
        self.encoder_seq_len = config.encoder_seq_len
        self.decoder_seq_len = config.decoder_seq_len
        self.IGNORE_TOKEN_ID = -100
        self.dis_scoring = dis_scoring
        self.ppl_scoring = ppl_scoring
        assert (self.dis_scoring or self.ppl_scoring)
        if self.dis_scoring:
            self.fc = nn.Linear(config.d_model, 1, bias=False)
            self.fc.weight.data.normal_(mean=0.0, std=0.02)
        if self.ppl_scoring:
            self.temperature = nn.parameter.Parameter(torch.FloatTensor([config.init_temperature]))
            self.max_temperature = config.max_temperature
        else:
            self.temperature = torch.FloatTensor([config.init_temperature])

    def load_bart(self, bart_path):
        bart = torch.load(os.path.join(bart_path, 'pytorch_model.bin'))
        def get_parameter_weight(pointer, attrs):
            p = pointer
            for attr in attrs.split('.'):
                p = getattr(p, attr)
            return p
        for n in bart:
            if '.k_proj.bias' in n:
                continue
            parameter = get_parameter_weight(self, n)
            parameter.data = bart[n]

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    # history_input_ids             : [batch_size, encoder_seq_len]
    # history_segment_ids           : [batch_size, encoder_seq_len]
    # history_global_attention_mask : [batch_size, encoder_seq_len]
    # history_local_position_ids    : [batch_size, encoder_seq_len]
    # candidate_input_ids           : [batch_size, sample_num, decoder_seq_len]
    # candidate_cls_indices         : [batch_size, sample_num]
    # targets                       : [batch_size, sample_num, decoder_seq_len]
    def forward(self, history_input_ids, history_segment_ids, history_global_attention_mask, history_local_position_ids, candidate_input_ids, candidate_cls_indices, targets):
        batch_size = history_input_ids.size(0)
        device = history_input_ids.device
        history_local_attention_mask = (history_segment_ids.unsqueeze(dim=1) == history_segment_ids.unsqueeze(dim=2)).unsqueeze(dim=1)
        history_global_position_ids = torch.arange(0, self.encoder_seq_len, dtype=torch.int32, device=device).unsqueeze(dim=0).expand(batch_size, -1)
        encoder_outputs = self.encoder(history_input_ids, history_local_position_ids, history_global_position_ids, history_local_attention_mask, history_global_attention_mask)
        if self.training:
            sample_num = candidate_input_ids.size(1)
            batch_sample_num = batch_size * sample_num
            candidate_input_ids = candidate_input_ids.view([batch_sample_num, -1])
            encoder_outputs = encoder_outputs.unsqueeze(dim=1).repeat(1, sample_num, 1, 1).view([batch_sample_num, self.encoder_seq_len, -1])
            history_global_attention_mask = history_global_attention_mask.unsqueeze(dim=1).repeat(1, sample_num, 1).view([batch_sample_num, -1])
            decoder_outputs = self.decoder(candidate_input_ids, encoder_outputs, history_global_attention_mask)
            if self.ppl_scoring:
                logits = self.lm_head(decoder_outputs).view([batch_size, sample_num, self.decoder_seq_len, -1])
                ppl = log_softmax(logits, dim=3)
                indices = torch.where(targets != self.IGNORE_TOKEN_ID, targets, 0).unsqueeze(dim=3)
                ppl = torch.gather(ppl, dim=3, index=indices).squeeze(dim=3)
                targets = targets == self.IGNORE_TOKEN_ID
                ppl.masked_fill_(targets, 0)
                ppl_scores = ppl.sum(dim=2) / (~targets).float().sum(dim=2) * torch.clamp(self.temperature, min=1 / self.max_temperature, max=self.max_temperature)
            else:
                ppl_scores = None
            if self.dis_scoring:
                indices = torch.arange(0, batch_sample_num, dtype=torch.int32, device=device) * self.decoder_seq_len + candidate_cls_indices.view(-1)
                cls_hidden_states = decoder_outputs.view([batch_sample_num * self.decoder_seq_len, -1]).index_select(dim=0, index=indices)
                dis_scores = self.fc(cls_hidden_states).view([batch_size, sample_num])
            else:
                dis_scores = None
        else:
            assert batch_size == 1, 'Inference batch size must be 1'
            sample_num = candidate_input_ids.size(0)
            encoder_outputs = encoder_outputs.expand(sample_num, -1, -1)
            history_global_attention_mask = history_global_attention_mask.expand(sample_num, -1)
            decoder_outputs = self.decoder(candidate_input_ids, encoder_outputs, history_global_attention_mask)
            if self.ppl_scoring:
                logits = self.lm_head(decoder_outputs)
                ppl = log_softmax(logits, dim=2)
                indices = torch.where(targets != self.IGNORE_TOKEN_ID, targets, 0).unsqueeze(dim=2)
                ppl = torch.gather(ppl, dim=2, index=indices).squeeze(dim=2)
                targets = targets == self.IGNORE_TOKEN_ID
                ppl.masked_fill_(targets, 0)
                ppl_scores = ppl.sum(dim=1) / (~targets).float().sum(dim=1)
            else:
                ppl_scores = None
            if self.dis_scoring:
                indices = torch.arange(0, sample_num, dtype=torch.int32, device=device) * self.decoder_seq_len + candidate_cls_indices.view(-1)
                cls_hidden_states = decoder_outputs.view([sample_num * self.decoder_seq_len, -1]).index_select(dim=0, index=indices)
                dis_scores = self.fc(cls_hidden_states).squeeze(dim=1)
            else:
                dis_scores = None
        return ppl_scores, dis_scores
