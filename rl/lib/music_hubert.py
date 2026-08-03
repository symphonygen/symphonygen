import torch
from torch import nn
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.hubert.modeling_hubert import HubertFeatureEncoder, HubertModel, HubertEncoder
from rl.lib.config_music_hubert import MusicHubertConfig

class MusicHubertFeatureProjection(nn.Module):
    def __init__(self, config: MusicHubertConfig):
        super().__init__()
        self.feat_proj_layer_norm = config.feat_proj_layer_norm

        self.feature_dimension = config.conv_dim[-1]
        if self.feat_proj_layer_norm:
            self.layer_norm = nn.LayerNorm(self.feature_dimension, eps=config.layer_norm_eps)
        self.projection = nn.Linear(self.feature_dimension, config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # Non-projected hidden states are needed for quantization
        if self.feat_proj_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class MusicHubertModel(HubertModel):
    # Overwrite config class
    config_class = MusicHubertConfig
    base_model_prefix = "music_hubert"

    def __init__(self, config: MusicHubertConfig):
        """
        initialize the with the grandparent method HubertPreTrainedModel.__init__()
        and modify the HuBERTModel.__init__()
        """
        super(HubertModel, self).__init__(config)

        self.config = config

        self.feature_extractor = HubertFeatureEncoder(config)
        self.feature_projection = MusicHubertFeatureProjection(config) # Replace Feature Projection for introcuing new feature

        if config.feature_extractor_cqt:
            raise NotImplementedError

        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        self.encoder = HubertEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor.forward(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # Compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder.forward(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0] # take last_hidden from encoder output

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
