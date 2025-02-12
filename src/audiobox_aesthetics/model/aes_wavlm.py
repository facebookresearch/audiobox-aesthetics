# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import sys
from torch import nn
import torch

from audiobox_aesthetics.model.utils import create_mlp_block
from audiobox_aesthetics.model.wavlm import WavLM, WavLMConfig


DEFAULT_AUDIO_CFG = WavLMConfig(
    {
        "extractor_mode": "default",
        "encoder_layers": 12,
        "encoder_embed_dim": 768,
        "encoder_ffn_embed_dim": 3072,
        "encoder_attention_heads": 12,
        "activation_fn": "gelu",
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "activation_dropout": 0.0,
        "encoder_layerdrop": 0.05,
        "dropout_input": 0.1,
        "dropout_features": 0.1,
        "layer_norm_first": False,
        "conv_feature_layers": "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        "conv_bias": False,
        "feature_grad_mult": 0.1,
        "mask_length": 10,
        "mask_prob": 0.8,
        "mask_selection": "static",
        "mask_other": 0.0,
        "no_mask_overlap": False,
        "mask_min_space": 1,
        "mask_channel_length": 10,
        "mask_channel_prob": 0.0,
        "mask_channel_selection": "static",
        "mask_channel_other": 0.0,
        "no_mask_channel_overlap": False,
        "mask_channel_min_space": 1,
        "conv_pos": 128,
        "conv_pos_groups": 16,
        "relative_position_embedding": True,
        "num_buckets": 320,
        "max_distance": 800,
        "gru_rel_pos": True,
        "normalize": False,
    }
)


@dataclass(eq=False)
class Normalize:
    mean: float
    std: float

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


AXES_NAME = ["CE", "CU", "PC", "PQ"]


@dataclass(eq=False)
class WavlmAudioEncoderMultiOutput(nn.Module):
    proj_num_layer: int = 1
    proj_ln: bool = False
    proj_act_fn: str = "gelu"
    proj_dropout: float = 0
    nth_layer: int = 13
    use_weighted_layer_sum: bool = True
    precision: str = "32"
    normalize_embed: bool = True
    output_dim: int = 1

    def __post_init__(self):
        super().__init__()
        amodel_cfg = DEFAULT_AUDIO_CFG
        self.wavlm_model = WavLM(amodel_cfg)
        wavlm_out_dim = self.wavlm_model.cfg.encoder_embed_dim

        self.axes_name = AXES_NAME
        self.proj_layer = nn.ModuleDict(
            {
                x: nn.Sequential(
                    *create_mlp_block(
                        wavlm_out_dim,
                        self.output_dim,
                        self.proj_num_layer,
                        self.proj_act_fn,
                        self.proj_ln,
                        dropout=self.proj_dropout,
                    )
                )
                for x in self.axes_name
            }
        )
        if self.use_weighted_layer_sum:
            self.layer_weights = nn.ParameterDict(
                {
                    x: torch.nn.Parameter(torch.ones(self.nth_layer) / (self.nth_layer))
                    for x in self.axes_name
                }
            )

        precision_map = {
            "64": torch.float64,
            "32": torch.float32,
            "16": torch.half,
            "bf16": torch.bfloat16,
        }
        self.precision = precision_map[str(self.precision)]
        self.enable_autocast = str(self.precision) in {"16", "bf16"}
        print(
            f"precision: {self.precision}, enable autocast: {self.enable_autocast}",
            file=sys.stderr,
        )

    def forward(self, batch):
        assert batch["wav"].ndim == 3

        # frames: [B, C, T]
        wav = batch["wav"].squeeze(1)

        if "mask" in batch:
            padding_mask = ~batch["mask"].squeeze(1)
        else:
            padding_mask = torch.zeros_like(wav, dtype=torch.bool)

        with (
            torch.amp.autocast(
                device_type=wav.device.type,
                dtype=self.precision,
                enabled=self.enable_autocast,
            ),
            torch.no_grad(),
        ):
            if self.wavlm_model.cfg.normalize:
                wav = torch.nn.functional.layer_norm(wav, wav.shape)
            (_, all_outputs), embed_padding_mask = self.wavlm_model.extract_features(
                source=wav,
                padding_mask=padding_mask,
                output_layer=self.nth_layer,
                ret_layer_results=True,
            )
            all_outputs = torch.stack([gg[0] for gg in all_outputs], dim=-1)  # T B C L
            preds = {}
            for name in self.axes_name:
                if self.use_weighted_layer_sum:
                    norm_weights = torch.nn.functional.softmax(
                        self.layer_weights[name], dim=-1
                    )  # L
                    audio_embed = torch.einsum("tbcl,l->btc", all_outputs, norm_weights)
                else:
                    audio_embed = all_outputs[-1][0].transpose(1, 0)

                embed_mask = (
                    (~embed_padding_mask).unsqueeze(dim=-1).type_as(audio_embed)
                )
                audio_embed = (audio_embed * embed_mask).sum(dim=1) / embed_mask.sum(
                    dim=1
                ).clamp(min=1)
                if self.normalize_embed:
                    audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)

                preds[name] = self.proj_layer[name](audio_embed).squeeze(-1)
        return preds
