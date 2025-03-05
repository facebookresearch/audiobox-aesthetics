# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import json
import logging
import re
from typing import Any, Dict, List
from tqdm import tqdm
import torch
import torchaudio
import torch.nn.functional as F

from .utils import load_model

from .model.aes import AesMultiOutput, Normalize

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# STRUCT
Batch = Dict[str, Any]

# CONST
AXES_NAME = ["CE", "CU", "PC", "PQ"]


def read_wav(meta):
    path = meta["path"]

    if "start_time" in meta:
        start = meta["start_time"]
        end = meta["end_time"]
        sr = torchaudio.info(path).sample_rate
        wav, _ = torchaudio.load(
            path, frame_offset=start * sr, num_frames=(end - start) * sr
        )
    else:
        wav, sr = torchaudio.load(path)

    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)

    return wav, sr


def make_inference_batch(
    input_wavs: list,
    hop_size=10,
    window_size=10,
    sample_rate=16000,
    pad_zero=True,
):
    wavs = []
    masks = []
    weights = []
    bids = []
    offset = hop_size * sample_rate
    winlen = window_size * sample_rate
    for bid, wav in enumerate(input_wavs):
        for ii in range(0, wav.shape[-1], offset):
            wav_ii = wav[..., ii : ii + winlen]
            wav_ii_len = wav_ii.shape[-1]
            if wav_ii_len < winlen and pad_zero:
                wav_ii = F.pad(wav_ii, (0, winlen - wav_ii_len))
            mask_ii = torch.zeros_like(wav_ii, dtype=torch.bool)
            mask_ii[:, 0:wav_ii_len] = True
            wavs.append(wav_ii)
            masks.append(mask_ii)
            weights.append(wav_ii_len / winlen)
            bids.append(bid)
    return wavs, masks, weights, bids


@dataclass
class AesPredictor:
    checkpoint_pth: str
    precision: str = "bf16"
    batch_size: int = 1
    data_col: str = "path"
    sample_rate: int = 16000  # const

    def __post_init__(self):
        self.setup_model()

    def setup_model(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logging.info(f"Setting up Aesthetic model on {self.device}")

        if self.checkpoint_pth is not None:
            logging.info("Using local checkpoint ...")
            # Original way to load model directly using load_state_dict
            checkpoint_file = load_model(self.checkpoint_pth)

            # Rename keys
            with open(checkpoint_file, "rb") as fin:
                ckpt = torch.load(fin, map_location=self.device)
                state_dict = {
                    re.sub("^model.", "", k): v for (k, v) in ckpt["state_dict"].items()
                }

            model = AesMultiOutput(
                **(
                    {
                        k: ckpt["model_cfg"][k]
                        for k in [
                            "proj_num_layer",
                            "proj_ln",
                            "proj_act_fn",
                            "proj_dropout",
                            "nth_layer",
                            "use_weighted_layer_sum",
                            "precision",
                            "normalize_embed",
                            "output_dim",
                        ]
                    }
                    | {"target_transform": ckpt["target_transform"]}
                )
            )
            model.load_state_dict(state_dict)
        else:
            logging.info("Using HF from_pretrained to load AES model ...")
            # load from HF repo (using safetensors)
            model = AesMultiOutput.from_pretrained("facebook/audiobox-aesthetics")

        model.to(self.device)
        model.eval()

        self.model = model

        self.target_transform = {
            axis: Normalize(
                mean=model.target_transform[axis]["mean"],
                std=model.target_transform[axis]["std"],
            )
            for axis in AXES_NAME
        }

    def audio_resample_mono(self, data_list: List[Batch]) -> List:
        wavs = []
        for ii, item in enumerate(data_list):
            if isinstance(item[self.data_col], str):
                # wav, sr = torchaudio.load(item[self.data_col])
                wav, sr = read_wav(item)
            else:
                wav = item[self.data_col]
                sr = item["sample_rate"]

            wav = torchaudio.functional.resample(
                wav,
                orig_freq=sr,
                new_freq=self.sample_rate,
            )
            wav = wav.mean(dim=0, keepdim=True)
            wavs.append(wav)
        return wavs

    def forward(self, batch):
        with torch.inference_mode():
            bsz = len(batch)
            wavs = self.audio_resample_mono(batch)
            wavs, masks, weights, bids = make_inference_batch(
                wavs,
                10,
                10,
                sample_rate=self.sample_rate,
            )

            # collate
            wavs = torch.stack(wavs).to(self.device)
            masks = torch.stack(masks).to(self.device)
            weights = torch.tensor(weights).to(self.device)
            bids = torch.tensor(bids).to(self.device)

            assert wavs.shape[0] == masks.shape[0] == weights.shape[0] == bids.shape[0]
            preds_all = self.model({"wav": wavs, "mask": masks})
            all_result = {}
            for axis in AXES_NAME:
                preds = self.target_transform[axis].inverse(preds_all[axis])
                weighted_preds = []
                for bii in range(bsz):
                    weights_bii = weights[bids == bii]
                    weighted_preds.append(
                        (
                            (preds[bids == bii] * weights_bii).sum() / weights_bii.sum()
                        ).item()
                    )
                all_result[axis] = weighted_preds
            # re-arrenge result
            all_rows = [
                dict(zip(all_result.keys(), vv)) for vv in zip(*all_result.values())
            ]
            return all_rows


def load_dataset(path, start=None, end=None) -> List[Batch]:
    metadata = []
    with open(path) as fr:
        for ii, fi in enumerate(fr):
            if start <= ii < end:
                fi = json.loads(fi)
                metadata.append(fi)
    return metadata


def initialize_predictor(ckpt=None):
    model_predictor = AesPredictor(checkpoint_pth=ckpt, data_col="path")
    return model_predictor


def main_predict(input_file, ckpt, batch_size=10):
    predictor = initialize_predictor(ckpt)

    # load file
    if isinstance(input_file, str):
        metadata = load_dataset(input_file, 0, 2**64)
    else:
        metadata = input_file

    outputs = []
    for ii in tqdm(range(0, len(metadata), batch_size)):
        output = predictor.forward(metadata[ii : ii + batch_size])
        # convert to json string
        outputs.extend([json.dumps(x) for x in output])
    assert len(outputs) == len(
        metadata
    ), f"Output {len(outputs)} != input {len(metadata)} length"

    return outputs
