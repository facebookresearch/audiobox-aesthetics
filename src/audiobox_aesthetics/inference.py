import re

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from audiobox_aesthetics.model.aes_wavlm import Normalize, WavlmAudioEncoderMultiOutput
from audiobox_aesthetics.infer import make_inference_batch

from pydantic import BaseModel
import torchaudio

from pydantic import Field
from typing import Optional, List
import json

AXIS_NAME_LOOKUP = {
    "CE": "Content Enjoyment",
    "CU": "Content Usefulness",
    "PC": "Production Complexity",
    "PQ": "Production Quality",
}


class AudioFile(BaseModel):
    """
    Audio file to be processed
    """

    path: str
    start_time: Optional[float] = Field(None, description="Start time in seconds")
    end_time: Optional[float] = Field(None, description="End time in seconds")


class AudioFileList(BaseModel):
    """
    List of audio files to be processed
    """

    files: List[AudioFile]

    @classmethod
    def from_jsonl(cls, filename: str) -> "AudioFileList":
        audio_files = []
        with open(filename, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                audio_file = AudioFile(**data)
                audio_files.append(audio_file)
        return cls(files=audio_files)


# model


class AudioBoxAesthetics(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="audiobox-aesthetics",
    repo_url="https://github.com/facebookresearch/audiobox-aesthetics",
):
    def __init__(
        self,
        proj_num_layer: int = 1,
        proj_ln: bool = False,
        proj_act_fn: str = "gelu",
        proj_dropout: float = 0.0,
        nth_layer: int = 13,
        use_weighted_layer_sum: bool = True,
        precision: str = "bf16",
        normalize_embed: bool = True,
        output_dim: int = 1,
        target_transform: dict = None,
        sample_rate: int = 16_000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.encoder = WavlmAudioEncoderMultiOutput(
            proj_num_layer=proj_num_layer,
            proj_ln=proj_ln,
            proj_act_fn=proj_act_fn,
            proj_dropout=proj_dropout,
            nth_layer=nth_layer,
            use_weighted_layer_sum=use_weighted_layer_sum,
            precision=precision,
            normalize_embed=normalize_embed,
            output_dim=output_dim,
        )
        self.target_transform = {
            axis: Normalize(
                mean=target_transform[axis]["mean"],
                std=target_transform[axis]["std"],
            )
            for axis in target_transform.keys()
        }

    def _load_base_checkpoint(self, checkpoint_pth: str):
        with open(checkpoint_pth, "rb") as fin:
            ckpt = torch.load(fin, map_location="cpu", weights_only=True)
            state_dict = {
                re.sub("^model.", "", k): v for (k, v) in ckpt["state_dict"].items()
            }

            self.encoder.load_state_dict(state_dict)

    def forward(self, batch, inference_mode: bool = True):
        if inference_mode:
            with torch.inference_mode():
                result = self.encoder(batch)
        else:
            result = self.encoder(batch)
        return result

    def _process_single_audio(self, wav: torch.Tensor, sample_rate: int):
        """
        Process a single audio file to the target sample rate and return a tensor of shape (1, 1, T)
        """
        target_sample_rate = self.sample_rate
        wav = torchaudio.functional.resample(wav, sample_rate, target_sample_rate)

        # convert to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav, target_sample_rate

    def load_audio(self, path: str, start_time: float = None, end_time: float = None):
        """
        Load an audio file form path

        Args:
            path: str - path to the audio file
            start_time: float - start time in seconds
            end_time: float - end time in seconds
        Returns:
            wav: torch.Tensor - audio tensor of shape (1, 1, T)
        """
        wav, sample_rate = torchaudio.load(path)
        if start_time is not None and end_time is not None:
            if start_time and end_time:
                wav = wav[
                    :, int(start_time * sample_rate) : int(end_time * sample_rate)
                ]
            elif start_time:
                wav = wav[:, int(start_time * sample_rate) :]
            elif end_time:
                wav = wav[:, : int(end_time * sample_rate)]

        wav, _sr = self._process_single_audio(wav, sample_rate)

        return wav

    def predict_from_files(
        self, audio_file_list: AudioFileList | AudioFile
    ) -> List[dict]:
        """
        Predict the aesthetic score for a list of audio files
        """
        if isinstance(audio_file_list, AudioFile):
            audio_file_list = AudioFileList(files=[audio_file_list])

        wavs = [
            self.load_audio(file.path, file.start_time, file.end_time)
            for file in audio_file_list.files
        ]

        return self.predict_from_wavs(wavs)

    def predict_from_wavs(self, wavs: List[torch.Tensor] | torch.Tensor):
        """
        Predict the aesthetic score for a single audio file

        Args:
            wavs: List[torch.Tensor] - list of audio tensors of shape (1, 1, T) - must be at the sample rate of the model
        Returns:
            preds: List[dict] - list of dictionaries containing the aesthetic scores for each axis
        """

        if isinstance(wavs, torch.Tensor):
            wavs = [wavs]

        n_wavs = len(wavs)

        wavs, masks, weights, bids = make_inference_batch(
            wavs,
            10,
            10,
            sample_rate=self.sample_rate,
        )

        # stack wavs, masks, weights, bids
        wavs = torch.stack(wavs)
        masks = torch.stack(masks)
        weights = torch.tensor(weights)
        bids = torch.tensor(bids)

        if not wavs.shape[0] == masks.shape[0] == weights.shape[0] == bids.shape[0]:
            raise ValueError("Batch size mismatch")

        preds_all = self.forward({"wav": wavs, "mask": masks})
        all_result = {}

        # predict scores across all axis
        for axis in self.target_transform.keys():
            preds = self.target_transform[axis].inverse(preds_all[axis])
            weighted_preds = []
            for bii in range(n_wavs):
                weights_bii = weights[bids == bii]
                weighted_preds.append(
                    (
                        (preds[bids == bii] * weights_bii).sum() / weights_bii.sum()
                    ).item()
                )
            all_result[axis] = weighted_preds
        # re-arrenge result
        preds = [dict(zip(all_result.keys(), vv)) for vv in zip(*all_result.values())]

        return preds
