# audiobox-aesthetics

[![PyPI - Version](https://img.shields.io/pypi/v/audiobox-aesthetics)](https://pypi.org/project/audiobox-aesthetics/) [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/facebook/audiobox-aesthetics)

Unified automatic quality assessment for speech, music, and sound.

* Paper [arXiv](https://arxiv.org/abs/2502.05139) / [MetaAI](https://ai.meta.com/research/publications/meta-audiobox-aesthetics-unified-automatic-quality-assessment-for-speech-music-and-sound/).
* Blogpost [ai.meta.com](https://ai.meta.com/blog/machine-intelligence-research-new-models/)

<img src="assets/aes_model.png" alt="Model" height="400px">

## Installation

1. Install via pip
 ```
 pip install audiobox_aesthetics
 ```

2. Install directly from source

 This repository requires Python 3.9 and Pytorch 2.2 or greater. To install, you can clone this repo and run:
 ```
 pip install -e .
 ```

## Pre-trained Models

Model | S3 | HuggingFace
|---|---|---|
All axes | [checkpoint.pt](https://dl.fbaipublicfiles.com/audiobox-aesthetics/checkpoint.pt) | [HF Repo](https://huggingface.co/facebook/audiobox-aesthetics)

## Usage

### How to run prediction using CLI:

1. Create a jsonl files with the following format (for instance using `find /path/to/your/folder -type f \( -iname "*.wav" -o -iname "*.flac" \) | jq -R -c '{path: .}' > files.jsonl`):

 ```
 {"path":"/path/to/a.wav"}
 {"path":"/path/to/b.flac"}
 ...
 {"path":"/path/to/z.wav"}
 ```
 or if you only want to predict aesthetic scores from certain timestamp
 ```
 {"path":"/path/to/a.wav", "start_time":0, "end_time": 5}
 {"path":"/path/to/b.flac", "start_time":3, "end_time": 10}
 ```
 and save it as `input.jsonl`

2. Run following command
 ```
 audio-aes input.jsonl --batch-size 100 > output.jsonl
 ```
 If you haven't downloade the checkpoint, the script will try to download it automatically. Otherwise, you can provide the path by `--ckpt /path/to/checkpoint.pt`

 If you have SLURM, run the following command
 ```
 audio-aes input.jsonl --batch-size 100 --remote --array 5 --job-dir $HOME/slurm_logs/ --chunk 1000 > output.jsonl
 ```
 Please adjust CPU & GPU settings using `--slurm-gpu, --slurm-cpu` depending on your nodes.


3. Output file will contain the same number of rows as `input.jsonl`. Each row contains 4 axes of prediction with a JSON-formatted dictionary. Check the following table for more info:
 
 Axes name | Full name
 |---|---|
 CE | Content Enjoyment
 CU | Content Usefulness
 PC | Production Complexity
 PQ | Production Quality
    
 Output line example:
 ```
 {"CE": 5.146, "CU": 5.779, "PC": 2.148, "PQ": 7.220}
 ```

4. (Extra) If you want to extract only one axis (i.e. CE), post-process the output file with the following command using `jq` utility: 
    
    ```jq '.CE' output.jsonl > output-aes_ce.txt```


### How to run prediction from Python script or interpreter

1. Infer from file path
```
from audiobox_aesthetics.infer import initialize_predictor
predictor = initialize_predictor()
predictor.forward([{"path":"/path/to/a.wav"}, {"path":"/path/to/b.flac"}])
```

2. Infer from torch tensor
```
from audiobox_aesthetics.infer import initialize_predictor
predictor = initialize_predictor()
wav, sr = torchaudio.load("/path/to/a.wav")
predictor.forward([{"path":wav, "sample_rate": sr}])
```
### How to load model using HuggingFace way (for finetuning, etc)

```
from audiobox_aesthetics.model.aes import AesMultiOutput
model = AesMultiOutput.from_pretrained("facebook/audiobox-aesthetics")
# finetune the model
...
# finished finetuning & upload the model
model.push_to_hub("<your_hf_username>/<your_hf_repo>")
```

## Evaluation dataset
We released our evaluation dataset consisting of 4 axes of aesthetic annotation scores. 

Here, we show an example of how to read and re-map each annotation to the actual audio file.
```
{
 "data_path": "/your_path/LibriTTS/train-clean-100/1363/139304/1363_139304_000011_000000.wav", 
 "Production_Quality": [8.0, 8.0, 8.0, 8.0, 8.0, 9.0, 8.0, 5.0, 8.0, 8.0], 
 "Production_Complexity": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
 "Content_Enjoyment": [8.0, 6.0, 8.0, 5.0, 8.0, 8.0, 8.0, 6.0, 8.0, 6.0], 
 "Content_Usefulness": [8.0, 6.0, 8.0, 7.0, 8.0, 9.0, 8.0, 6.0, 10.0, 7.0]
}
```
1. Recognize the dataset name from data_path. In the example, it is LibriTTS.
2. Replace "/your_path/" into your downloaded LibriTTS directory. 
3. Each axis contains 10 scores annotated by 10 different human annotators.

data_path | URL
|---|---|
LibriTTS |  https://openslr.org/60/
cv-corpus-13.0-2023-03-09 | https://commonvoice.mozilla.org/en/datasets
EARS | https://sp-uhh.github.io/ears_dataset/
MUSDB18 | https://sigsep.github.io/datasets/musdb.html
musiccaps | https://www.kaggle.com/datasets/googleai/musiccaps
(audioset) unbalanced_train_segments | https://research.google.com/audioset/dataset/index.html 
PAM | https://zenodo.org/records/10737388

## License
The majority of audiobox-aesthetics is licensed under CC-BY 4.0, as found in the LICENSE file.
However, portions of the project are available under separate license terms: [https://github.com/microsoft/unilm](https://github.com/microsoft/unilm) is licensed under MIT license.

## Citation
If you found this repository useful, please cite the following BibTeX entry.

```
@article{tjandra2025aes,
    title={Meta Audiobox Aesthetics: Unified Automatic Quality Assessment for Speech, Music, and Sound},
    author={Andros Tjandra and Yi-Chiao Wu and Baishan Guo and John Hoffman and Brian Ellis and Apoorv Vyas and Bowen Shi and Sanyuan Chen and Matt Le and Nick Zacharov and Carleigh Wood and Ann Lee and Wei-Ning Hsu},
    year={2025},
    url={https://arxiv.org/abs/2502.05139}
}
```

## Acknowledgements
Part of the model code is copied from [https://github.com/microsoft/unilm/tree/master/wavlm](WavLM).

