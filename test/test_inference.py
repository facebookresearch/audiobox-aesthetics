from audiobox_aesthetics.inference import AudioBoxAesthetics, AudioFileList, AudioFile

# cached results from running the CLI
cli_results = {
    "sample_audio/libritts_spk-84.wav": {
        "CE": 6.1027421951293945,
        "CU": 6.3574299812316895,
        "PC": 1.7401179075241089,
        "PQ": 6.733065128326416,
    },
}

model_name = "thunnai/audiobox-aesthetics"


def test_inference():
    audio_path = "sample_audio/libritts_spk-84.wav"
    audio_file = AudioFile(path=audio_path)
    model = AudioBoxAesthetics.from_pretrained(model_name)
    model.eval()

    predictions = model.predict_from_files(audio_file)
    single_pred = predictions[0]

    print(single_pred)

    assert single_pred["CE"] == cli_results[audio_path]["CE"]
    assert single_pred["CU"] == cli_results[audio_path]["CU"]
    assert single_pred["PC"] == cli_results[audio_path]["PC"]
    assert single_pred["PQ"] == cli_results[audio_path]["PQ"]


def test_inference_load_from_jsonl():
    audio_file_list = AudioFileList.from_jsonl("sample_audio/test.jsonl")
    model = AudioBoxAesthetics.from_pretrained(model_name)
    model.eval()

    audio_path = audio_file_list.files[0].path
    predictions = model.predict_from_files(audio_file_list)

    single_pred = predictions[0]
    assert single_pred["CE"] == cli_results[audio_path]["CE"]
    assert single_pred["CU"] == cli_results[audio_path]["CU"]
    assert single_pred["PC"] == cli_results[audio_path]["PC"]
    assert single_pred["PQ"] == cli_results[audio_path]["PQ"]


def test_inference_twice_on_same_audio_yields_same_result():
    audio_file = AudioFile(path="sample_audio/libritts_spk-84.wav")
    model = AudioBoxAesthetics.from_pretrained(model_name)
    model.eval()

    predictions_a = model.predict_from_files(audio_file)
    predictions_b = model.predict_from_files(audio_file)

    single_pred_a = predictions_a[0]
    single_pred_b = predictions_b[0]

    assert single_pred_a["CE"] == single_pred_b["CE"]
    assert single_pred_a["CU"] == single_pred_b["CU"]
    assert single_pred_a["PC"] == single_pred_b["PC"]
    assert single_pred_a["PQ"] == single_pred_b["PQ"]


def test_loading_from_wav():
    audio_path = "sample_audio/libritts_spk-84.wav"
    model = AudioBoxAesthetics.from_pretrained(model_name)
    model.eval()

    wav = model.load_audio(audio_path)
    predictions = model.predict_from_wavs(wav)

    single_pred = predictions[0]
    assert single_pred["CE"] == cli_results[audio_path]["CE"]
    assert single_pred["CU"] == cli_results[audio_path]["CU"]
    assert single_pred["PC"] == cli_results[audio_path]["PC"]
    assert single_pred["PQ"] == cli_results[audio_path]["PQ"]
