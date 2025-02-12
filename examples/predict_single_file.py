from audiobox_aesthetics.inference import AudioBoxAesthetics

model = AudioBoxAesthetics.from_pretrained("audiobox-aesthetics")
model.eval()

wav = model.load_audio("sample_audio/libritts_spk-84.wav")
predictions = model.predict_from_wavs(wav)
print(predictions)
