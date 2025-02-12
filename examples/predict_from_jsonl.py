from audiobox_aesthetics.inference import AudioBoxAesthetics, AudioFileList

model = AudioBoxAesthetics.from_pretrained("audiobox-aesthetics")
model.eval()


audio_file_list = AudioFileList.from_jsonl("examples/example.jsonl")
predictions = model.predict_from_files(audio_file_list)
print(predictions)
