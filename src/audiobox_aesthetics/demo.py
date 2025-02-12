import gradio as gr
from audiobox_aesthetics.inference import (
    AudioBoxAesthetics,
    AudioFile,
    AXIS_NAME_LOOKUP,
)

# Load the pre-trained model
model = AudioBoxAesthetics.from_pretrained("audiobox-aesthetics")
model.eval()


def predict_aesthetics(audio_file):
    # Create an AudioFile instance
    audio_file_instance = AudioFile(path=audio_file)

    # Predict using the model
    predictions = model.predict_from_files(audio_file_instance)

    single_prediction = predictions[0]

    data_view = [
        [AXIS_NAME_LOOKUP[key], value] for key, value in single_prediction.items()
    ]

    return single_prediction, data_view


def create_demo():
    # Create a Gradio Blocks interface
    with gr.Blocks() as demo:
        gr.Markdown("# AudioBox Aesthetics Prediction")
        with gr.Group():
            gr.Markdown("""Upload an audio file to predict its aesthetic scores.
                        
                This demo uses the AudioBox Aesthetics model to predict aesthetic scores for audio files along 4 axes:
                - Content Enjoyment (CE)
                - Content Usefulness (CU) 
                - Production Complexity (PC)
                - Production Quality (PQ)
                
                Scores range from 0 to 10.

                For more details, see the [paper](https://arxiv.org/abs/2502.05139) or [code](https://github.com/facebookresearch/audiobox-aesthetics/tree/main).
            """)

        with gr.Row():
            with gr.Group():
                with gr.Column():
                    audio_input = gr.Audio(
                        sources="upload", type="filepath", label="Upload Audio"
                    )
                    submit_button = gr.Button("Predict", variant="primary")
            with gr.Group():
                with gr.Column():
                    output_data = gr.Dataframe(
                        headers=["Axes name", "Score"],
                        datatype=["str", "number"],
                        label="Aesthetic Scorest",
                    )
                    output_text = gr.Textbox(label="Raw prediction", interactive=False)

        submit_button.click(
            predict_aesthetics,
            inputs=audio_input,
            outputs=[output_text, output_data],
        )

        # Add examples
        gr.Examples(
            examples=[
                "sample_audio/libritts_spk-84.wav",
                "sample_audio/libritts_spk-3170.wav",
            ],
            inputs=audio_input,
            outputs=[output_text, output_data],
            fn=predict_aesthetics,
            cache_examples=True,
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
