from flask import Flask, request, send_file, render_template, abort, current_app, jsonify
import os
import torch
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline
)
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './records'

device = "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device).float()
processor = AutoProcessor.from_pretrained(model_id)

whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=256,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

transcript_list = {}
question_page_number = 1
question_list = [
    'Can you introduce yourself in as much detail as possible?',
    'You indicated that you are a student. What are some factors that you consider when choosing a class? Try to be as specific as possible.',
    'You need to submit a report by this weekend and you only have three days left to complete it. Unfortunately, you became extremely sick and are unable to finish the report on time. Call the professor and explain your situation so that you can get an extension.'
]

@app.route('/')
def index():
    return render_template("question_page.html", question_page_number = question_page_number)

@app.route('/get_question')
def get_text():
    text = question_list[question_page_number - 1]
    return jsonify({'text': text})

@app.route('/save_recording', methods=['POST'])
def save_recording():
    try:
        file = request.files['audio_file']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], f'record{question_page_number}.wav'))

        return 'Recording saved successfully!', 200
    except Exception as e:
        return f'Error saving recording: {e}', 500
    
@app.route('/transcript')
def transcript():
    # thread background 작업
    def backgroundTask(audio_file, question_page_number):
        result = whisper_pipe(audio_file, generate_kwargs={"language": "english"})
        transcript_list[question_page_number] = result["text"]
        for key, value in transcript_list.items():
            print(key, value)

    def startBackgroundTask(audio_file, question_page_number):
        thread = threading.Thread(target=backgroundTask, args=(audio_file, question_page_number,))
        thread.start()

    global question_page_number
    audio_file = f'records/record{question_page_number}.wav'
    startBackgroundTask(audio_file, question_page_number)
    
    question_page_number += 1
    return render_template("question_page.html", question_page_number = question_page_number)

if __name__ == '__main__':
    app.run(port=5001, debug=True)