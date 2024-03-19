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
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# from celery import Celery

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './records'
# app.config.update(
#     CELERY_BROKER_URL='redis://localhost:6379',
#     CELERY_RESULT_BACKEND='redis://localhost:6379'
# )

# celery로 전사

# BROKER_URL = 'redis://localhost:6379/0'
# CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
# celery = Celery(app.name, broker=BROKER_URL, backend=CELERY_RESULT_BACKEND)

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
transcript_list[2] = "two"
transcript_list[1] = "one"
transcript_list[3] = "three"
transcript_list[5] = "five"
transcript_list[4] = "four"
transcript_list[6] = "six"
transcript_list[7] = "seven"
transcript_list[8] = "eight"
transcript_list[9] = "nine"
transcript_list[11] = "eleven"
transcript_list[10] = "ten"
transcript_list[12] = "twelve"
transcript_list[13] = "thirteen"
transcript_list[14] = "fourteen"
question_page_number = 15
question_list = [
    'Can you introduce yourself in as much detail as possible?',
    'You indicated that you are a student. What are some factors that you consider when choosing a class? Try to be as specific as possible.',
    'You need to submit a report by this weekend and you only have three days left to complete it. Unfortunately, you became extremely sick and are unable to finish the report on time. Call the professor and explain your situation so that you can get an extension.',
    'Can you introduce yourself in as much detail as possible?',
    'Can you introduce yourself in as much detail as possible?',
    'Can you introduce yourself in as much detail as possible?',
    'Can you introduce yourself in as much detail as possible?',
    'Can you introduce yourself in as much detail as possible?',
    'Can you introduce yourself in as much detail as possible?',
    'Can you introduce yourself in as much detail as possible?',
    'Can you introduce yourself in as much detail as possible?',
    'Can you introduce yourself in as much detail as possible?',
    'Can you introduce yourself in as much detail as possible?',
    'Can you introduce yourself in as much detail as possible?',
    'Can you introduce yourself in as much detail as possible?',
]
score_list = []

# @celery.task()
# def backgroundTask(audio_file, question_page_number):
#     with app.app_context():
#         result = whisper_pipe(audio_file, generate_kwargs={"language": "english"})
#         transcript_list[question_page_number] = result["text"]
#         for key, value in transcript_list.items():
#             print(key, value)

@app.route('/')
def index():
    return render_template("question_page.html", question_page_number = question_page_number)

@app.route('/get_question')
def get_text():
    text = question_list[question_page_number - 1]
    return jsonify({'text': text})

@app.route('/get_transcriptListLength')
def get_transcriptListLength():
    transcriptListLength = len(transcript_list)
    return jsonify({'len': transcriptListLength})

@app.route('/get_ScoreListLength')
def get_ScoreListLength():
    scoreListLength = len(score_list)
    return jsonify({'len': scoreListLength})

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

    try:
        audio_file = f'records/record{question_page_number}.wav'
        startBackgroundTask(audio_file, question_page_number)
        # Celery 백그라운드 실행
        # backgroundTask.delay(audio_file, question_page_number)
        return 'transcript successfully!', 200
    except Exception as e:
        return f'Error transcript: {e}', 500

@app.route('/grading')
def grading():
    def load_model_and_tokenizer(model_path):
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        return model, tokenizer
    
    models_paths = {
        "Task_Completion": "model/Task_Completion",
        "Delivery": "model/Delivery",
        "Accuracy": "model/Accuracy",
        "Appropriateness": "model/Appropriateness"
    }

    models_and_tokenizers = {criteria: load_model_and_tokenizer(path) for criteria, path in models_paths.items()}

    def predict(criteria, text, models_and_tokenizers):
        model, tokenizer = models_and_tokenizers[criteria]
    
        # 텍스트를 토크나이징하여 모델 입력 형식으로 변환
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    
        # GPU 사용 가능 여부 확인
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
        # 예측 수행
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # probabilities = torch.softmax(logits, dim=1)
    
        # return probabilities
        return logits
    
    sorted_transcript_list = [transcript_list[key] for key in sorted(transcript_list.keys())]
    # 각 평가 항목별로 예측 수행
    for question, transcript in zip(question_list, sorted_transcript_list):
        a = f"question : {question}  \n\n answer : {transcript}"
        scores = {}
        # 각 평가 항목에 대해 점수 예측
        for criteria in models_paths.keys():
            print(a)
            score = predict(criteria, a, models_and_tokenizers)
            scores[criteria] = score
            print(f"{criteria}: {score}")
            
        print('\n\n\n')
        score_list.append(scores)

    for s in score_list:
        print(s)

    return render_template("show_score_page.html", score_list = score_list)


@app.route('/question_page_next')
def question_page_next():
    global question_page_number
    question_page_number += 1
    if question_page_number < 16:
        return render_template("question_page.html", question_page_number = question_page_number)
    else:
        return render_template("processing_score_page.html")

if __name__ == '__main__':
    app.run(port=5001, debug=True)