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
from queue import Queue
import time

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

task_queue = Queue() # Queue containing question_page_number to be transcribed 

transcript_list = {}

question_page_number = 1
question_list = ["You need to submit a report by this weekend and you only have three days left to complete it. Unfortunately, you became extremely sick and are unable to finish the report on time. Call the professor and explain your situation so that you can get an extension.",
                 "You indicated that you are a student. What are some factors that you consider when choosing a class? Try to be as specific as possible.",
                 "What was one of the most difficult projects or homework assignments that you have ever done for a class? Why was it difficult for you? Please explain the project or homework assignment in detail.",
                 "You need to submit a report by this weekend and you only have three days left to complete it. Unfortunately, you became extremely sick and are unable to finish the report on time. Call the professor and explain your situation so that you can get an extension.",
                 "Please describe a common space that you share with your roommate at home. What kind of furniture or home appliances are there? What is the layout of the place? Please use as much detail as possible.",
                 "When we live with other people, we tend to experience some difficulties. Tell me about a disagreement you had with your roommate. What was the situation and how did it start? How did you resolve the situation with your roommate?",
                 "Your roommate says that she/he wants to bring her/his family dog to the house that you are currently living in. Unfortunately, you do not want to live with a pet. Give your roommate three reasons why the dog cannot live in the house.",
                 "Which city in the world would you currently like to visit the most? Why did you choose that city? What does it look like? What kind of places do you want to visit there? Please describe in detail why you want to visit this city.",
                 "Tell me about a difficult situation that you experienced while you were traveling. What exactly happened? How did you solve the problem? Describe the experience in detail.",
                 "You booked a hotel and when you arrived at the hotel the manager said that there were no reservations under your name. Please explain your situation to the manager in order to reserve the room.",
                 "What type of bar do you want to visit? What is the decor or mood like? What kind of drinks are they selling? Imagine and describe the bar that you want to visit in detail.",
                 "You went to a bar, but you don’t know anything about their drinks menu. Please ask the bartender three questions about the signature drink.",
                 "Your friend is having a birthday party at a bar. However, a mutual friend called you to ask that you make an excuse for them for being late for the party. Please explain the situation to the host of the party and give two excuses for your friend’s tardiness.",
                 "There are so many illegal students and other foreign visitors who stayed on in Korea after the outbreak of Covid-19. A lot of these people stay in Korea illegally instead of going back to their home countries. Why do you think this is happening?",
                 "In this situation, do you think the government has to intervene and manage the situation? Why or why not? Please elaborate on your ideas.",
                 ]
score_list = []

@app.route('/')
def index():
    # thread background 작업
    def backgroundTask():
        while True:
            question_page_number_ = task_queue.get()
            audio_file = f'records/record{question_page_number_}.wav'
            result = whisper_pipe(audio_file, generate_kwargs={"language": "english"})
            transcript_list[question_page_number_] = result["text"]
            for key, value in transcript_list.items():
                print(key, value)

            if len(transcript_list) == 15:
                break
            
    def startBackgroundTask():
        thread = threading.Thread(target=backgroundTask, args=())
        thread.start()

    startBackgroundTask()
    return render_template("question_page.html", question_page_number = question_page_number)

@app.route('/get_question')
def get_question():
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
    try:
        task_queue.put(question_page_number)
        return jsonify({'message': 'task_queue saved successfully!'}), 200
    except Exception as e:
        return jsonify({'error': f'Error saving recording: {e}'}), 500

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


@app.route('/question_page')
def question_page_next():
    global question_page_number
    question_page_number += 1
    if question_page_number < 16:
        return render_template("question_page.html", question_page_number = question_page_number)
    else:
        return render_template("processing_score_page.html")

if __name__ == '__main__':
    app.run(port=5001, debug=True)