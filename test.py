from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import mysql.connector
import random
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

app = Flask(__name__)
app.secret_key = "ABCD"

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

connection = mysql.connector.connect(
    host='db4free.net',
    user='hyeongjin',
    password='abcd1234',
    database='opic_automatic'
)

task_queue = Queue() # Queue containing question_page_number to be transcribed 
transcript_list = {}
selected_options = {}  # 각 분야별 선택 옵션 저장 딕셔너리
question_list = [0 for i in range(15)]

# 설문지 내용
# 1 = '종사 분야' 
# 2 = '거주 방식' 
# 3 = '여가 및 취미'
survey_questions = {
    1: "현재 귀하는 어느 분야에 종사하고 계십니까?",
    2: "현재 귀하는 어디에 살고 계십니까?",
    3: "귀하는 여가 및 취미활동으로 주로 무엇을 하십니까? (두 개 이상 선택)",
}

survey_options = {
    1: ["사업자/직장인", "학생", "취업준비생"],
    2: ["개인주택이나 아파트에 홀로 거주", "친구나 룸메이트와 함께 주택이나 아파트에 거주", "가족과 함께 주택이나 아파트에 거주"],
    3: ["운동", "게임", "SNS", "문화생활", "여행", "자기관리", "예술활동", "자기개발"],
}

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    session['username'] = username
    return render_template('start_page.html')

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    global selected_options
    next_page = 1

    if request.method == 'POST':
        survey_page = int(request.form.get('survey-page'))
        next_page_direction = request.form.get('next-page-direction')

        if next_page_direction == 'next':
            next_page = survey_page + 1
        else: # next_page_direction == 'back'
            next_page = survey_page - 1

        if survey_page == 3:
            selected_options[survey_page] = request.form.getlist('option')
        else:
            selected_options[survey_page] = request.form.get('option')

        if next_page == 4:
            global question_list
            print(selected_options)
            cursor = connection.cursor()
            index = 0
            question_list[index] = "Can you introduce yourself in as much detail as possible?"
            index += 1

            query = "SELECT question_text FROM question WHERE property = %s AND link = %s"
            option_value = selected_options.get(1) # 종사 분야
            for i in range(3):
                cursor.execute(query, (option_value, i))
                question_list[index] = cursor.fetchone()[0]
                index += 1

            option_value = selected_options.get(2) # 거주 방식
            for i in range(3):
                cursor.execute(query, (option_value, i))
                question_list[index] = cursor.fetchone()[0]
                index += 1

            option_value = random.choice(selected_options.get(3)) # 여가 및 취미
            for i in range(3):
                cursor.execute(query, (option_value, i))
                question_list[index] = cursor.fetchone()[0]
                index += 1

            option_value = random.choice(['롤플레이1', '롤플레이2', '롤플레이3', '롤플레이4'])
            for i in range(3):
                cursor.execute(query, (option_value, i))
                question_list[index] = cursor.fetchone()[0]
                index += 1

            option_value = random.choice(['돌발질문:코로나', '돌발질문:코인', '돌발질문:출산율'])
            for i in range(2):
                cursor.execute(query, (option_value, i))
                question_list[index] = cursor.fetchone()[0]
                index += 1
            for question in question_list:
                print(question)
            return redirect(url_for('background_start'))

        print(selected_options)

    return render_template('survey_page.html', 
                           survey_page=next_page, 
                           question=survey_questions[next_page], 
                           options=survey_options[next_page], 
                           selected_option=selected_options.get(next_page)
                           )

@app.route('/background_start')
def background_start():
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
    session['question_page_number'] = 1
    return render_template("question_page.html", question_page_number = session['question_page_number'])

@app.route('/question_page', methods=['GET', 'POST'])
def question_page():
    session['question_page_number'] += 1
    if session['question_page_number'] < 16:
        return render_template("question_page.html", question_page_number = session['question_page_number'])
    else:
        return render_template("processing_score_page.html")

@app.route('/get_question')
def get_text():
    text = question_list[session['question_page_number'] - 1]
    return jsonify({'text': text})

@app.route('/get_transcriptListLength')
def get_transcriptListLength():
    transcriptListLength = len(transcript_list)
    return jsonify({'len': transcriptListLength})

@app.route('/save_recording', methods=['POST'])
def save_recording():
    try:
        file = request.files['audio_file']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], f'record{session['question_page_number']}.wav'))

        return 'Recording saved successfully!', 200
    except Exception as e:
        return f'Error saving recording: {e}', 500
    
@app.route('/transcript')
def transcript():
    try:
        task_queue.put(session['question_page_number'])
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

    cursor = connection.cursor()
    models_and_tokenizers = {criteria: load_model_and_tokenizer(path) for criteria, path in models_paths.items()}

    def predict(criteria, text, models_and_tokenizers):
        model, tokenizer = models_and_tokenizers[criteria]
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits.squeeze().item()

        rounded_predictions = round(predictions, 2)
        return rounded_predictions

    question_num = 0
    # scores_summary = {f"Q{idx}_{criteria}": [] for idx, criteria in enumerate(models_paths.keys())}
    scores_summary = {criteria: [] for criteria in models_paths.keys()}
    sorted_transcript_list = [transcript_list[key] for key in sorted(transcript_list.keys())]

    for question, transcript in zip(question_list, sorted_transcript_list):
        text = f"question : {question}  \n\n answer : {transcript}"

        question_scores = []
        question_num += 1
        for criteria in models_paths.keys():
        # for criteria in scores_summary.keys():
            score = predict(criteria, text, models_and_tokenizers)
            question_scores.append(score)
            scores_summary[criteria].append(score)
            # print(f"{criteria}: {score}")

        question_average = sum(question_scores) / len(question_scores)
        print(f"질문 {question}에 대한 평균 점수: {question_average}")
        insert_query = "INSERT INTO answer (username, question_number, question_text, answer_text, score) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(insert_query, (session['username'], question_num, question, transcript, question_average))
    
    criteria_averages = {criteria: sum(scores) / len(scores) for criteria, scores in scores_summary.items()}
    total_average_score = sum(criteria_averages.values()) / len(criteria_averages)

    print(f"총 평균 점수: {total_average_score}")

    if total_average_score >= 4.5: 
        grade = "AL"
    elif 3.5 <= total_average_score < 4.5:
        grade = "IH"
    elif 2.5 <= total_average_score < 3.5:
        grade = "IM"
    elif 1.5 <= total_average_score < 2.5:
        grade = "IL"
    else:
        grade = "NH"

    insert_grade_query = "INSERT INTO grade (username, grade) VALUES (%s, %s)"
    cursor.execute(insert_grade_query, (session['username'], grade))
    connection.commit()
    return render_template("show_score_page.html", grade = grade)

if __name__ == '__main__':
    app.run(debug=True)