from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

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
selected_options = {}  # 각 분야별 선택 옵션 저장 딕셔너리

@app.route('/')
def index():
    return redirect(url_for('survey'))

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
            print(selected_options)
            return render_template('start_page.html')

        print(selected_options)

    return render_template('survey_page.html', 
                           survey_page=next_page, 
                           question=survey_questions[next_page], 
                           options=survey_options[next_page], 
                           selected_option=selected_options.get(next_page)
                           )

if __name__ == '__main__':
    app.run(debug=True)