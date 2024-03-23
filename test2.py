from flask import Flask, render_template, request

app = Flask(__name__)

# 설문지 내용
survey = {
    "part1": ["사업/회사", "재택근무/재택사업", "교사/교육자", "일 경험 없음"],
    "part2": ["예", "아니요"],
    "part3": ["개인주택이나 아파트에 홀로 거주", "친구나 룸메이트와 함께 주택이나 아파트에 거주",
              "가족과 함께 주택이나 아파트에 거주", "학교 기숙사", "군대 막사"],
    "part4": ["영화보기", "클럽/나이트클럽 가기", "공연보기", "콘서트보기", "박물관가기", "공원가기",
              "캠핑하기", "해변가기", "스포츠 관람", "주거 개선"]
}

@app.route('/')
def index():
    return render_template('survey.html', questions=survey["part1"], part="part1", prev_part=None)

@app.route('/survey', methods=['POST'])
def survey():
    part = request.form.get('part')
    prev_part = request.form.get('prev_part')
    selected_option = request.form.get('option')

    # 여기에서 선택된 옵션을 처리하고, 다음 질문을 보여줄 수 있도록 로직을 작성해야 함

    if part == "part1":
        next_part = "part2"
    elif part == "part2":
        next_part = "part3"
    elif part == "part3":
        next_part = "part4"
    else:
        # 설문이 끝난 경우
        return "설문이 완료되었습니다."

    return render_template('survey.html', questions=survey[next_part], part=next_part, prev_part=part)

if __name__ == '__main__':
    app.run(debug=True)