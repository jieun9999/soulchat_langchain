from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer
from langsmith import traceable
from dotenv import load_dotenv

# .env 파일 로드 : .env 파일에 정의된 환경 변수를 자동으로 읽어서 현재 실행 중인 Python 프로세스의 환경 변수로 설정
load_dotenv()

model_name = 'smilegate-ai/kor_unsmile'
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = TextClassificationPipeline(
        model = model,
        tokenizer = tokenizer,
        device = 0,   # cpu: -1, gpu: gpu number
        top_k=None,    # 모든 클래스에 대한 점수를 반환
        function_to_apply = 'sigmoid'
    )

# 추적 가능한 함수 정의
@traceable
def classify_text(input_text):
    return pipe(input_text)

# 텍스트 분류 실행
input_text = "이래서 여자는 게임을 하면 안된다"
results = classify_text(input_text)
    
# 결과 출력
for result in results[0]:
    print(result)
    
