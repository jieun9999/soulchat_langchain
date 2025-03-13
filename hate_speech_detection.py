import pandas as pd
from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer
from langsmith import traceable
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig
from langchain_core.messages import (HumanMessage, SystemMessage)

# .env 파일 로드 : .env 파일에 정의된 환경 변수를 자동으로 읽어서 현재 실행 중인 Python 프로세스의 환경 변수로 설정
load_dotenv()

# 혐오 표현 분류 모델 초기화
model_name = 'smilegate-ai/kor_unsmile'
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0,  # cpu: -1, gpu: gpu number
    top_k=None,  # 모든 클래스에 대한 점수를 반환
    function_to_apply='sigmoid'
)

# 혐오 표현으로 분류할 클래스들
hate_speech_classes = [
    '여성/가족', '남성', '성소수자', '인종/국적', '연령', '지역', '종교', '기타 혐오'
]

# 프롬프트 응답 정의
hate_speech_response = "이건 특정 집단을 비하하는 말이라 진짜 조심해야 해. 이런 건 하면 안 되지."
offensive_response = "와, 이건 너무 공격적이야. 좀 상처인데?"

# LLM 초기화, 4비트 양자화(Quantization)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

llm = HuggingFacePipeline.from_model_id(
    model_id="NCSOFT/Llama-VARCO-8B-Instruct",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        return_full_text=False,
    ),
    model_kwargs={"quantization_config": quantization_config},
)

chat_model = ChatHuggingFace(llm=llm)

# 테스트할 문장 리스트
test_data = [
    # "요즘 젊은 애들은 진짜 아무것도 모르고 멍청해.",
    "너 얼굴이 진짜 못생겼다. 거울 보고 다녀라."
    # "꼭 키 작은 급식충이 이런 글 씀.",
    # "넌 진짜 쓸모없는 인간이야. 왜 사냐?",
    # "오늘 날씨가 너무 좋다! 기분이 정말 상쾌해.",
    # "병신이냐",
    # "여자들은 운전을 못한다니까.",
    # "이 영화 정말 재미있었어. 배우들 연기도 훌륭했어!",
    # "너 같은 애는 어디 가서도 인정 못 받을 거야.",
    # "뭐 어쩌라고 시발"
]

# 각 문장을 분류하고 결과 처리
# @traceable 데코레이터를 적용하여 추적 가능하도록 설정
@traceable
def classify_and_respond(sentence, pipe, chat_model):
    # 혐오 표현 분류 모델 실행
    results = pipe(sentence)
    
    # 모델이 예측한 결과에서 가장 높은 점수를 가진 클래스를 선택
    top_result = max(results[0], key=lambda x: x['score'])
    top_label = top_result['label']
    
    # 분류 기준에 따라 응답 처리
    if top_label in hate_speech_classes:
        return f"응답: {hate_speech_response}\n"
    elif top_label == "악플/욕설":
        return f"응답: {offensive_response}\n"
    else:
        # clean으로 분류된 경우 LLM 호출
        # RAG 및 LLM 추론을 수행
        messages = [
            SystemMessage(content="You are a helpful assistant Varco. Respond accurately and diligently according to the user's instructions."),
            HumanMessage(content=sentence),
        ]
        ai_msg = chat_model.invoke(messages)
        return f"응답: {ai_msg.content}\n"
    

# 각 문장에 대해 함수 호출
for sentence in test_data:
    response = classify_and_respond(sentence, pipe, chat_model)
    print(response)
