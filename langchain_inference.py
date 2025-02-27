from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig
from langchain_core.messages import (HumanMessage,SystemMessage)
import os
##############################################################################################
# 랭체인(LangChain)을 활용하여 허깅페이스(HuggingFace)에 배포된 사전학습 모델을 활용하여 LLM 체인을 구성
# NCSOFT의 Llama-VARCO-8B-Instruct모델은 단순히 텍스트를 입력받아 텍스트를 생성하는 일반적인 언어 모델이 아니라, ChatGPT스타일의 채팅 모델로 설계
# Hugging Face의 pipeline API나 LangChain의 HuggingFacePipeline만으로는 적절히 처리하기 어렵습니다. 
# 채팅 모델에는 ChatHuggingFace를 사용하는 것이 맞습니다.
###############################################################################################

# hf_access_token.txt 파일에서 토큰 읽기
with open('/workspace/hdd/kakao_data_and_code/hf_access_token.txt', 'r') as file:
    hf_token = file.read().strip()  # 파일 내용 읽고 양쪽 공백 제거

# 환경 변수에서 토큰 가져오기
api_key = os.getenv(hf_token)

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

messages = [
    SystemMessage(content="You are a helpful assistant Varco. Respond accurately and diligently according to the user's instructions."),
    HumanMessage(
        content="안녕하세요"
    ),
]

ai_msg = chat_model.invoke(messages)

print(ai_msg.content)