from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig
from langchain_core.messages import (HumanMessage,SystemMessage)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_community.document_loaders import PyPDFLoader
import re
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain import hub as prompts
from langsmith import traceable
import pandas as pd
from sentence_transformers import util
import json

##############################################################################################
# 1. LLM 설정: NCSOFT/Llama-VARCO-8B-Instruct
###############################################################################################

# LLM 추론
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
        max_new_tokens=100,  # 2문장, 총 길이 30자 이내
        do_sample=False,
        repetition_penalty=1.03,
        return_full_text=False,
    ),
    device=0,
    model_kwargs={"quantization_config": quantization_config},
)

chat_model = ChatHuggingFace(llm=llm)

##############################################################################################
# 2. JSON 파일 로드 및 상황 정보 추출
###############################################################################################

# JSON 파일 로드
json_file_path = "/workspace/hdd/5.empatic_conversation/TL_불안_연인/Empathy_불안_연인_1.json"
with open(json_file_path, "r", encoding="utf-8") as file:
    empathy_data = json.load(file)

# JSON 데이터에서 "utterances" 키 추출
utterances = empathy_data.get("utterances", [])

# 전체 대화 패턴 추출
conversation_patterns = []
for utterance in utterances:
    role = utterance["role"]
    text = utterance["text"]
    conversation_patterns.append(f"[{role.capitalize()}] {text}")

# 상황 설명 생성
context_description = f"""
당신은 유저와 연인(lover)입니다. 따라서, 당신의 목표는 주어진 사용자(User)의 발화(instruction)에 대해 공감하고,다정하고 사려깊은 말투로 적절한 조언을 제공하는 것입니다.

[역할]
당신은 **Listener**의 역할을 맡고 있습니다. Listener는 상대방의 말을 경청하고, 공감하며, 위로와 조언을 제공합니다. 대화에서 Listener의 역할을 충실히 수행해야 합니다.

[주의]
유저의 쿼리는 대화 패턴과 다를 수 있습니다. 따라서, 대화 패턴을 모방하는 것이지 동일한 상황이라고 생각하면 안 됩니다. 
유저의 쿼리를 기반으로 적절히 공감하고 위로하며, 조언을 제공하세요.

[대화 패턴]
"""
# 대화 패턴 추가
for i, pattern in enumerate(conversation_patterns, 1):
    context_description += f"{i}. {pattern}\n"

##############################################################################################
# 3. 사용자 쿼리와 LLM 메시지 구성
###############################################################################################

# 사용자 쿼리 (비슷한 상황)
query = "나 자동차에 부딪힐뻔 했어... 지금 생각하면 아찔해"

# LLM에 전달할 메시지
messages = [
    SystemMessage(
        content=context_description
    ),
    HumanMessage(
        content=query
    )
]

##############################################################################################
# 4. LLM 호출 및 응답 생성
###############################################################################################

response = chat_model.invoke(input=messages)

# 결과 출력
print(f"✅ 컨텍스트 : {context_description}")
print(f"▶️ 응답 : {response.content}\n")