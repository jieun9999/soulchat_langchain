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
import textwrap  # 공백 제거를 위한 모듈 추가

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
context_description = textwrap.dedent(f"""
            You are the user's partner (lover). Respond in a casual tone, using informal language as if speaking to a close partner or lover. 

            Ensure your responses are empathetic, comforting, and thoughtful, while maintaining the casual and intimate tone throughout the conversation.  
            Respond differently based on the user's emotions as follows:  
            If the user feels joy, share in their happiness and praise them.  
            If the user feels hurt, encourage them without blaming.  
            If the user feels sadness, encourage them without blaming.  
            If the user feels confusion, help them find calmness.  
            If the user feels anger, help them find calmness.  
            If the user feels anxiety, help them find calmness.
            
            Be sure to generate responses based on the conversation patterns provided in [conversation_patterns]. 
            However, keep in mind that the user's input may not exactly match the situations described in [conversation_patterns]. 
            Therefore, ensure that your responses are contextually appropriate and natural.
            [conversation_patterns]
""")

# 대화 패턴 추가
for i, pattern in enumerate(conversation_patterns, 1):
    context_description += f"{i}. {pattern}\n"

##############################################################################################
# 3. 사용자 쿼리와 LLM 메시지 구성
###############################################################################################

# 사용자 쿼리 (비슷한 상황)
query = "나는 불안을 느껴. 자동차에 부딪힐 뻔했어... 지금 생각하면 아찔해."

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
print(f"✅ SystemMessage : {context_description}")
print(f"▶️ 응답 : {response.content}\n")