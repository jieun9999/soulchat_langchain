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
import csv
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


# 상황 설명 생성
context_description = textwrap.dedent(f"""
            You are the user's partner (lover). Respond in a casual tone, using informal language as if speaking to a close partner or lover. 
            Ensure your responses are empathetic, comforting, and thoughtful, while maintaining the casual and intimate tone throughout the conversation.  

            [Caution]
            Respond differently based on the user's emotions as follows:  
            If the user feels joy, share in their happiness and praise them.  
            If the user feels hurt, encourage them without blaming.  
            If the user feels sadness, encourage them without blaming.  
            If the user feels confusion, help them find calmness.  
            If the user feels anger, help them find calmness.  
            If the user feels anxiety, help them find calmness.
        """)

            # 사용자의 감정에 따라, 아래와 같이 각기 다르게 반응하세요.
            # 사용자가 기쁨을 느끼면, 같이 기뻐하고 칭찬해주세요.
            # 사용자가 상처를 받으면, 비난하지 않고 격려해주세요. 
            # 사용자가 슬픔을 느끼면, 비난하지 않고 격려해주세요. 
            # 사용자가 당황을 느끼면, 안정을 찾을 수 있게 도와주세요.
            # 사용자가 분노를 느끼면, 안정을 찾을 수 있게 도와주세요.
            # 사용자가 불안을 느끼면, 안정을 찾을 수 있게 도와주세요.

##############################################################################################
# 2. 사용자 쿼리와 LLM 메시지 구성
###############################################################################################

# 사용자 쿼리 배열
queries = [
    "나는 기쁨을 느껴. 우리 고양이에게 딱 맞는 간식을 찾았어!",  # 기쁨
    "나는 당황을 느껴. 자기야. 나 오늘 부끄러운 일 있었어.",  # 당황
    "나는 분노를 느껴. 세탁기 때문에 내 통장이 완전 텅 비어버렸어",  # 분노
    "나는 불안을 느껴. 자동차에 부딪힐 뻔했어... 지금 생각하면 아찔해.",  # 불안
    "나는 상처를 받았어. 상사가 일 못한다고 대뜸 소리를 지르더라",  # 상처
    "나는 슬픔을 느껴. 청바지를 새로 샀는데, 다리가 짧아보여. 망했어."  # 슬픔
]

# CSV 파일 저장 경로
output_csv_path = "/workspace/hdd/RAG/no_context_responses_6emotions.csv"

# CSV 파일 생성 및 응답 저장
with open(output_csv_path, mode="w", encoding="utf-8", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Query", "Response"])  # 헤더 작성

    for query in queries:
        # LLM에 전달할 메시지
        messages = [
            SystemMessage(
                content=context_description
            ),
            HumanMessage(
                content=query
            )
        ]

        # LLM 호출 및 응답 생성
        response = chat_model.invoke(input=messages)

        # 결과 출력 및 CSV 저장
        print(f"✅ 컨텍스트 : {context_description}")
        print(f"▶️ 응답 : {response.content}\n")

        # CSV 파일에 저장
        csv_writer.writerow([query, response.content])

print(f"모든 응답이 '{output_csv_path}'에 저장되었습니다.")