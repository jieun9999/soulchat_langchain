import os
import json
import csv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig
from langchain_core.messages import HumanMessage, SystemMessage
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
# 2. JSON 파일 경로 리스트 정의
###############################################################################################

# JSON 파일 경로 리스트
json_file_paths = [
    "/workspace/hdd/5.empatic_conversation/TL_기쁨_연인/Empathy_기쁨_연인_1.json",
    "/workspace/hdd/5.empatic_conversation/TL_당황_연인/Empathy_당황_연인_1.json",
    "/workspace/hdd/5.empatic_conversation/TL_분노_연인/Empathy_분노_연인_1.json",
    "/workspace/hdd/5.empatic_conversation/TL_불안_연인/Empathy_불안_연인_1.json",
    "/workspace/hdd/5.empatic_conversation/TL_상처_연인/Empathy_상처_연인_1.json",
    "/workspace/hdd/5.empatic_conversation/TL_슬픔_연인/Empathy_슬픔_연인_1.json"
]

# 사용자 쿼리 배열 (각 감정에 대응)
queries = [
    "우리 고양이에게 딱 맞는 간식을 찾았어!",  # 기쁨
    "자기야. 나 오늘 부끄러운 일 있었어.",  # 당황
    "세탁기 때문에 내 통장이 완전 텅 비어버렸어",  # 분노
    "나 자동차에 부딪힐 뻔했어... 지금 생각하면 아찔해.",  # 불안
    "상사가 일 못한다고 대뜸 소리를 지르더라",  # 상처
    "청바지를 새로 샀는데, 다리가 짧아보여. 망했어."  # 슬픔
]

# CSV 파일 저장 경로
output_csv_path = "/workspace/hdd/RAG/rag_context_responses.csv"

##############################################################################################
# 3. CSV 파일 생성 및 JSON 파일별 작업 수행
###############################################################################################

with open(output_csv_path, mode="w", encoding="utf-8", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Query", "Response"])  # 헤더 작성

    # JSON 파일별로 작업 수행
    for json_path, query in zip(json_file_paths, queries):
        # JSON 파일 로드
        with open(json_path, "r", encoding="utf-8") as file:
            empathy_data = json.load(file)

        # JSON 데이터에서 "utterances" 키 추출
        utterances = empathy_data.get("utterances", [])

        # 대화 패턴 추출
        conversation_patterns = []
        for utterance in utterances:
            role = utterance["role"]
            text = utterance["text"]
            conversation_patterns.append(f"[{role.capitalize()}] {text}")

        # 상황 설명 생성
        # 멀티라인 문자열의 불필요한 들여쓰기를 자동으로 제거
        # Automatically remove unnecessary indentation from the multiline string
        context_description = textwrap.dedent(f"""
            You are the user's partner (lover). Respond in a casual tone, using informal language as if speaking to a close partner or lover. 
            Ensure your responses are empathetic, comforting, and thoughtful, while maintaining the casual and intimate tone throughout the conversation.

            [Role]
            You take on the role of a **Listener**. The Listener listens attentively to the other person's words, empathizes, and provides comfort and advice. You must faithfully fulfill the role of the Listener in the conversation.

            [Caution]
            The user's utterance (instruction) may differ from the conversation patterns. Therefore, mimic the conversation patterns but do not assume it is the same situation.
            Based on the user's utterance (instruction), empathize appropriately, provide comfort, and offer advice.
            If the user displays self-deprecating or self-critical behavior, do not empathize with those attitudes. Instead, encourage and uplift them with positive and supportive responses.

            [Conversation Patterns]
        """)

        # 대화 패턴 추가
        for i, pattern in enumerate(conversation_patterns, 1):
            context_description += f"{i}. {pattern}\n"

        # LLM에 전달할 메시지
        messages = [
            SystemMessage(content=context_description),
            HumanMessage(content=query)
        ]

        # LLM 호출 및 응답 생성
        response = chat_model.invoke(input=messages)

        # 결과 출력 및 CSV 저장
        print(f"✅ 컨텍스트 : {context_description}")
        print(f"▶️ 응답 : {response.content}\n")

        # CSV 파일에 저장
        csv_writer.writerow([query, response.content])

print(f"모든 응답이 '{output_csv_path}'에 저장되었습니다.")
