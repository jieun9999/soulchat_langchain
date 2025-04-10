from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig
from langchain_core.messages import (HumanMessage,SystemMessage)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import pprint
import re
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langsmith import traceable
import pandas as pd
from sentence_transformers import util
##############################################################################################
# 청킹 방법: 먼저 제목을 기준으로 큰 단위로 분할한 뒤, 각 섹션 내에서 RecursiveCharacterTextSplitter를 사용해 청킹
# 임베딩 모델 : open ai의 text-embedding-3-small + CacheBackedEmbeddings(캐시용)
# 벡터 스토어 : Chromadb 
# Retriever : 제목에 가중치를 부여하는 방식
# LLM : NCSOFT/Llama-VARCO-8B-Instruct
###############################################################################################


# 1. 문서 로드
file_path = "/workspace/hdd/LangChain/6categories.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load() #PDF의 각 페이지를 독립적으로 처리

# 모든 페이지의 텍스트를 하나로 합침 (페이지 번호 제거 적용)
all_text = "\n".join([doc.page_content for doc in docs])

# 2. 문서 분할
# 제목 리스트 정의
titles = [
    "캐릭터의 주관적인 의견을 묻는 질문", 
    "캐릭터의 주관적인 경험을 묻는 질문", 
    "상대방의 말에 대한 반응을 표현",
    "사실중심적인 경험을 공유",
    "의견을 표현",
    "감정중심적인 의견을 표현"
]

def split_into_sections(text, titles):
    """
    제목 리스트를 기준으로 텍스트를 섹션별로 분할.
    """
    sections = {}
    current_title = None
    current_content = []

    # 텍스트를 줄 단위로 순회
    for line in text.split("\n"):
        line = line.strip()  # 줄 양쪽 공백 제거
        if line in titles:  # 새로운 제목 발견 시
            if current_title:  # 현재 섹션 저장
                sections[current_title] = "\n".join(current_content).strip()
            current_title = line  # 새로운 제목으로 갱신
            current_content = []  # 새로운 섹션 시작
        elif current_title:  # 현재 섹션에 내용 추가
            current_content.append(line)

    # 마지막 섹션 저장
    if current_title:
        sections[current_title] = "\n".join(current_content).strip()

    return sections

# RecursiveTextSplitter 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # 각 청크의 최대 문자 수
    chunk_overlap=100,  # 청크 간 중복 문자 수
    separators=["\n\n", "\n", " "]  # 큰 단위부터 작은 단위로 분할
)

# (1). 제목 리스트를 기준으로 섹션 분할
sections = split_into_sections(all_text, titles)

# (2). 각 섹션 내에서 RecursiveCharacterTextSplitter로 추가 분할
final_chunks = []
for title, section_content in sections.items():
    # 섹션 내용을 RecursiveCharacterTextSplitter로 분할
    section_chunks = text_splitter.split_text(section_content)
    
    # 제목과 내용을 하나의 청크로 묶기
    for idx, chunk in enumerate(section_chunks):
        if idx == 0:
            # 첫 번째 청크에는 제목 포함
            final_chunks.append(f"제목: {title}\n{chunk}")
        else:
            # 두 번째 청크부터는 제목 없이 내용만 추가
            final_chunks.append(chunk)

# 청킹된 결과를 txt 파일로 저장
output_file_path = "/workspace/hdd/LangChain/chunks2000_6categories.txt"

with open(output_file_path, "w", encoding="utf-8") as file:
    for i, chunk in enumerate(final_chunks):
        file.write(f"청크 {i + 1}:\n{chunk}\n\n")  # 청크 번호와 내용 저장

print(f"청킹된 결과가 {output_file_path}에 저장되었습니다.")

# 3. 임베딩 모델 불러오기
# .env 파일 로드 : .env 파일에 정의된 환경 변수를 자동으로 읽어서 현재 실행 중인 Python 프로세스의 환경 변수로 설정
load_dotenv()

# OpenAI의 "text-embedding-3-small" 모델을 사용하여 임베딩을 생성합니다.
openai_embedding = OpenAIEmbeddings(model="text-embedding-3-small")
# 캐시를 지원하는 임베딩 래퍼 생성
store = LocalFileStore("./cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    openai_embedding, store, namespace = openai_embedding.model
)

## 4. 유사도 계산함수 및 가중치 고려한 검색함수 로드
# 제목과 본문 유사도 계산 함수
def calculate_similarity(query, chunk, embedder, title_weight=0):
    title, content = chunk.split("\n", 1) if "\n" in chunk else (chunk, "") 
    # 청크(chunk)에서 제목과 본문을 분리
    title = title.replace("제목: ", "").strip()
    query_emb, title_emb, content_emb = embedder.embed_query(query), embedder.embed_query(title), embedder.embed_query(content) 
    # 쿼리(query), 제목(title), **본문(content)**을 임베딩 모델(embedder)을 사용해 각각 벡터로 변환
    title_sim, content_sim = util.cos_sim(query_emb, title_emb).item(), util.cos_sim(query_emb, content_emb).item()
    # 코사인 유사도를 계산:
    # query_emb와 title_emb 간의 유사도: title_sim.
    # query_emb와 content_emb 간의 유사도: content_sim.
    return (title_weight * title_sim) + content_sim, title_sim, content_sim

# 가중치를 고려한 검색 함수
def search_with_weight(vector_store, query, k=3, title_weight=0):
    results = vector_store.similarity_search_with_score(query=query, k=k)
    weighted_results = [
        {
            "chunk": result[0].page_content,
            "final_score": final_score,
            "title_sim": title_sim,
            "content_sim": content_sim
        }
        for result in results #results 리스트에서 하나씩 result를 가져옵니다.
            for final_score, title_sim, content_sim in [calculate_similarity(query, result[0].page_content, cached_embedder, title_weight)] #calculate_similarity 함수는 튜플 (final_score, title_sim, content_sim)을 반환합니다.[calculate_similarity(...)]는 리스트로 감싸져 있으므로, 여기서 반환된 튜플을 바로 반복문으로 풀어냅니다.
    ]   
    
    return sorted(weighted_results, key=lambda x: x["final_score"], reverse=True)[:k]

## 5. 벡터스토어 생성 또는 로드
def create_or_load_vectorstore(chunks, embedder, store_path="/workspace/hdd/chroma_db"):
    docs = [Document(page_content=chunk) for chunk in chunks]
    if os.path.exists(store_path):
        print("기존 벡터스토어를 로드합니다...")
        return Chroma(collection_name="persona_collection", embedding_function=embedder, persist_directory=store_path)
    print("벡터스토어를 생성합니다...")
    vector_store = Chroma(collection_name="persona_collection", embedding_function=embedder, persist_directory=store_path)
    vector_store.add_documents(docs)
    return vector_store

# 실행
vector_store = create_or_load_vectorstore(final_chunks, cached_embedder)

# 검색 수행
queries = [
  "그 소식을 듣고 하루 종일 우울하고 심란했어..",
  "그런 일이 생겼다니 정말 화가 나!!! 어떻게 그럴 수가 있지?",
  "내일 발표가 있어서 너무 긴장돼.ㅠㅠㅠㅠㅠ 잘할 수 있을지 걱정이야.",
  "그게 되겠냐?",
  "그래? 그럼 다음에는 무슨 일이 있었어?",
  "ㅋㅋㅋㅋㅋㅋ 그거 진짜 웃긴데!",
  "맞아 ㅎ 나도 그렇게 느꼈어.",
  "그거 진짜 신기하다! 나도 그런 경험 해보고 싶어.",
  "최근에 갔던 식당 중에 어디가 제일 좋았음?",
  "오늘 아침에 동네 카페에서 커피 한 잔 마시며 책을 읽었엉.",
  "그 책은 읽기 쉬운 편이더라.",
  "최근에 생긴 카페 가봤는데 나름 괜찮더라궁 ㅎㅎ ",
  "지난달에 새 직장을 시작했거든",
  "어제 퇴근하고 집에 가는 길에 비가 많이 내렸어", 
  "그 친구와 헤어지고 나서 한동안 너무 슬프더라.. 마음이 텅 빈 느낌이었어.", 
  #가벼운 의견 표현과 감정적인 의견표현의 구분이 잘 안되는 경향
]

results_list = []
for query in queries:
    results = search_with_weight(vector_store, query)
    for result in results:
        results_list.append({
            "쿼리": query,
            "최종 점수": result["final_score"],
            "제목 유사도 (가중치 0배)": result["title_sim"],
            "본문 유사도": result["content_sim"],
            "검색된 청크": result["chunk"]
        })

# 결과 저장
output_csv_path = "/workspace/hdd/LangChain/rag_script_weights_6categories.csv"
pd.DataFrame(results_list).to_csv(output_csv_path, index=False, encoding="utf-8-sig")
print(f"검색 결과가 CSV 파일로 저장되었습니다: {output_csv_path}")


# 6. 프롬프트 생성


# 7. LLM 추론

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype="float16",
#     bnb_4bit_use_double_quant=True,
# )

# llm = HuggingFacePipeline.from_model_id(
#     model_id="NCSOFT/Llama-VARCO-8B-Instruct",
#     task="text-generation",
#     pipeline_kwargs=dict(
#         max_new_tokens=512,
#         do_sample=False,
#         repetition_penalty=1.03,
#         return_full_text=False,
#     ),
#     model_kwargs={"quantization_config": quantization_config},
# )

# chat_model = ChatHuggingFace(llm=llm)

# messages = [
#     SystemMessage(content="You are a helpful assistant Varco. Respond accurately and diligently according to the user's instructions."),
#     HumanMessage(
#         content="안녕하세요"
#     ),
# ]

# ai_msg = chat_model.invoke(messages)

# print(ai_msg.content)