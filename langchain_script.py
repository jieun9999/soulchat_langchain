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
##############################################################################################
# 청킹 방법: 먼저 제목을 기준으로 큰 단위로 분할한 뒤, 각 섹션 내에서 RecursiveCharacterTextSplitter를 사용해 청킹
# 임베딩 모델 : open ai의 text-embedding-3-small + CacheBackedEmbeddings(캐시용)
# 벡터 스토어 : Chromadb 
# Retriever :
# LLM : NCSOFT/Llama-VARCO-8B-Instruct
###############################################################################################


# 1. 문서 로드
file_path = "/workspace/hdd/RAG/persona_250304.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load() #PDF의 각 페이지를 독립적으로 처리
docs = docs[1:]  # 첫 번째 페이지를 제외

# print(docs[5].page_content)
# pprint.pp(docs[5].metadata) # 0부터 9까지의 인덱스만 존재

# 페이지 번호 제거 함수 정의
def remove_page_numbers(text):
    """
    텍스트에서 페이지 번호를 제거합니다.
    일반적으로 페이지 번호는 숫자만 있는 줄로 나타나므로 이를 제거합니다.
    """
    # 숫자만 있는 줄을 찾아 제거 (페이지 번호로 추정)
    return re.sub(r"^\d+$", "", text, flags=re.MULTILINE).strip()

# 모든 페이지의 텍스트를 하나로 합침 (페이지 번호 제거 적용)
all_text = "\n".join([remove_page_numbers(doc.page_content) for doc in docs])

# 2. 문서 분할
# 제목 리스트 정의
titles = [
    "외모", "직업과 능력", "싫어하는 것", "두려워하는 것",
    "연애와 인간관계", "어린 시절과 학창 시절", "지환과 유저의 관계",
    "취미생활", "이상형", "일상 루틴", "좋아하는 음식","싫어하는 음식", "가치관과 철학",
    "스트레스 해소 방법", "특이한 습관", "좋아하는 장소","소중히 여기는 물건"
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
    chunk_size=1000,  # 각 청크의 최대 문자 수
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
output_file_path = "/workspace/hdd/RAG/chunks1000_title_RecursiveCharacterTextSplitter.txt"

with open(output_file_path, "w", encoding="utf-8") as file:
    for i, chunk in enumerate(final_chunks):
        file.write(f"청크 {i + 1}:\n{chunk}\n\n")  # 청크 번호와 내용 저장

print(f"청킹된 결과가 {output_file_path}에 저장되었습니다.")

# # 3. 임베딩 모델 불러오기
# .env 파일 로드 : .env 파일에 정의된 환경 변수를 자동으로 읽어서 현재 실행 중인 Python 프로세스의 환경 변수로 설정
load_dotenv()

# OpenAI의 "text-embedding-3-small" 모델을 사용하여 임베딩을 생성합니다.
openai_embedding = OpenAIEmbeddings(model="text-embedding-3-small")
# 캐시를 지원하는 임베딩 래퍼 생성
store = LocalFileStore("./cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    openai_embedding, store, namespace = openai_embedding.model
)

# 4. 벡터스토어 생성 및 청크 저장
vector_store = Chroma(
    collection_name="persona_collection",
    embedding_function=cached_embedder,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
# 각 청크를 Document 형태로 변환
# final_chunks : 텍스트를 작은 단위로 나눈 결과를 담고 있는 리스트
# final_chunks 리스트의 각 텍스트 청크를 Document 객체로 변환합니다.
docs = [Document(page_content=chunk) for chunk in final_chunks]

# 각 청크에 대해 임베딩을 생성하여 캐시에 저장
for doc in docs:
    embedding = cached_embedder.embed_query(doc.page_content)  
# 이 과정에서:
# 캐시에 해당 텍스트의 임베딩이 존재하는지 확인합니다.
# 존재하면 캐시된 임베딩을 반환합니다.
# 존재하지 않으면 OpenAI API를 호출하여 새 임베딩을 생성하고, 이를 캐시에 저장합니다.

vector_store.add_documents(docs) # 벡터스토어(Vector Store)에 문서(docs)를 추가

@traceable
def perform_search(vector_store, query, k=3):
    """
    벡터 스토어에서 검색을 수행하고 결과를 반환합니다.
    """
    results = vector_store.similarity_search_with_score(query=query, k=k)
    return results

# 5. Retriever 생성
query = "지환이가 싫어하는 것은 무엇인가요?"
results = perform_search(vector_store, query)

# 6. 프롬프트 생성


# ## 7. LLM 추론
# # .env 파일 로드
# load_dotenv()

# # Hugging Face Access Token 가져오기
# hf_access_token = os.environ.get("HF_ACCESS_TOKEN")

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