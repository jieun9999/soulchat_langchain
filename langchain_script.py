from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig
from langchain_core.messages import (HumanMessage,SystemMessage)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
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
from langchain import hub as prompts
from langsmith import traceable
import pandas as pd
from sentence_transformers import util
##############################################################################################
# 청킹 방법: 먼저 제목을 기준으로 큰 단위로 분할한 뒤, 각 섹션 내에서 RecursiveCharacterTextSplitter를 사용해 청킹
# 임베딩 모델 : open ai의 text-embedding-3-small + CacheBackedEmbeddings(캐시용)
# 벡터 스토어 : Chromadb 
# Retriever :
# LLM : NCSOFT/Llama-VARCO-8B-Instruct
###############################################################################################


# 1. 문서 로드
file_path = "/workspace/hdd/RAG/persona_250313.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load() #PDF의 각 페이지를 독립적으로 처리
# docs = docs[1:]  # 첫 번째 페이지를 제외

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
    "이름", "생일", "출생지", "거주지","가족 관계", "별자리", "MBTI", "외모",
    "평상시 성격 : 내향적", "평상시 성격 : 이성적", "연애 상대에게는 감성적인 성격","패션 스타일", "직업", "개발 능력", 
    "관심기술","공부방법: 알고리즘 문제풀이","공부방법: 서비스 개발","공부하는 장소 : 학교 도서관","공부하는 장소 : 조용한 카페",
    "일상적인 주말 아침", "일상적인 주말 점심", "일상적인 주말 저녁",
    "싫어하는 장소, 환경", "싫어하는 사람",
    "실패에 대한 두려움", "실패 대처 방법: 주위의 피드백 활용","예측 불가능한 상황에 대한 두려움",
    "연애경험","이성과의 대화 스타일", "인간관계","어린 시절 : 초등학교", "학창 시절 : 중학교", "학창 시절 : 고등학교",
    "지환과 유저의 첫만남", "유저에 대한 감정","학교 끝나고 유저와 데이트 : 카페에서 공부", "학교 끝나고 유저와 데이트 : 캠퍼스 산책 데이트",
    "취미생활 : 러닝", "취미생활 : 영화", "취미생활 : 음악",
    "좋아하는 스타일, 이상형",  
    "일상적인 아침", "일상적인 점심", "일상적인 저녁",
    "좋아하는 음식 : 김치볶음밥","좋아하는 음식 : 삼겹살",
    "싫어하는 음식 : 해산물", "싫어하는 음식 : 고수",
    "일에 대한 가치관", "기술에 대한 가치관", "인간관계에 대한 가치관",
    "스트레스 해소 방법 : 산책", "스트레스 해소 방법 : 초콜릿",
    "특이한 습관 : 손톱 뜯기",  "특이한 습관 : 책상 정리",
    "좋아하는 장소 : 공원 벤치","좋아하는 장소 : PC방",
    "소중히 여기는 물건 : 첫번째 노트북", "소중히 여기는 물건 : 손목시계", 
    "좋아하는 여행 스타일 : 자연 속 힐링", "좋아하는 여행지 : 강릉"
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

# 4. 백터스토어 생성 or 이미 존재하면 로드

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

# 벡터스토어 저장 경로
vector_store_path = "./chroma_langchain_db"

# 벡터스토어 로드 또는 생성
if os.path.exists(vector_store_path):
    print("기존 벡터스토어를 로드합니다...")
    vector_store = Chroma(
        collection_name="persona_collection",
        embedding_function=cached_embedder,
        persist_directory=vector_store_path,
    )
else:
    print("벡터스토어를 생성합니다...")
    vector_store = Chroma(
        collection_name="persona_collection3",
        embedding_function=cached_embedder,
        persist_directory=vector_store_path,  # 데이터를 로컬에 저장
    )
    vector_store.add_documents(docs)  # 벡터스토어(Vector Store)에 문서(docs)를 추가
    print("벡터스토어가 생성되었습니다.")

# @traceable
# def perform_search(vector_store, query, k=1):
#     """
#     벡터 스토어에서 검색을 수행하고 결과를 반환합니다.
#     """
#     result = vector_store.similarity_search_with_score(query=query, k=k)
#     return result

# 제목과 본문 유사도 계산 함수
# query: 사용자가 입력한 검색어(쿼리). 
# chunk: 제목과 본문이 포함된 텍스트 청크.
# embedder: 텍스트를 벡터로 변환하는 임베딩 모델.
# title_weight: 제목의 유사도에 부여할 가중치 (기본값은 1.5).
def calculate_similarity(query, chunk, embedder, title_weight=1.5):
    title, content = chunk.split("\n", 1) if "\n" in chunk else (chunk, "") # 청크(chunk)에서 제목과 본문을 분리
    title = title.replace("제목: ", "").strip()
    query_emb, title_emb, content_emb = embedder.embed_query(query), embedder.embed_query(title), embedder.embed_query(content) # 쿼리(query), 제목(title), **본문(content)**을 임베딩 모델(embedder)을 사용해 각각 벡터로 변환
    title_sim, content_sim = util.cos_sim(query_emb, title_emb).item(), util.cos_sim(query_emb, content_emb).item()
    # 코사인 유사도를 계산:
    # query_emb와 title_emb 간의 유사도: title_sim.
    # query_emb와 content_emb 간의 유사도: content_sim.
    return (title_weight * title_sim) + content_sim, title_sim, content_sim
    # 반환값:최종 유사도, 제목 유사도, 본문 유사도

@traceable
def search_with_weight(vector_store, query, k=1, title_weight=1.5):
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

# 5. Retriever 생성

def replace_pronouns(query, persona_name="서지환", user_name="유저"):
    # 1. 대명사 매핑
    pronoun_map = {
        # 1인칭 (사용자 본인)
        r"\b나는\b": f"{user_name}은",
        r"\b내가\b": f"{user_name}이",
        r"\b내\b": f"{user_name}의",
        r"\b나의\b": f"{user_name}의",
        r"\b날\b": f"{user_name}을",
        r"\b나를\b": f"{user_name}을",
        r"\b나에게\b": f"{user_name}에게",
        r"\b내게\b": f"{user_name}에게",
        r"\b나한테\b": f"{user_name}한테",
        r"\b나와\b": f"{user_name}과",
        r"\b나랑\b": f"{user_name}랑",

        # 2인칭 (페르소나)
        r"\b너는\b": f"{persona_name}은",
        r"\b네가\b": f"{persona_name}이",
        r"\b니가\b": f"{persona_name}이",
        r"\b너의\b": f"{persona_name}의",
        r"\b네\b": f"{persona_name}의",
        r"\b니\b": f"{persona_name}의",
        r"\b너를\b": f"{persona_name}을",
        r"\b너한테\b": f"{persona_name}한테",
        r"\b너에게\b": f"{persona_name}에게",
        r"\b너랑\b": f"{persona_name}랑",
        r"\b너와\b": f"{persona_name}와",
        r"\b너도\b": f"{persona_name}도",

        # 1인칭 복수 (우리)
        r"\b우리\b": f"{user_name}와 {persona_name}",
        r"\b우리는\b": f"{user_name}와 {persona_name}은",
        r"\b우리의\b": f"{user_name}와 {persona_name}의",
        r"\b우릴\b": f"{user_name}와 {persona_name}을",
        r"\b우리를\b": f"{user_name}와 {persona_name}을",
        r"\b우리에게\b": f"{user_name}와 {persona_name}에게",
        r"\b우리한테\b": f"{user_name}와 {persona_name}한테",
        r"\b우리랑\b": f"{user_name}와 {persona_name}랑",
        r"\b우리와\b": f"{user_name}와 {persona_name}와",
    }

    # 2. 정규식 적용
    for pattern, replacement in pronoun_map.items():
        query = re.sub(pattern, replacement, query)

    return query

queries = [
    "우리 처음 만났을 때 기억나?",
    "어떤 기술에 관심이 있어?",
    "다음에 카페 가서 공부하자! 어때?",
    "너는 주말에 보통 뭐 해?",
    "내가 고민이 있어.. 넌 스트레스 받을 때 어떻게 해?",
    "넌 어떤 스타일의 사람이 좋아?",
    "너 생일 언제야? 생일에 뭐 하고 싶어?",
    "요즘 공부하느라 힘들어.. 너는 공부 어떻게 해?",
    "배고프다.. 우리 뭐 먹을까?",
    "너는 실패하면 어떻게 해?",
    "너는 노트북 없이 하루 살 수 있어?",
    "우리 학교 끝나고 뭐 할까?",
    "너는 평소에 어떤 성격이야?",
    "우리 이번 주말에 영화 보러 갈래?",
    "너랑 있으면 편해! 너는 사람 만날 때 어때?",
    "너는 아침형 인간이야? 저녁형 인간이야?",
    "우리 같이 공부하면 집중 잘 될까?",
    "너는 대화할 때 어떤 스타일이야?",
    "넌 감성적인 편이야? 논리적인 편이야?",
    "너는 여행 가는 거 좋아해?"
]


# 응답 결과를 CSV 파일로 저장
output_csv_path = "/workspace/hdd/RAG/rag_results1.csv"
# 검색 결과를 저장할 리스트
# search_results = []

#rag 결과 저장 리스트
rag_results = []

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
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        return_full_text=False,
    ),
    model_kwargs={"quantization_config": quantization_config},
)

chat_model = ChatHuggingFace(llm=llm)

# LangSmith에서 저장된 프롬프트 가져오기
# 'prompt_id'는 LangSmith에서 프롬프트의 고유 식별자입니다
prompt_id = "character_chat4"
chain = prompts.pull(prompt_id)
print(f"❤️{chain}❤️")
prompt_template = ChatPromptTemplate.from_template(chain.from_template)


# # 프롬프트 데이터로부터 ChatPromptTemplate 생성
# prompt_template = ChatPromptTemplate.from_messages(prompt_data.content)


for query in queries:
    # 검색 수행
    query = replace_pronouns(query)
    result = search_with_weight(vector_store, query) #상위 문서 1개 반환
    # print(result)
    chunk = result[0]['chunk'] # 검색된 청크
    similarity_score = result[0]['final_score']  # 유사도 점수

    #llm 넣기
    response = llm.invoke(prompt_template.format(Initiation=query, Context=chunk))

    response = chat_model.invoke

    best_result = {
        "쿼리": query,
        "유사도 점수": similarity_score,
        "검색된 청크": chunk,
        "생성된 응답" : response
    }

    rag_results.append(best_result)
        

# 리스트를 DataFrame으로 변환
df = pd.DataFrame(rag_results)

# CSV 파일로 저장
df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

print(f"검색 결과가 CSV 파일로 저장되었습니다: {output_csv_path}")
# print("RAG에서 질문과 관련성이 가장 높은 청크 하나로 응답 생성성을 완료했습니다.")




