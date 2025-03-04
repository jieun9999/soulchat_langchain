from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig
from langchain_core.messages import (HumanMessage,SystemMessage)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import pprint
##############################################################################################
# 랭체인(LangChain)을 활용하여 허깅페이스(HuggingFace)에 배포된 사전학습 모델을 활용하여 LLM 체인을 구성
# NCSOFT의 Llama-VARCO-8B-Instruct모델은 단순히 텍스트를 입력받아 텍스트를 생성하는 일반적인 언어 모델이 아니라, ChatGPT스타일의 채팅 모델로 설계
# Hugging Face의 pipeline API나 LangChain의 HuggingFacePipeline만으로는 적절히 처리하기 어렵습니다. 
# 채팅 모델에는 ChatHuggingFace를 사용하는 것이 맞습니다.
###############################################################################################

## Retrieval(검색기능) 붙이기
# 1. 문서 로드
file_path = "/workspace/hdd/RAG/persona_250304.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

print(docs[9].page_content[:200])
pprint.pp(docs[9].metadata) # 0부터 9까지의 인덱스만 존재


# # 2. 문서 분할
# # RecursiveTextSplitter 설정
# # 페르소나 텍스트는 구조화된 문서로, 의미 단위를 유지하면서 자연스럽게 분할하는 것이 중요하기 때문
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=100,  # 각 청크의 최대 문자 수
#     chunk_overlap=10,  # 청크 간 중복 문자 수
#     separators=["\n\n", "\n", " "]  # 큰 단위부터 작은 단위로 분할
# )
# # 텍스트 분할
# chunks = text_splitter.split_text(text)

# # 분할된 텍스트 출력
# # for i, chunk in enumerate(chunks):
# #     print(f"Chunk {i + 1}:")
# #     print(chunk)
# #     print("-" * 50)

# # 3. 임베딩 모델 불러오기
# model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
# embeddings = HuggingFaceEmbeddings(model_name=model_name)

# # 4. 벡터스토어 생성 및 청크 저장
# vector_store = Chroma(
#     collection_name="nerd_boy_txt_collection",
#     embedding_function=embeddings,
#     persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
# )
# # 각 청크를 Document 형태로 변환
# # chunks : 텍스트를 작은 단위로 나눈 결과를 담고 있는 리스트
# # chunks 리스트의 각 텍스트 청크를 Document 객체로 변환합니다.
# docs = [Document(page_content=chunk) for chunk in chunks]
# vector_store.add_documents(docs)

# # 5. Retriever 생성
# query = "유저와의 관계?"
# results = vector_store.similarity_search(
#     query=query,
#     k=1
# )
# # 검색 결과 출력
# print("검색 결과:")
# for res in results:
#     print(f"* {res.page_content}")

# 6. 


# ## 7. LLM 추론
# # hf_access_token.txt 파일에서 토큰 읽기
# with open('/workspace/hdd/RAG/hf_access_token.txt', 'r') as file:
#     hf_token = file.read().strip()  # 파일 내용 읽고 양쪽 공백 제거

# # 환경 변수에서 토큰 가져오기
# api_key = os.getenv(hf_token)

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