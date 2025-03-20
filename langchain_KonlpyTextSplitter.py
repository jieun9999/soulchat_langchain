from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig
from langchain_core.messages import (HumanMessage,SystemMessage)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import re
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import KonlpyTextSplitter
from konlpy.tag import Kkma

##############################################################################################
# 청킹 방법: KonlpyTextSplitter 사용
###############################################################################################

# Konlpy는 내부적으로 Java 기반의 한국어 형태소 분석기를 사용합니다(Kkma, Hannanum 등). 따라서 JVM이 필요하며, 이를 위해 Java 설치 및 환경 변수 설정이 필요합니다.
kkma = Kkma()
print(kkma.morphs("테스트 문장을 분석합니다."))

## Retrieval(검색기능) 붙이기
# 1. 문서 로드
file_path = "/workspace/hdd/RAG/persona_250304.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load() #PDF의 각 페이지를 독립적으로 처리
docs = docs[1:]  # 첫 번째 페이지를 제외

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
text_splitter = KonlpyTextSplitter(chunk_size=500, chunk_overlap=100)
# SemanticChunker를 사용하여 청킹
final_chunks = text_splitter.split_text(all_text)

# 청킹된 결과를 txt 파일로 저장
output_file_path = "/workspace/hdd/RAG/chunks_KonlpyTextSplitter.txt"

with open(output_file_path, "w", encoding="utf-8") as file:
    for i, chunk in enumerate(final_chunks):
        file.write(f"청크 {i + 1}:\n{chunk}\n\n")  # 청크 번호와 내용 저장

print(f"청킹된 결과가 {output_file_path}에 저장되었습니다.")
