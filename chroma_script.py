import chromadb
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# ChromaDB 클라이언트 생성
chroma_client = chromadb.PersistentClient(path="/workspace/chroma_data")
# PersistentClient는 로컬에서 데이터베이스를 저장하고 로드 
# 그 외에 메모리에 저장하는 EphmeralClient, 네트워크를 통해서 접속하는 HttpClient가 있습니다. 실제 서비스에서는 HttpClient가 권장됩니다.

# 컬렉션 생성
# 컬렉션은 임베딩, 문서 및 추가 메타데이터를 저장하는 곳입니다.
# 새로운 컬렉션 생성
collection_name = "collection_bge-m3-korean"
collection = chroma_client.create_collection(
    name=collection_name
) 
# my_collections = 유클리드 유사도 사용
# my_collections2 = 유사도 계산 함수로 consine 유사도 사용. 
# collection_embedding1024 = 임베딩 차원 1024를 사용하는 컬렉션, jina-embeddings-v3

# 데이터 불러오기
df = pd.read_excel("/workspace/output/음식_싱글턴 대화 추출.xlsx")
df.sample(5)

# 임베딩 모델 불러오기
model = SentenceTransformer("upskyy/bge-m3-korean", trust_remote_code=True)
# ChromaDB의 내장 임베딩 모델이 아닌 한국어 임베딩 모델을 따로 불러와서 사용

# # 데이터 삽입하기
for index, row in tqdm(df.iterrows(), total=df.shape[0]): #엑셀 파일의 각 행을 반복 처리
    # 각 행의 질문과 답변을 가져오기
    question = row['User']
    answer = row['Answer']

    # 질문과 답변을 임베딩으로 변환
    question_embedding = model.encode(question)
    answer_embedding = model.encode(answer)

    # ChromaDB 컬렉션에 데이터 삽입
    collection.upsert(
        embeddings=[question_embedding, answer_embedding], # 질문과 답변의 임베딩
        documents=[question, answer], #원본 텍스트
        metadatas=[
            {"type" : "question"},
            {"type": "answer"}
        ], # 메타데이터
        ids=[f"question_{index}", f"answer_{index}"]  # 고유 ID
    )

print("데이터 삽입 완료!")
# print(f"컬렉션 내 벡터 개수: {collection.count()}") 항상 198로 동일

# 쿼리 실행하기
query_text = "점심에 뭐먹을까?" 
query_embedding = model.encode([query_text]) #, prompt="Retrieve semantically similar text: "

# Step 1. 질문과 유사한 질문 검색
question_results = collection.query(
    query_embeddings=question_embedding,
    where={"type": "question"},  # type이 "question"인 데이터만 검색
    n_results=1,  # 가장 유사한 질문 1개만 가져옴
)

print(question_results)

# print(question_results) 결과
# {'ids': [['question_98']], # query 메서드는 다중 결과를 지원, n_results=1로 설정했더라도 반환값은 [['question_98']]처럼 2차원 배열 형태로 반환
#  'embeddings': None, 
#  'documents': [['라면에 김치 빠지면 안 돼']], 
#  'uris': None, 'data': None, 
#  'metadatas': [[{'type': 'question'}]], 
#  'distances': [[0.0]], 
#  'included': [<IncludeEnum.distances: 'distances'>, <IncludeEnum.documents: 'documents'>, <IncludeEnum.metadatas: 'metadatas'>]}

# print(answer_results) 결과
# {'ids': ['answer_98'], 
#  'embeddings': None, 
#  'documents': ['오 좀 먹을 줄 아네'], 
#  'uris': None, 'data': None, 
#  'metadatas': [{'type': 'answer'}], 
#  'included': [<IncludeEnum.documents: 'documents'>, 
#               <IncludeEnum.metadatas: 'metadatas'>]}

# 응답 여러 개 두고, 하나만 고를 때 -> semantic_search로 2차적으로 상위 1개 결정
# final_hits = util.semantic_search(query_embedding, retrieved_embeddings, top_k=1)

# Step 2. 해당 질문의 ID를 기반으로 응답 검색
if question_results['ids']:
    matched_question_id = question_results['ids'][0][0] # 가장 유사한 질문의 ID
    answer_id = matched_question_id.replace("question_", "answer_") # 대응되는 응답 ID 추출

    # 응답 검색
    answer_results = collection.get(ids=[answer_id])  # ID로 응답 검색
    print("질문:", question_results['documents'][0][0])
    print("응답:", answer_results['documents'][0])

else:
    print("유사한 질문을 찾을 수 없습니다.")