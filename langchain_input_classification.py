from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig
from langchain_core.prompts import PromptTemplate

import get_speech_from_json

# CSV 파일 저장을 위해 csv 모듈을 임포트합니다.
import csv
import os # 파일 존재 여부 확인 등 (선택적)

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
        max_new_tokens=25,
        do_sample=False,
        repetition_penalty=1.03,
        return_full_text=False,
    ),
    device=0, # CUDA device 0 사용
    model_kwargs={"quantization_config": quantization_config},
)

##############################################################################################
# 2. 입력값과 1개의 chain 만들기
###############################################################################################

user_input_classification_prompt = PromptTemplate.from_template("""
다음은 사용자의 대화 입력입니다:
"{query}"

이 대화는 다음 9가지 카테고리 중 어디에 해당할까요?

- 부정적 경험 및 불만 표출 (예: "이런 서비스는 진짜 최악이에요. 왜 이렇게 대충 만들었죠?")
- 힘든 일, 위로가 필요한 상황 (예: "오늘 정말 힘든 하루였어요.")
- 좋은 소식 공유 및 축하받고 싶은 상황 (예: "저 취업했어요!")
- 예상치 못한 상황, 놀라운 경험 (예: "길에서 초등학교 때 선생님을 만났어요! 완전 깜짝 놀랐어요.")
- 경험 제시 (예: "요즘 너무 바빠서 잠을 거의 못 자요. 다들 이런가요?")
- 가벼운 리액션 및 피드백: 가벼운 리액션 및 피드백 (예: "아ㅋㅋ 진짜?")
- 유머와 가벼운 농담 (예: "고양이가 내 얼굴에 점프했어요. 주인인지 장애물인지 헷갈리나 봐요.")
- 문제 해결이 필요한 질문 (예: "이 문제 어떻게 해결해야 할까요?")
- 대화 시작 및 의견 요청 (예: "오늘 날씨가 정말 좋네요. 주말에 뭐 하실 계획 있으세요?")

정확한 카테고리 하나만 출력하세요. '부정적 경험 및 불만 표출'의 우선순위를 낮게 두세요.
3개의 카테고리를 선정 후, 최종적으로 하나만 선택하세요.
자신의 확신도를 0~100 사이의 숫자로 평가하세요.
출력 형식: [카테고리] (확신도: XX%)
""" )

##############################################################################################
# 3. 체인 구성 (나중 합칠 걸 위해)
###############################################################################################

sequential_chain = (
    user_input_classification_prompt  # 프롬프트 템플릿
    | llm                            # 언어 모델 호출
)

##############################################################################################
# 6. 체인 실행 및 CSV 저장
###############################################################################################

# 이전에 다운로드 받은 kakao dataset에서 문장을 가져와 테스트합니다.
target_directory = "/workspace/hdd/1.korean_SNS_multiTurn_conversation_data/3.openData/1.data/Training/2.labellingData/DailyTrend1.health_and_foodAndDrink"

# ★★★★★ CSV 저장을 위한 설정 시작 ★★★★★
output_csv_filename = "llm_classification_results.csv"
row_counter = 1 # CSV 파일의 '번호' 컬럼을 위한 카운터

# CSV 파일을 쓰기 모드('w')로 엽니다. 파일이 이미 있으면 덮어씁니다.
# newline='' 옵션은 CSV 파일에 불필요한 빈 줄이 생기는 것을 방지합니다.
# encoding='utf-8-sig'는 UTF-8 인코딩을 사용하되, Excel에서 한글 깨짐을 방지하기 위한 BOM을 추가합니다.
try:
    with open(output_csv_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        # CSV 작성을 위한 writer 객체를 생성합니다.
        csv_writer = csv.writer(csvfile)

        # CSV 파일의 헤더(첫 번째 행)를 작성합니다.
        header = ['번호', 'query', 'response']
        csv_writer.writerow(header)

        print(f"데이터 추출 및 분류 시작. 결과는 '{output_csv_filename}'에 저장됩니다...")
        # JSON 파일에서 텍스트 데이터를 가져옵니다.
        # extract_texts_from_directory 함수가 최대 500개(기본값) 또는 지정된 만큼의 텍스트를 반환합니다.
        limit = 500
        texts = get_speech_from_json.extract_texts_from_directory(target_directory, limit) # 여기서 최대 갯수 조절 가능

        if not texts:
             print(f"'{target_directory}'에서 추출된 텍스트가 없습니다.")
        else:
            # 가져온 텍스트 데이터에 대해 반복 작업을 수행합니다.
            for text in texts:
                # 체인을 실행하기 위한 입력 데이터를 준비합니다.
                input_data = {"query": text}

                # 체인을 실행하여 LLM의 응답(분류 결과)을 얻습니다.
                try:
                    response = sequential_chain.invoke(input_data)
                except Exception as e:
                    print(f"오류 발생: 쿼리 '{input_data['query']}' 처리 중 오류 - {e}")
                    response = "오류 발생" # 오류 발생 시 응답 필드에 표시

                # ★★★★★ CSV 파일에 데이터 행 작성 ★★★★★
                # 현재 번호, 원본 텍스트(query), LLM 응답(response)을 리스트로 묶어 CSV 파일에 씁니다.
                csv_writer.writerow([row_counter, input_data['query'], response])

                # 콘솔에도 진행 상황을 출력합니다 (기존 코드 유지).
                print(f"▶️ {input_data['query']} => {response}\n")

                # 다음 행을 위해 번호 카운터를 증가시킵니다.
                row_counter += 1

                # 만약 100개만 처리하고 싶다면 아래 코드를 활성화합니다.
                if row_counter > limit:
                    print(f"{limit}개 항목 처리 완료. 중단합니다.")
                    break

        print(f"총 {row_counter - 1}개의 항목 처리가 완료되어 '{output_csv_filename}'에 저장되었습니다.")

except IOError as e:
    print(f"오류: CSV 파일 '{output_csv_filename}'을(를) 쓰기 위해 열 수 없습니다. 권한을 확인하거나 파일이 다른 프로그램에서 열려있지 않은지 확인하세요. ({e})")
except Exception as e:
    print(f"스크립트 실행 중 예상치 못한 오류 발생: {e}")

# 결과 출력 (스크립트 종료 메시지)
print("\n스크립트 실행 완료.")