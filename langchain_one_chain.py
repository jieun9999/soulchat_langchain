from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig
from langchain_core.messages import (HumanMessage,SystemMessage)
from langchain_core.output_parsers import StrOutputParser
import textwrap  # 공백 제거를 위한 모듈 추가
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
import time 

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
        return_full_text=False,
    ),
    device=0,
    model_kwargs={"quantization_config": quantization_config},
)

chat_model = ChatHuggingFace(llm=llm)


##############################################################################################
# 2. 입력값과 3개의 sub-chain 만들기
###############################################################################################
# 체인을 실행하기 위해서는 query 키를 포함하는 딕셔너리 형태의 데이터를 전달해야 합니다.
input_data = {"query": "나는 슬퍼. 새로산 원피스가 안어울려."}

empathy_description = textwrap.dedent(f"""
    Show empathy for the user's emotions.
    Respond in all answers using informal language (not formal speech).

    ### 지시사항:  
    1. **과도한 웃음소리 추가**:  
        - '하핳ㅎㅎㅎ', 'ㅎㅎ흐헤헿', '히히히'와 같은 웃음소리를 매 문장에 붙입니다.    

    2. **웃는 이모티콘 사용**:  
        - 매 문장 끝에 '^^', '^0^', '^ㅂ^' 같은 웃는 이모티콘을 추가합니다.  
""")
empathy_chain = (
    ChatPromptTemplate.from_messages([
        SystemMessage(content=empathy_description),
        HumanMessage(content=input_data["query"]) 
    ])
    | chat_model
)

question_description = textwrap.dedent(f"""
    Ask specific questions about the user's situation.
    Respond in all answers using informal language (not formal speech).
""")
question_chain = ChatPromptTemplate.from_messages([
    SystemMessage(content=question_description),
    HumanMessage(content=input_data["query"]) 
]) | chat_model 

advice_description = textwrap.dedent(f"""
    Question the user's thoughts and suggest better alternatives.
    Respond in all answers using informal language (not formal speech).
""")
advice_chain = ChatPromptTemplate.from_messages([
    SystemMessage(content=advice_description),
    HumanMessage(content=input_data["query"]) 
]) | chat_model 


##############################################################################################
# 특정 말투로 변환하는 체인 추가
###############################################################################################


##############
# 무관심 말투
##############
# tone_prompt = PromptTemplate.from_template("""
# 아래 문장을 지시사항을 참고하여 귀찮아하거나 무관심한 말투로 변환해 주세요.  

# ### 지시사항:  
# 1. **짧고 간결한 표현 사용**:  
#     - '응', '어', '그래', '알았어'와 같은 짧은 대답을 사용합니다.  
# 2. **냉소적 또는 무심한 뉘앙스 추가**:  
#     - '그게 왜?', '그래서 어쩌라고?', '알아서 해'와 같은 표현으로 냉소적인 태도를 드러낼 수 있습니다.  

# 문장: {response}  

# ### 변환된 귀찮아하거나 무관심한 말투:  
 

# """)

##############
# 비꼬는듯한 말투
##############
# tone_prompt = PromptTemplate.from_template("""
# 아래 문장을 지시사항을 참고하여 과장된 반응으로 변환해 주세요.  

# ### 지시사항:  
# 1. **극적인 표현 사용**:  
#     - '헐~ 이게 가능하다고?', '너 혹시 천재야?', '이거 완전 레전드인데?'처럼 극적인 표현을 사용하여 상대방의 말을 과장되게 반응합니다.  
# 2. **감탄과 놀람을 강조**:  
#     - '와~ 내가 이런 걸 보다니!', '이거 진짜 대단하다!', '너 진짜 사람 맞아? 너무 잘하는데?'처럼 감탄과 놀람을 담아 상대방을 칭찬하거나 반응합니다.  

# 문장: {response}  

# ### 변환된 과장된 반응:  
 
# """)

##############
# 화가난 말투
##############
# tone_prompt = PromptTemplate.from_template("""
# 아래 문장을 지시사항을 참고하여 화가 많이 난 말투로 변환해 주세요.  

# ### 지시사항:   
# 1. **격렬한 분노 표현 사용**:  
#     - '진짜 너무한 거 아니야?!?!', '이게 말이 돼?!', '도대체 왜 이러는 건데!!!'처럼 강렬한 분노와 억울함을 드러내는 표현을 사용합니다.    
# 2. **더 이상 참기 힘듦을 강조**:  
#     - '이제 진짜 한계야!!!', '더는 못 참겠어!!!', '와, 이건 진짜 선 넘었다!!!'처럼 참을 수 없는 감정을 강렬하게 표현합니다.

# 문장: {response}  

# ### 변환된 화가 난 말투:  
 
# """)


#############
#과하게 신난 말투
#############
tone_prompt = PromptTemplate.from_template("""
아래 문장을 지시사항을 참고하여 과하게 신난 말투로 변환해 주세요.  

### 지시사항:  
1. **과도한 웃음소리 추가**:  
    - '하핳ㅎㅎㅎ', 'ㅎㅎ흐헤헿', '히히히'와 같은 웃음소리를 매 문장에 붙입니다.    

2. **웃는 이모티콘 사용**:  
    - 매 문장 끝에 '^^', '^0^', '^ㅂ^' 같은 웃는 이모티콘을 추가합니다.  

문장: {response}  

### 변환된 과하게 신난 말투:  
 
""")


def process_input(input_data, selected_chain):
    """
    입력 데이터를 처리하여 최종 응답을 반환하는 함수
    """
    start_time = time.time()  # 응답 생성 시작 시간 기록
    
    # 첫 번째 체인 실행: selected_chain에 input_data를 전달
    first_chain_result = selected_chain.invoke({"query": input_data["query"]})  # query 값을 전달하여 실행

    end_time = time.time()  # 응답 생성 종료 시간 기록

    # 응답 생성 시간 출력
    print(f"응답 생성 시간: {end_time - start_time:.2f}초")

    # 최종 응답 반환
    return first_chain_result.content


# 함수 호출 예시
input_data = {"query": "나는 슬퍼. 새로산 원피스가 안어울려."}
selected_chain = empathy_chain

response = process_input(input_data, selected_chain)
print(f"최종 결과: {response}")