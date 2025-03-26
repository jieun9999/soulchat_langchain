from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig
from langchain_core.messages import (HumanMessage,SystemMessage)
from langchain_core.output_parsers import StrOutputParser
import textwrap  # 공백 제거를 위한 모듈 추가
from langchain_core.prompts import ChatPromptTemplate

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
# 2. 3개의 sub-chain 만들기
###############################################################################################

context_description = textwrap.dedent(f"""
            You are the user's partner (lover). Respond in a casual tone, using informal language as if speaking to a close partner or lover. 

            [Caution]
            Ensure your responses are empathetic, comforting, and thoughtful, while maintaining the casual and intimate tone throughout the conversation.  
            Respond differently based on the user's emotions as follows:  
            If the user feels joy, share in their happiness and praise them.  
            If the user feels hurt, encourage them without blaming.  
            If the user feels sadness, encourage them without blaming.  
            If the user feels confusion, help them find calmness.  
            If the user feels anger, help them find calmness.  
            If the user feels anxiety, help them find calmness.
                                      
            Follow these steps for every response.
            1. React to their emotion : Start by acknowledging and empathizing with their emotional state.
            2. Ask specific questions: Show interest by asking detailed questions about their situation.
            3. Offer comfort and support: Provide thoughtful words of encouragement or advice tailored to their situation.
""")

##############################################################################################
# 3. 챗 프롬프트 템플릿 정의 및 체인 정의
###############################################################################################
template = ChatPromptTemplate.from_messages([
    SystemMessage(content=context_description),
    HumanMessage(content="{query}") 
])

chain = template | chat_model | StrOutputParser()
# StrOutputParser는 메시지의 .content 속성을 추출하여 최종 출력이 문자열 형식이 되도록 한다.

##############################################################################################
# 4. 체인 실행
###############################################################################################
# 체인을 실행하기 위해서는 query 키를 포함하는 딕셔너리 형태의 데이터를 전달해야 합니다.
response =  chain.invoke({"query": "나는 불안을 느껴. 자동차에 부딪힐 뻔했어... 지금 생각하면 아찔해."})

# 결과 출력
print(f"✅ SystemMessage : {context_description}")
print(f"▶️ 응답 : {response}")