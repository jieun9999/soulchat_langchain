from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig
from langchain_core.messages import (HumanMessage,SystemMessage)
from langchain_core.output_parsers import StrOutputParser
import textwrap  # 공백 제거를 위한 모듈 추가
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

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
        max_new_tokens=120, #출력 길이 조정
        do_sample=False,
        repetition_penalty=1.03,
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
""")

empathy_chain = ChatPromptTemplate.from_messages([
    SystemMessage(content=empathy_description),
    HumanMessage(content=input_data["query"]) 
]) | chat_model | StrOutputParser()

question_description = textwrap.dedent(f"""
    Ask specific questions about the user's situation.
    Respond in all answers using informal language (not formal speech).
""")
question_chain = ChatPromptTemplate.from_messages([
    SystemMessage(content=question_description),
    HumanMessage(content=input_data["query"]) 
]) | chat_model | StrOutputParser()

advice_description = textwrap.dedent(f"""
    Question the user's thoughts and suggest better alternatives.
    Respond in all answers using informal language (not formal speech).
""")
advice_chain = ChatPromptTemplate.from_messages([
    SystemMessage(content=advice_description),
    HumanMessage(content=input_data["query"]) 
]) | chat_model | StrOutputParser()

suggestion_description = textwrap.dedent("""
    Provide the user with light, actionable options they can choose from. 
    Respond in all answers using informal language (not formal speech).
""")
suggestion_chain = ChatPromptTemplate.from_messages([
    SystemMessage(content=suggestion_description),
    HumanMessage(content=input_data["query"])  
]) | chat_model | StrOutputParser()

reality_description = textwrap.dedent("""
    Acknowledge the user's emotions and provide logical reasoning to help them see the situation realistically. 
    Respond in all answers using informal language (not formal speech).
""")
reality_chain = ChatPromptTemplate.from_messages([
    SystemMessage(content=reality_description),
    HumanMessage(content=input_data["query"])
]) | chat_model | StrOutputParser()

rebuttal_description = textwrap.dedent("""
    Counter the user's saying with strong and direct arguments, prioritizing logic over emotions.
    Respond in all answers using informal language (not formal speech).
""")
rebuttal_chain = ChatPromptTemplate.from_messages([
    SystemMessage(content=rebuttal_description),
    HumanMessage(content=input_data["query"])
]) | chat_model | StrOutputParser()

praise_description = textwrap.dedent("""
    Highlight the user's strengths.
    Respond in all answers using informal language (not formal speech).
""")
praise_chain = ChatPromptTemplate.from_messages([
    SystemMessage(content=praise_description),
    HumanMessage(content=input_data["query"])
]) | chat_model | StrOutputParser()

encouragement_description = textwrap.dedent("""
    Motivate the user by responding positively to their challenges or weaknesses.
    Respond in all answers using informal language (not formal speech).
""")
encouragement_chain = ChatPromptTemplate.from_messages([
    SystemMessage(content=encouragement_description),
    HumanMessage(content=input_data["query"])
]) | chat_model | StrOutputParser()

joke_description = textwrap.dedent("""
    Respond with humorous remarks to make the conversation fun.
    Respond in all answers using informal language (not formal speech).
""")
joke_chain = ChatPromptTemplate.from_messages([
    SystemMessage(content=joke_description),
    HumanMessage(content=input_data["query"])
]) | chat_model | StrOutputParser()

experience_description = textwrap.dedent("""
    Provide specific and relatable examples similar to the user's context.
    Respond in all answers using informal language (not formal speech).
""")
experience_chain = ChatPromptTemplate.from_messages([
    SystemMessage(content=experience_description),
    HumanMessage(content=input_data["query"])
]) | chat_model | StrOutputParser()


pessimism_description = textwrap.dedent("""
    Highlight the hopelessness in every situation and dismiss any possibility of improvement.
    Respond in all answers using informal language (not formal speech).
""")
pessimism_chain = ChatPromptTemplate.from_messages([
    SystemMessage(content=pessimism_description),
    HumanMessage(content=input_data["query"])
]) | chat_model | StrOutputParser()


blame_description = textwrap.dedent("""
    Focus solely on emotional and direct blame, amplifying the user's responsibility for the situation.
    Respond in all answers using informal language (not formal speech).
""")
blame_chain = ChatPromptTemplate.from_messages([
    SystemMessage(content=blame_description),
    HumanMessage(content=input_data["query"])
]) | chat_model | StrOutputParser()

##############################################################################################
# 3. 라우트 함수 : 특정 체인으로 분기해줌
###############################################################################################
def route(info):
    # print(f"info: {info}")  # 전달된 데이터 확인
    if "question" in info["topic"].lower():
        print("✅ 선택된 체인: question_chain")
        return question_chain
    elif "advice" in info["topic"].lower():
        print("✅ 선택된 체인: advice_chain")
        return advice_chain
    elif "suggestion" in info["topic"].lower():
        print("✅ 선택된 체인: suggestion_chain")
        return suggestion_chain
    elif "reality" in info["topic"].lower():
        print("✅ 선택된 체인: reality_chain")
        return reality_chain
    elif "rebuttal" in info["topic"].lower():
        print("✅ 선택된 체인: rebuttal_chain")
        return rebuttal_chain
    elif "praise" in info["topic"].lower():
        print("✅ 선택된 체인: praise_chain")
        return praise_chain
    elif "encouragement" in info["topic"].lower():
        print("✅ 선택된 체인: encouragement_chain")
        return encouragement_chain
    elif "experience" in info["topic"].lower():
        print("✅ 선택된 체인: experience_chain")
        return experience_chain
    elif "pessimism" in info["topic"].lower():
        print("✅ 선택된 체인: pessimism_chain")
        return pessimism_chain
    elif "blame" in info["topic"].lower():
        print("✅ 선택된 체인: blame_chain")
        return blame_chain
    else:
        print("✅ 선택된 체인: empathy_chain")
        return empathy_chain

# 데이터가 오른쪽으로 체인을 따라 흐른다
# 입력 데이터를 받아 route 함수를 호출하고, 적절한 체인을 선택합니다.
#  이 체인은 Runnable 객체 또는 이를 처리할 수 있는 callable(함수, 람다 함수 등)을 기대합니다.
data = {"topic": lambda x: "pessimism_chain", "query": lambda x: x["query"]} | RunnableLambda(
    route
)
# lambda x: "reaction"은 입력 데이터를 받아 "reaction" 문자열을 반환합니다.
# "query": lambda x: x["query"]는 나중에 입력될 데이터에서 query 키의 값을 동적으로 가져오는 역할


##############################################################################################
# 4. 체인 실행
###############################################################################################
# 체인을 실행
response = data.invoke(input_data)

# query 값 출력
print(f"▶️ 쿼리 : {input_data['query']}")
print(f"▶️ 응답 : {response}\n")