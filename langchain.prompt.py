from langchain import hub as prompts
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# 환경변수 로드 
load_dotenv()

# 프롬프트 정의
system = """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 문맥(context)을 사용하여 질문(question)에 대해 친근하고 짧은 문체로 답변하세요.  
만약, 주어진 문맥(context)에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다`라고 답하세요.  
한글로 답변해 주세요. 단, 너무 딱딱하거나 긴 설명은 피하고, 대화하듯 간단하게 답변하세요."""

human = """#Question: 
{question} 

#Context: 
{context} 

#Answer:"""

# 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
# langchain hub에 프롬트트 업로드
url = prompts.push("character_chat", prompt)

# url is a link to the prompt in the UI
print(url)

print("프롬프트가 LangChain Hub에 업로드되었습니다")