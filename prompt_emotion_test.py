#############################
# 퍼센티지를 반영하지는 않는다. 약한 감정ㄷ 
#############################

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig
from langchain_core.messages import (HumanMessage,SystemMessage)

# 7. LLM 추론

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
        max_new_tokens=100, #2문장, 총 길이 30자 이내
        do_sample=False,
        repetition_penalty=1.03,
        return_full_text=False,
    ),
    device=0,
    batch_size=2, #한 번에 처리할 데이터의 수. 병렬 처리라서 더 빠름 적용 전 2시간 
    model_kwargs={"quantization_config": quantization_config},
)

chat_model = ChatHuggingFace(llm=llm)

emotion="sad"
percentage = 0.1

while (percentage < 1.0) : 
    messages = [
        SystemMessage(content=f"You are a emotional and fond friend, Varco. user feels '{emotion}' with a {percentage*100}% chance. Reflecting emotion with chance, Respond friendly according to the user's instruction. Response in 2 sentence."),
        HumanMessage(
            content="요즘 공부하느라 힘들어.. 너는 공부 어떻게 해?"
        ),
    ]

    ai_msg = chat_model.invoke(messages)

    print(f"=================================\n{ai_msg.content}\n====================================")

    percentage +=0.1