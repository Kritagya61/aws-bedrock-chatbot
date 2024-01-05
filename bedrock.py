import os
import boto3
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

AWS_ACCESS_KEY = "ASIAYZ7GE5SNFKNIJ7RH"
AWS_SECRET_ACCESS_KEY = "BPIutYG2xj9YHSvpWcP6L6rqGHUNsfFOC5dlsMOZ"
AWS_SESSION_TOKEN = "IQoJb3JpZ2luX2VjEN///////////wEaCmFwLXNvdXRoLTEiRjBEAiB28GBXIiYFIeHpDQJ0WN/xrL/fLkP5xei4zIlMMaQ1LwIgVO6+6gHfcL4nVwjJg6SdDtCDV/2dbFe3mcHpxhYkV4oqtgMIeBAEGgw2MDU1MzYxODU0OTgiDO9X6MBilNGXv+gUKyqTA9e5arYiXEPImr0Xlo9TIwnbP32b4QTuCe3nv6TI4n1B+bv5D9jNl1KgoqnF1QDesil+KNiimRWZDOwd4T4GHtmbNrDOXorCOCkkoKZIs2dANA39sYDnZrfj3RtN0jpK4f8C4Vi6xB4Z0HmV3uuI9aDo3t+yLgSN7MQlC3pRTkJOT2tohyTuB04xbJGV0E+S5NOhFVFXLCa+2oyjx9HaNjbWwsYpYFB6GRsD6caSjkhL+o74bAFJluuKxt3LSC/2kW09/Ph2I8Ua/VoTu/w8DymxCNUzW59ORRARBI/btX+gefJn3sBILNqETnaGjEQETSPbas35BXfniX+lFjY8PdDE8NuUwp/fX3iUUBP8OOwpkuS469lHJgb5qpfSN7hHdtr5B02IhF9g58vQ41DiVVPM4XRcZHad1Eeq9zlo3XJFJ4U1FucRAVk5f/86kWowazbFsUTDfuhSITO5nN7164vyTdh3ty1Kp0JCYr6dcBkoQBIW6bq7+kCHQE2kRkE42akIqSWLbmniOyzjw+M3dgoNUHowvoPbrAY6pwEIy3HXgh+PT0HIeGFdCU3JNElLhRngKLNkfad2FUP1UJ0HwkD9tXA288UeT9/Z6ENo86/gb2VfiVP5WfZ2ZC3SFmnDbeAce21aLRssEtyNho+3rDny9/kq+2Ig6JKfXMNRYXEp+ofXPfXedyn9lphqgBXBcz7/g29yqlRZ3oBCXZeeSB4htaNzAdS7g18brdo52aRBMK8G8LrF4VPaR5cPzopeqLxrKg=="
AWS_REGION = "us-east-1"


# class BedrockLLM:

# @staticmethod
# def get_bedrock_client():
#     """
#     This function will return the bedrock client.
#     """
#     bedrock_client = boto3.client(
#         'bedrock',
#         region_name=AWS_REGION,
#         aws_access_key_id=AWS_ACCESS_KEY,
#         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#         aws_session_token=AWS_SESSION_TOKEN
#     )
#
#     return bedrock_client

# @staticmethod
def bedrock_chain():
    bedrock_client_2 = boto3.client(
        'bedrock-runtime',
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN
    )

    titan_llm = Bedrock(
        model_id="amazon.titan-text-express-v1", client=bedrock_client_2
    )
    titan_llm.model_kwargs = {"temperature": 0.5, "maxTokenCount": 700}

    prompt_template = """System: The following is a friendly conversation between a knowledgeable helpful assistant and a customer.
        The assistant works at the Hotel Paradise, a five-star hotel located in the heart of the city, with luxurious rooms, a spa, a pool, and a restaurant.
        The assistant is talkative and provides lots of specific details from it's context.

        Current conversation:
        {history}

        User: {input}
        Bot:"""
    PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=prompt_template
    )

    memory = ConversationBufferMemory(human_prefix="User", ai_prefix="Bot")
    conversation = ConversationChain(
        prompt=PROMPT,
        llm=titan_llm,
        verbose=True,
        memory=memory,
    )

    return conversation


# @staticmethod
def run_chain(chain, prompt):
    num_tokens = chain.llm.get_num_tokens(prompt)
    return chain({"input": prompt}), num_tokens


# @staticmethod
def clear_memory(chain):
    return chain.memory.clear()

# bedrock_client = BedrockLLM.get_bedrock_client()
# FM_list = bedrock_client.list_foundation_models()
# print(FM_list)
