import os
import boto3
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

AWS_ACCESS_KEY = "ASIAYZ7GE5SNCBZAMZJ2"
AWS_SECRET_ACCESS_KEY = "efC3qov4Zey4ohe6512sYZTYBBToIo14g6KzgLgu"
AWS_SESSION_TOKEN="IQoJb3JpZ2luX2VjEPr//////////wEaCmFwLXNvdXRoLTEiRzBFAiEAhK2HlHArH1oK9BmhWRMFD4NEcTrLUINVRQbhTwyH7g0CIHTQZuXOXXKLW9X61nFjaDpWd34Hl4lWum4xR/lcZp2bKrcDCJP//////////wEQBBoMNjA1NTM2MTg1NDk4IgzQeiJ5mmYhr08N++kqiwN7t2kMJ1mLs8LSd48rGZfHvvD+co51wNndWEQQjXuUFGvpw0WJJGDrEL9868nptvLubmLHBEDwIr2u2tGYpMQureDjKIG681+AkGsB17Nyq+074Qy72xPMRr0Rmm2FuFIXk9Fknsqbgjf7C6Fs0Wm02X3ccuS80zS7Vd/EhbeE6KHRA6RoJvPXHz+j+bW74SCfANtLG0RouPdrc+G6oPv0kmKvFZfSetMxd5nDO/eLE2Koi8dsxSlZBmXZMxa7/mCeSRJTIfFBcqVGUbPhFkRaPPT5oxrBrpAXOxuZbeyV/MJ9gMmS3vkTQgRUHvorXQbWvohogtpR3zKcJPj0RO5NA0my/hK+Z8fcVpUt6msX6H7vjBUt6pXOf+jJMptzZzH1x0TXmVaQtXo9/qpEbf4FotihjYs6PhXQHhIHzT21eUpVvLTR/INgWcLnuqiy4VXQT1qmz+g8IOUJ1ALlKMaA+EiCUCGdSech+/M9aYvJS9yBnx/g3Xfrxl1xJOw6OHPpBLdSIPT2NWw8RTCpjuGsBjqmAZ5VkeFKVM2p7DSXdJEaOwovErcL71rtwQXN0H/7LizwJDvfSaTiV6i8MftMbrPM7kGp4dcS8wd2aVKU07FpzQ6+9Dng4yAvqtVi/iXFZDeYQyfat0E+szVJKH3GlP40fSerUwSze7AtfZUln+4blGhJVH/7d8ZiRTDdLmPkkQ7XOs7NCZBN86SxEESQnxbEJmSdCe4JkvyLF29c1tdSVXx8xfOvnKE="
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
