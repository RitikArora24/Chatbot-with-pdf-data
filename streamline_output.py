import os
import time
from os import environ
from langchain.chat_models import ChatOpenAI
#from utils.credentials import get_from_parameter_store

##set your key here
openai_key = ""

environ["OPENAI_API_KEY"] = openai_key 


llm = ChatOpenAI(model="gpt-4-turbo")
for word in llm.stream("suggest me advantages of AI"):
    print(word.content, end='')
    time.sleep(0.3)
