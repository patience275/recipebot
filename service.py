
import bentoml
from bentoml.io import JSON

from langchain_mistralai import ChatMistralAI
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableParallel

llm = ChatMistralAI(model="mistral-tiny")

memory=ConversationSummaryMemory(llm=llm, return_messages=True,memory_key='chat_history')

prompt=ChatPromptTemplate([
    ('system','you are an excellent chef. using the {ingredients} provided by the user, create a delicious meal. include a catchy name, utensils to use, and the steps to follow to create the meal you suggest'),
    MessagesPlaceholder(variable_name ='chat_history'),
     ('user','{ingredients}')
])

chain=RunnableParallel(
    chat_history=lambda x:memory.chat_memory.messages,
    ingredients=RunnablePassthrough() 
) | prompt | llm

def get_recipie(ingredients):
    response = chain.invoke({"ingredients": ingredients})
    return response.content

svc=bentoml.Service('recipie_bot-service')

@svc.apis(input=JSON(),output=JSON())
def recipie(requests):
  ingredients=requests.get('ingredients',"")
  result=get_recipie(ingredients)
  return{'recipie':result}

