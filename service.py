

import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableParallel

from dotenv import load_dotenv
import os

load_dotenv()  
api_key = os.getenv("MISTRAL_API_KEY")

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

st.title("👨‍🍳 Your Personal Recipe Bot")
st.markdown("Enter a list of ingredients below, and I'll whip up a delicious recipe for you!")

# Input field for ingredients
ingredients_input = st.text_area(
    "Ingredients (e.g., chicken, basil, tomatoes, pasta)",
    height=100
)

# Button to trigger the recipe generation
if st.button("Generate Recipe"):
    if ingredients_input:
        # Show a loading spinner while processing
        with st.spinner('Cooking up a masterpiece...'):
            try:
                # Call your original function to get the recipe
                recipe = get_recipie(ingredients_input)
                
                # Display the generated recipe in a markdown block
                st.markdown("### Your Recipe")
                st.write(recipe)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Please ensure your MISTRAL_API_KEY is correctly configured.")
    else:
        st.error("Please enter some ingredients to generate a recipe.")



