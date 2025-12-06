from langchain_ollama import ChatOllama
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage)
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("OLLAMA_API_KEY")

client = ChatOllama(
   model="deepseek-v3.1:671b",
    base_url="https://ollama.com",
    api_key=api_key
)


chat_history = []

system_message = SystemMessage(content="You are a helpful AI Assistant.")
chat_history.append(system_message)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat. Goodbye!")
        break

    human_message = HumanMessage(content=user_input)
    chat_history.append(human_message)

    response = client.invoke(chat_history)
    ai_message = AIMessage(content=response.content)
    chat_history.append(ai_message)

    print(f"AI: {response.content}\n")