from fastapi import FastAPI 
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os
from datetime import datetime, timedelta
import uuid
from langdetect import detect

INACTIVITY_TIMEOUT = timedelta(minutes=5)

def remove_inactive_users():
    now = datetime.now()
    inactive_users = [user_id for user_id, details in connected_users.items()
                      if now - datetime.strptime(details["last_active"], "%Y-%m-%d %H:%M:%S") > INACTIVITY_TIMEOUT]
    for user_id in inactive_users:
        del connected_users[user_id]
        print(f"Removed inactive user: {user_id}")

def generate_user_id():
    return str(uuid.uuid4())

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str
    user_id : str

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "Embeddings")



# tool2
vectorstore2 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool2"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))
retriever2 = vectorstore2.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.7})
retriever_tool2 = create_retriever_tool(retriever=retriever2,                           
                                       name="MMUY_scheme1_SC_ST",
                                       description="Retrieves relevant information from stored documents summarizing all the information without missing any")

# Tool 1.3
vectorstore3 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool3"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))

retriever3 = vectorstore3.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.7})
retriever_tool3 = create_retriever_tool(retriever=retriever3,                           
                                       name="MMUY_scheme2_Extremely_Backward_Class",
                                       description="Retrieves relevant information from stored documents summarizing all the information without missing any")

# Tool 1.4
vectorstore4 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool4"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))

retriever4 = vectorstore4.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.7})
retriever_tool4 = create_retriever_tool(retriever=retriever4,                           
                                       name="MMUY_scheme3_YUVA",
                                       description="Retrieves relevant information from stored documents summarizing all the information without missing any")

# Tool 1.5
vectorstore5 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool5"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))

retriever5 = vectorstore5.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.7})
retriever_tool5 = create_retriever_tool(retriever=retriever5,                           
                                       name="MMUY_scheme4_Mahila",
                                       description="Retrieves relevant information from stored documents summarizing all the information without missing any")

# Tool 1.6
vectorstore6 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool6"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))

retriever6 = vectorstore6.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.7})
retriever_tool6 = create_retriever_tool(retriever=retriever6,                           
                                       name="MMUY_scheme5_Alpsankhyak",
                                       description="Retrieves relevant information from stored documents summarizing all the information without missing any")

# Tool 1.7 (MMUY)
vectorstore7 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool7"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))

retriever7 = vectorstore7.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.7})
retriever_tool7 = create_retriever_tool(retriever=retriever7,                           
                                       name="MMUY_Mukhyamantri_Udyami_Yojana",
                                       description="Retrieves relevant information from stored documents summarizing all the information without missing any, Also state the Sub-Schemes under this")


# Tool 1.8 (BLUY)
vectorstore8 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool8"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))

retriever8 = vectorstore8.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'lambda_mult': 0.7})
retriever_tool8 = create_retriever_tool(retriever=retriever8,                           
                                       name="BLUY_Bihar_Laghu_Udyami_Yojna",
                                       description="Retrieves relevant information from stored documents summarizing all the information without missing any")





# Direct Gemini Tool (tool 8)
chat = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
                              google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE")

@tool
def direct_llm_answer(query: str) -> str:
    """Directly generates an answer from the LLM and only relevant."""
    prompt = f"""
    You are an assistant that only answers queries about government schemes of Bihar, India.
    Do not answer anything unrelated to Bihar schemes. If a question is unrelated, politely inform the user.

    User question: {query}
    """
    response = chat.invoke(prompt)
    return response

tools = [retriever_tool2, retriever_tool3, retriever_tool4, retriever_tool5, retriever_tool6, retriever_tool7, retriever_tool8, direct_llm_answer]

chat_prompt_template = hub.pull("hwchase17/openai-tools-agent")
agent = create_tool_calling_agent(llm=chat, tools=tools, prompt=chat_prompt_template)

connected_users = {}

@app.post("/chat")
def chat_with_model(msg: Message):
    remove_inactive_users()
    if not msg.user_id or not msg.user_id.strip():
        user_id = generate_user_id()
    else:
        user_id = msg.user_id.strip()

    if user_id not in connected_users:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connected_users[user_id] = {
            "first_seen": now,
            "last_active": now,
            "total_messages": 0,
            "last_qa" : {"q" : "", "a" : "" }
        }

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    connected_users[user_id]["last_active"] = now
    connected_users[user_id]["total_messages"] += 1


    last_q = connected_users[user_id]["last_qa"]["q"]
    last_a = connected_users[user_id]["last_qa"]["a"]

    detected_language = detect(msg.text)
    prompt1 = f"Please answer the following question in {detected_language}. User's question: {msg.text}"


    context = ""
    if last_q and last_a:
        context = f"Previous question asked was : {last_q}\n and its answer was: {last_a}\n"

    full_input = f"{prompt1}{context}Current question is: {msg.text}"

    


    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
    )

    print(f"Last Question : {connected_users[user_id]['last_qa']['q']} \n \n Last Answer :{connected_users[user_id]['last_qa']['a']} ")

    response = agent_executor.invoke({"input": full_input})                                   # Final Answer

    steps = response.get("intermediate_steps", [])
    print("\n\n\nSTEPS ::::: ",steps)    


    connected_users[user_id]["last_qa"] = {
    "q": msg.text,
    "a": response.get("output", "")
}
    
    print(f"Last Question : {connected_users[user_id]['last_qa']['q']} \n \n Last Answer :{connected_users[user_id]['last_qa']['a']} ")
    



    recommended_question = "More about Udyami Yojna eligibility or application process?"

    print(f"Recommended follow-up question for user {user_id}: {recommended_question}")

    print("\n--- Connected Users Log ---")
    for uid, details in connected_users.items():
        print(f"User ID: {uid}, First Seen: {details['first_seen']}, Last Active: {details['last_active']}, Total Messages: {details['total_messages']}")
    print("---------------------------\n")

    
    
    return {
        "user_id": user_id,
        "response": response.get("output", "No response generated"),
        "intermediate_steps": response.get("intermediate_steps", []),
        "recommended_question": recommended_question
    }

    


