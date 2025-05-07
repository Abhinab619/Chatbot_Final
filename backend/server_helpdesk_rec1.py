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
from langdetect import detect,detect_langs

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



# tool10
vectorstore10 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool10"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))
retriever10 = vectorstore10.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'lambda_mult': 0.7})
retriever_tool10 = create_retriever_tool(retriever=retriever10,                           
                                       name="Udyami_Yojna",
                                       description="You are an expert assistant for the Udyami Yojna scheme. Using the information retrieved from your knowledge base, provide complete and accurate answers related to the Mukhyamantri Udyami Yojna, including but not limited to: scheme overview, projects/enterprises included, eligibility criteria, required documents, step-by-step application and selection process, financial assistance and benefits,fund disbursement, training and installment procedures, loan repayment guidelines, and any important conditions or restrictions. Summarize all relevant details concisely without omitting key points.")

# tool_table
vectorstore11 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool11"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))
retriever11 = vectorstore11.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'lambda_mult': 0.7})
retriever_tool11 = create_retriever_tool(retriever=retriever11,                           
                                       name="Udyami_Yojna_Project_list_with_fund",
                                       description="Use this tool when asked more information about projects to retrieve detailed information about various names, machinery specifications, quantities, production capacity per hour, estimated electricity load, shed preparation cost, cost of machinery, working capital, and total project cost. ")





# Direct Gemini Tool (tool 9)
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

tools = [retriever_tool10,retriever_tool11, direct_llm_answer]

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

    langs = detect_langs(msg.text)
    print(langs)




# Filter for only English and Hindi
    allowed_langs = ['en', 'hi']
    filtered_langs = [lang for lang in langs if lang.lang in allowed_langs]

# Choose the one with the highest probability
    if not filtered_langs:
        prompt1 = f"Please answer the following questions in en. User's question: {msg.text}"
    else:
        best_lang = max(filtered_langs, key=lambda x: x.prob).lang
        lang_map = {'en': 'English', 'hi': 'Hindi'}
        prompt1 = f"Please answer the following question in {lang_map[best_lang]}. User's question: {msg.text}"


    context = ""
    if last_q and last_a:
        context = f"Previous question asked was : {last_q}\n and its answer was: {last_a}\n"

    full_input = f"{prompt1}{context}Current question is: {msg.text}, Keep the context of previous questions in mind, but dont use their answers"

    





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

    


