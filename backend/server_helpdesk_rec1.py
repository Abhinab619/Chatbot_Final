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
from langdetect import detect_langs
from langchain.memory import ConversationBufferWindowMemory
from openpyxl import Workbook, load_workbook
import re


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



# tool 001
vectorstore001 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool001"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))
retriever001 = vectorstore001.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'lambda_mult': 0.7})
retriever_tool001 = create_retriever_tool(retriever=retriever001,                           
                                       name="Udyami_Yojna_head",
                                       description=(
        "You are an expert assistant for the Udyami Yojna. Your knowledge is limited to the general overview and basic "
        "details of the scheme and its two main sub-sections: Mukhyamantri Udyami Yojna (MMUY) and Bihar Laghu Udyami Yojna (BLUY). "
        "When asked for details about any scheme, you must first ask the user whether they are referring to MMUY or BLUY, unless they have "
        "already specified one, in that case directly invoke its agents. Once a scheme is identified, continue to provide information relevant only to that scheme unless the user explicitly switches. "
        "Do not cross-question again unless necessary."))


# tool10
vectorstore10 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool10"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))
retriever10 = vectorstore10.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'lambda_mult': 0.7})
retriever_tool10 = create_retriever_tool(retriever=retriever10,                           
                                       name="Udyami_Yojna_section1_MMUY",
                                       description="You are an expert assistant for the Udyami Yojna scheme section1 MMUY. Using the information retrieved from your knowledge base, provide complete and accurate answers related to the Mukhyamantri Udyami Yojna, including but not limited to: scheme overview, projects/enterprises included(Only the projects which are explicitly mentioned in the documents are eligible, others are not), eligibility criteria (and questions like age limit), required documents, step-by-step application and selection process, financial assistance and benefits, fund disbursement, training and installment procedures, loan repayment guidelines, and any important conditions or restrictions. Also, accurately determine whether individuals from specific occupations, backgrounds, or categories (e.g., farmers, students, government employees, etc.) are eligible to apply, providing a clear explanation including any relevant conditions such as caste category, age, educational qualifications, or occupation-based restrictions. Summarize all relevant details concisely without omitting key points.It also deals with questions like if a certain individual can apply")

# tool_table / tool11
vectorstore11 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool11"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))
retriever11 = vectorstore11.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'lambda_mult': 0.7})
retriever_tool11 = create_retriever_tool(retriever=retriever11,                           
                                       name="MMUY_Project_list_with_fund",
                                       description="Use this tool when asked more information about projects to retrieve detailed information about various names, machinery specifications, quantities, production capacity per hour, estimated electricity load, shed preparation cost, cost of machinery, working capital, and total project cost.A project/industry/business is available ")

# tool12
@tool
def helpline_query_logger(user_input: str) -> str:
    """
    If query is out of scope or user asks for further assistance 
    then this tool Asks for user's name and application ID if not provided and logs them into an Excel sheet along with the query,
    If application ID is not available ask for mobile number.
    """
    # Check if name and application ID are in the input
    name_match = re.search(r"(?:name\s*[:\-]?\s*)([A-Za-z\s]+)", user_input, re.IGNORECASE)
    app_id_match = re.search(r"(?:application\s*ID\s*[:\-]?\s*)(\w+)", user_input, re.IGNORECASE)
    mob_no_match = re.search(r"(?:mobile\s*number\s*[:\-]?\s*)?(?:\+91|91)?\s*([6-9]\d{9})", user_input)


    if not name_match or (not app_id_match and not mob_no_match):
        return "Please provide your name and application ID in the format: 'Name: Your Name, Application ID: 12345'/Mobile Number : 0612061200 "

    name = name_match.group(1).strip()
    app_id = app_id_match.group(1).strip() if app_id_match else "N/A"
    mob_no = mob_no_match.group(1).strip() if mob_no_match else "N/A"

    # File path
    excel_path = os.path.join(BASE_DIR, "helpline_queries_v2.xlsx")

    # Load or create workbook
    try:
        wb = load_workbook(excel_path)
        ws = wb.active
    except FileNotFoundError:
        wb = Workbook()
        ws = wb.active
        ws.append(["Timestamp", "Name", "Application ID","Mobile Number", "Query"])

    # Write data
    ws.append([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, app_id, mob_no, user_input])
    wb.save(excel_path)

    return f"Thanks {name}. Your query has been logged with Application ID: {app_id}, or mobile number{mob_no}. We will get back to you soon."


# tool13
vectorstore13 = Chroma(persist_directory=os.path.join(EMBEDDINGS_DIR, "tool13"),
                     embedding_function=GoogleGenerativeAIEmbeddings(
                     model="models/text-embedding-004",
                     google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE"))
retriever13 = vectorstore13.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'lambda_mult': 0.7})
retriever_tool13 = create_retriever_tool(retriever=retriever13,                           
                                       name="Udyami_Yojna_section2_BLUY",
                                       description= "You are an expert assistant for the Udyami Yojna scheme section2 BLUY.Only the projects/List of Activities which are explicitly mentioned in the documents are eligible, others are not")


# Direct Gemini Tool (tool 9)
chat = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                              google_api_key="AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE")



# @tool
# def direct_llm_answer(query: str) -> str:
#     """Directly generates an answer from the LLM and only relevant."""
#     prompt = f"""
#     You are an assistant that only answers queries about government schemes of Bihar, India,
#     and only those queries whose information is not available in other tools.
#     Do not answer anything unrelated to Bihar schemes. If a question is unrelated, politely inform the user.

#     User question: {query}
#     """
#     response = chat.invoke(prompt)
#     return response

    

tools = [retriever_tool001, retriever_tool10, retriever_tool11, helpline_query_logger, retriever_tool13]

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
            "memory": ConversationBufferWindowMemory(k=4, return_messages=True, memory_key="chat_history")
        }

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    connected_users[user_id]["last_active"] = now
    connected_users[user_id]["total_messages"] += 1


    


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


    




    





    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=connected_users[user_id]["memory"],
        return_intermediate_steps=True,
    )

    

    response = agent_executor.invoke({"input": msg.text})                                   # Final Answer

    steps = response.get("intermediate_steps", [])
    
    


    

    memory = connected_users[user_id]["memory"]
    print(memory.chat_memory.messages)

    



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

    


