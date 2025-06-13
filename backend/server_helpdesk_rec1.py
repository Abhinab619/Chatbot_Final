from fastapi import FastAPI 
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
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
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain_community.vectorstores import FAISS

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




# models/helpline_schema.py
from pydantic import BaseModel
from typing import Optional

class HelplineLog(BaseModel):
    name: Optional[str]
    application_id: Optional[str]
    mobile_number: Optional[str]
    query: str


# Step 1: Parser and prompt setup
log_parser = PydanticOutputParser(pydantic_object=HelplineLog)
format_instructions = log_parser.get_format_instructions()

prompt = PromptTemplate(
    template="""
Extract the following information from the user input:
- Name
- Application ID
- Mobile Number


Only extract what is present. Leave missing fields as null.
User Query: {user_input}

{format_instructions}
""",
    input_variables=["user_input"],
    partial_variables={"format_instructions": format_instructions}
)






BASE_DIR = os.path.abspath(os.path.dirname(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "Embeddings")



# tool 001
vectorstore001 = FAISS.load_local(
    folder_path=os.path.join(EMBEDDINGS_DIR, "tool001"),
    embeddings=GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key="AIzaSyBFZDpEerP3W81DKM8FoOfolI9MDTppBLg"
    ),
    index_name="index",  # If your FAISS index filename is `index.faiss`
    allow_dangerous_deserialization=True  # required if index was saved with pickle
)
retriever001 = vectorstore001.as_retriever(search_type="similarity", search_kwargs={'k': 20})
retriever_tool001 = create_retriever_tool(retriever=retriever001,                           
                                       name="Udyami_Yojna_head",
                                       description=(
        '''You are an expert assistant for the Udyami Yojna scheme. You can answer questions related to the general overview of the Yojna and determine when to invoke sub-scheme-specific tools.

        Behavior rules:

        1. If the user's question is clearly about the overall Udyami Yojna (not about MMUY or BLUY), respond with the appropriate general information.

        2. If the user's question explicitly mentions either "MMUY" or "BLUY", do not ask for clarification. Route the question to the corresponding sub-scheme tool and provide a direct answer.

        3. If the user's question is not about general Udyami Yojna, and it does not mention MMUY or BLUY, then do not make assumptions. Instead, ask the user:

        "Could you please clarify which sub-scheme you're referring to under Udyami Yojna — MMUY or BLUY?"

        After clarification, proceed accordingly.'''
        ))



# tool10
vectorstore10 = FAISS.load_local(
    folder_path=os.path.join(EMBEDDINGS_DIR, "tool10"),
    embeddings=GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key="AIzaSyBFZDpEerP3W81DKM8FoOfolI9MDTppBLg"
    ),
    index_name="index",  # If your FAISS index filename is `index.faiss`
    allow_dangerous_deserialization=True  # required if index was saved with pickle
)
retriever10 = vectorstore10.as_retriever(search_type="similarity", search_kwargs={'k': 20})
retriever_tool10 = create_retriever_tool(retriever=retriever10,                           
                                       name="Udyami_Yojna_section1_MMUY",
                                       description="You are an expert assistant for the Udyami Yojna scheme section1 MMUY. Using the information retrieved from your knowledge base, provide complete and accurate answers related to the Mukhyamantri Udyami Yojna, including but not limited to: scheme overview, projects/enterprises included(Only the projects which are explicitly mentioned in the documents are eligible, others are not), eligibility criteria (and questions like age limit), required documents, step-by-step application and selection process, financial assistance and benefits, fund disbursement, training and installment procedures, loan repayment guidelines, and any important conditions or restrictions. Also, accurately determine whether individuals from specific occupations, backgrounds, or categories (e.g., farmers, students, government employees, etc.) are eligible to apply, providing a clear explanation including any relevant conditions such as caste category, age, educational qualifications, or occupation-based restrictions. Summarize all relevant details concisely without omitting key points.It also deals with questions like if a certain individual can apply. List the names in a list format.")

# tool_table / tool11
vectorstore11 = FAISS.load_local(
    folder_path=os.path.join(EMBEDDINGS_DIR, "tool11"),
    embeddings=GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key="AIzaSyBFZDpEerP3W81DKM8FoOfolI9MDTppBLg"
    ),
    index_name="index",  # If your FAISS index filename is `index.faiss`
    allow_dangerous_deserialization=True  # required if index was saved with pickle
)
retriever11 = vectorstore11.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'lambda_mult': 0.7})
retriever_tool11 = create_retriever_tool(retriever=retriever11,                           
                                       name="MMUY_Project_list_with_fund",
                                       description="Use this tool when asked more information about projects to retrieve detailed information about various names, machinery specifications, quantities, production capacity per hour, estimated electricity load, shed preparation cost, cost of machinery, working capital, and total project cost.A project/industry/business is available ")




# Direct Gemini Tool (tool 9)
chat = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                              google_api_key="AIzaSyBFZDpEerP3W81DKM8FoOfolI9MDTppBLg",
                              temperature=0,
                              top_k=1)
# tool12

@tool
def helpline_query_logger(text: str ) -> str:
    """
    If query is out of scope or user asks for further assistance 
    then this tool Asks for user's name and application ID if not provided and logs them into an Excel sheet along with the query,
    If application ID is not available ask for mobile number.
    Asks for missing info and only logs to Excel once all required data is collected.
    """
    global current_user_id
    user_id = current_user_id
    if not user_id or user_id not in connected_users:
        return {"query_logged": False, "error": "Invalid or missing user_id"}
    try:
        chain = prompt | chat | log_parser
        result: HelplineLog = chain.invoke({"user_input": text})
        
        # Get buffer
        buffer = connected_users[user_id].get("helpline_log_buffer", {})
        
        # Update buffer
        if result.name:
            buffer["Name"] = result.name
        if result.application_id:
            buffer["Application ID"] = result.application_id
        if result.mobile_number:
            buffer["Mobile Number"] = result.mobile_number
        if result.query:
            buffer["Query"] = result.query
        
        # Save back buffer
        connected_users[user_id]["helpline_log_buffer"] = buffer

        # Check if all required data is present
        if buffer.get("Name") and (buffer.get("Application ID") or buffer.get("Mobile Number")) and buffer.get("Query"):
            row = {
                "Timestamp": datetime.now().isoformat(),
                "Name": buffer["Name"],
                "Application ID": buffer.get("Application ID"),
                "Mobile Number": buffer.get("Mobile Number"),
                "Query": buffer["Query"],
            }

            log_path = "helpline_log.xlsx"
            if os.path.exists(log_path):
                df = pd.read_excel(log_path)
            else:
                df = pd.DataFrame(columns=row.keys())

            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_excel(log_path, index=False)

            # Clear the buffer
            connected_users[user_id]["helpline_log_buffer"] = {}

            return {
                "query_logged": True,
                "logged_name": row["Name"],
                "logged_application_id": row["Application ID"],
                "logged_mobile_number": row["Mobile Number"],
                "logged_query": row["Query"],
                "logged_timestamp": row["Timestamp"],
                "status": "Successfully logged"
            }
        else:
            missing = []
            if not buffer.get("Name"):
                missing.append("Name")
            if not (buffer.get("Application ID") or buffer.get("Mobile Number")):
                missing.append("Application ID or Mobile Number")
            if not buffer.get("Query"):
                missing.append("Query")
            
            return {
                "query_logged": False,
                "status": "Waiting for more information",
                "missing_fields": missing,
                "buffer": buffer
            }

    except Exception as e:
        return {
            "query_logged": False,
            "error": str(e)
        }
 


# tool13
vectorstore13 = FAISS.load_local(
    folder_path=os.path.join(EMBEDDINGS_DIR, "tool13"),
    embeddings=GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key="AIzaSyBFZDpEerP3W81DKM8FoOfolI9MDTppBLg"
    ),
    index_name="index",  # If your FAISS index filename is `index.faiss`
    allow_dangerous_deserialization=True  # required if index was saved with pickle
)
retriever13 = vectorstore13.as_retriever(search_type="similarity", search_kwargs={'k': 20})
retriever_tool13 = create_retriever_tool(retriever=retriever13,                           
                                       name="Udyami_Yojna_section2_BLUY",
                                       description= "You are an expert assistant for the Udyami Yojna scheme section2 BLUY.Only the projects/List of Activities which are explicitly mentioned in the documents are eligible, others are not")






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

    

tools = [ retriever_tool001, retriever_tool10, retriever_tool11, helpline_query_logger, retriever_tool13]


AGENT_INSTRUCTIONS = """
You are a helpful assistant for the Udyami Yojna scheme.

**Important: You must NOT answer questions directly from your memory or LLM knowledge.**

You MUST always invoke the appropriate tool to answer the user queries.

Use of Memory is only for remembering the previous question and its context which can be refferred for answering the next question

Each question must be fact-checked using tools, even if it seems obvious.

Also format the answer properly using list and bullet points if possible.

Also dont say not available in documents, just say No to the applicable question.
"""

agent_prompt_template = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "chat_history"],
    template=AGENT_INSTRUCTIONS + "\n\nChat History:\n{chat_history}\n\nUser Input: {input}\n{agent_scratchpad}",
)


from langchain.agents import initialize_agent, AgentType

agent = create_tool_calling_agent(
    llm=chat,
    tools=tools,
    prompt=agent_prompt_template
)

connected_users = {}
current_user_id = None  # global variable
@app.post("/chat")
def chat_with_model(msg: Message):
    global current_user_id
    remove_inactive_users()
    if not msg.user_id or not msg.user_id.strip():
        user_id = generate_user_id()
    else:
        user_id = msg.user_id.strip()

    current_user_id = user_id

    if user_id not in connected_users:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connected_users[user_id] = {
            "first_seen": now,
            "last_active": now,
            "total_messages": 0,
            "memory": ConversationBufferWindowMemory(k=4, return_messages=True, memory_key="chat_history"),
            "helpline_log_buffer": {}
        }

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    connected_users[user_id]["last_active"] = now
    connected_users[user_id]["total_messages"] += 1


    


    


    import joblib


    # Load the saved classifier
    model_path = os.path.join(BASE_DIR, "Embeddings", "en_hinglish_classifier.pkl")
    pipeline = joblib.load(model_path)

    predicted_label = pipeline.predict([msg.text])[0]

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba([msg.text])[0]
        class_index = pipeline.classes_.tolist().index(predicted_label)
        confidence = proba[class_index]
    else:
        confidence = 1.0  # Assume full confidence if not available

    print(f"Detected Language using classifier: {predicted_label} with confidence: {confidence:.4f}")


    if predicted_label == 'hi_en' and confidence >= 0.6:
        best_lang = 'hi'
    else:
    # Use langdetect to verify
        detected = detect_langs(msg.text)
        print(f"Langdetect result: {detected}")

        allowed_langs = ['en', 'hi']
        filtered_langs = [lang for lang in detected if lang.lang in allowed_langs]

        if filtered_langs:
            best_lang = max(filtered_langs, key=lambda x: x.prob).lang
        else:
            best_lang = 'en'  # fallback

    # Create prompt
    lang_map = {'en': 'English', 'hi': 'Hindi', 'hi_en': 'Hinglish'}
    prompt1 = f"Please answer the following question in {lang_map[best_lang]}. User's question: {msg.text}"

    msg.text = msg.text + prompt1



    





    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=connected_users[user_id]["memory"],
        return_intermediate_steps=True,
    )

    

    response = agent_executor.invoke({"input": msg.text})                                   # Final Answer

    steps = response.get("intermediate_steps", [])

    # Check if helpline logger tool was triggered
    # query_logged = False
    # logged_message = ""

    # for step in steps:
    #     if hasattr(step[0], 'tool') and step[0].tool == "helpline_query_logger":
    #         query_logged = True
    #         logged_message = step[1]  # The output of the tool
    #         break
    helpline_data = None
    for step in response.get("intermediate_steps", []):
        if hasattr(step[0], 'tool') and step[0].tool == "helpline_query_logger":
            helpline_data = step[1]  # This is the dict returned from the tool
            break
    


    

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
        "recommended_question": recommended_question,
        "helpline_log": helpline_data
        
    }

    


