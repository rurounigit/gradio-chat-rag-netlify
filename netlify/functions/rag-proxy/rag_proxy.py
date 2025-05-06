# netlify/functions/rag-proxy/rag_proxy.py
import os
import json
import tempfile
import zipfile
import shutil
import logging
from pathlib import Path

# LangChain/Google Imports (Ensure these match your bot.py needs)
import langchain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration ---
# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)

# LangChain Debug (optional, disable in production for cleaner logs)
# langchain.debug = True

# --- Constants (Adapt from your bot.py) ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") # Get from Netlify env vars
# FAISS index is packaged WITH the function, use relative path
FAISS_INDEX_ZIP_PATH = "faiss_index_google.zip"
GOOGLE_EMBEDDING_MODEL_NAME = "models/text-embedding-004" # Or your model
GEMINI_LLM_MODEL_NAME = "gemini-1.5-flash-latest" # Use 1.5 Flash or your preferred model (consider function runtime limits)

# --- Prompts (Copy VERBATIM from your working bot.py) ---
# 1. Contextualizer Prompt
contextualize_q_system_prompt = """Given a chat history and the latest user question... [YOUR FULL PROMPT HERE] ...Now, process the following:"""
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 2. Angela Han Persona QA Prompt
persona_qa_system_prompt = """Instruction for LLM: Adopt the persona... [YOUR FULL PROMPT HERE] ...Answer (as Angela Han):"""
persona_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", persona_qa_system_prompt),
    ("human", "{input}"),
])

# --- Global Variables (Initialized on cold start) ---
# These can be initialized outside the handler to be reused across invocations
# if the execution environment is warm.
embeddings = None
llm = None
retriever = None
rag_chain = None
initialization_error = None

# --- Initialization Function ---
def initialize_components():
    global embeddings, llm, retriever, rag_chain, initialization_error

    if rag_chain: # Already initialized
        return True

    logger.info("Performing cold start initialization...")
    initialization_error = None # Reset error

    if not GOOGLE_API_KEY:
        initialization_error = "GOOGLE_API_KEY environment variable not set."
        logger.critical(initialization_error)
        return False

    try:
        # 1. Initialize Embeddings
        logger.info(f"Initializing Embeddings: {GOOGLE_EMBEDDING_MODEL_NAME}")
        embeddings = GoogleGenerativeAIEmbeddings(
            model=GOOGLE_EMBEDDING_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY
        )
        logger.info("Embeddings initialized successfully.")

        # 2. Load FAISS Index (from packaged zip)
        logger.info(f"Loading FAISS index from packaged zip: {FAISS_INDEX_ZIP_PATH}")
        if not Path(FAISS_INDEX_ZIP_PATH).is_file():
             raise FileNotFoundError(f"FAISS zip file not found at expected path: {Path(FAISS_INDEX_ZIP_PATH).resolve()}")

        temp_extract_dir = tempfile.mkdtemp()
        vector_store = None
        try:
            with zipfile.ZipFile(FAISS_INDEX_ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)
            logger.info(f"Extracted index to temporary directory: {temp_extract_dir}")

            index_path = temp_extract_dir # FAISS needs directory path
            expected_faiss = Path(index_path) / "index.faiss"
            expected_pkl = Path(index_path) / "index.pkl"

            if not expected_faiss.is_file() or not expected_pkl.is_file():
                 raise FileNotFoundError(f"index.faiss or index.pkl not found in extracted dir: {index_path}")

            vector_store = FAISS.load_local(
                index_path,
                embeddings, # Must use the initialized embeddings object
                allow_dangerous_deserialization=True # Required by FAISS load_local
            )
            retriever = vector_store.as_retriever(search_kwargs={'k': 6}) # Use your k value
            logger.info("FAISS index loaded and retriever created successfully.")

        finally:
            if Path(temp_extract_dir).exists():
                shutil.rmtree(temp_extract_dir)
                logger.info(f"Cleaned up temporary index directory: {temp_extract_dir}")

        # 3. Initialize LLM
        logger.info(f"Initializing LLM: {GEMINI_LLM_MODEL_NAME}")
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_LLM_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7, # Your desired temperature
            convert_system_message_to_human=True # Often needed for Gemini models in LangChain
        )
        logger.info("LLM initialized successfully.")

        # 4. Create RAG Chain
        logger.info("Creating RAG chain...")
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        question_answer_chain = create_stuff_documents_chain(llm, persona_qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        logger.info("RAG chain created successfully.")
        return True

    except Exception as e:
        initialization_error = f"Failed during initialization: {e}"
        logger.error(initialization_error, exc_info=True)
        # Reset globals if initialization failed partially
        embeddings = llm = retriever = rag_chain = None
        return False

# --- Netlify Function Handler ---
def handler(event, context):
    # Ensure initialization is complete (handles cold starts)
    if not rag_chain and not initialize_components():
        logger.error(f"Initialization failed, cannot process request. Error: {initialization_error}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": f"Server initialization failed: {initialization_error or 'Unknown error'}"})
        }
    elif initialization_error: # Should have been caught above, but double-check
         logger.error(f"Returning error due to previous initialization failure: {initialization_error}")
         return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": f"Server initialization failed: {initialization_error}"})
        }


    # Netlify passes HTTP method in event['httpMethod']
    if event.get('httpMethod', 'GET').upper() != 'POST':
        return {
            'statusCode': 405,
            'headers': {'Allow': 'POST', "Access-Control-Allow-Origin": "*"},
            'body': json.dumps({'error': 'Method Not Allowed'})
        }

    try:
        # Parse the incoming request body from Gradio-lite
        body = json.loads(event.get('body', '{}'))
        user_input = body.get('message')
        # Gradio chat history is usually [[user_msg, bot_msg], [user_msg, bot_msg], ...]
        history = body.get('history', [])

        if not user_input:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": "Missing 'message' in request body"})
            }

        logger.info(f"Received user input: '{user_input}'")
        logger.debug(f"Received history: {history}")

        # Convert Gradio history to LangChain Message format
        lc_history = []
        for user_msg, ai_msg in history:
            if user_msg: lc_history.append(HumanMessage(content=user_msg))
            if ai_msg: lc_history.append(AIMessage(content=ai_msg))
        # Limit history length if necessary (like in your bot)
        lc_history = lc_history[-10:] # Keep last 5 pairs (10 messages)
        logger.debug(f"Converted LangChain history: {lc_history}")


        # Invoke the RAG chain (use invoke for sync Netlify Python function)
        logger.info("Invoking RAG chain...")
        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": lc_history
        })

        answer = response.get('answer', "Sorry, I couldn't generate a response for that.").strip()
        if not answer:
            answer = "I looked into that but couldn't form a specific answer based on my knowledge."

        logger.info("RAG chain invocation successful.")
        logger.debug(f"Generated answer: '{answer[:100]}...'") # Log truncated answer

        # Send the successful response back to Gradio-lite
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                # CORS header is important for Gradio-lite (different origin in local dev)
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "POST, OPTIONS"
             },
            "body": json.dumps({"answer": answer})
        }

    except json.JSONDecodeError:
        logger.error("Failed to decode JSON body.", exc_info=True)
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": "Invalid JSON in request body"})
        }
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return {
            "statusCode": 500,
             "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": f"An internal server error occurred: {e}"})
        }