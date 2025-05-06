# netlify/functions/rag-proxy/rag_proxy.py
import os
import json
import tempfile
import zipfile
import shutil
import logging
from pathlib import Path

# LangChain/Google Imports (Ensure these match your needs)
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Optional: Enable LangChain debug logging for backend function if needed
# langchain.debug = True

# --- Constants ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") # Get from Netlify env vars
FAISS_INDEX_ZIP_PATH = "faiss_index_google.zip" # Relative path within the function bundle
GOOGLE_EMBEDDING_MODEL_NAME = "models/text-embedding-004" # Or your chosen embedding model
GEMINI_LLM_MODEL_NAME = "gemini-2.0-flash-lite" # Use the correct, available model name

# --- Prompts (Copy VERBATIM from your working bot.py or previous setup) ---
# Ensure these are exactly as you need them
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
    ("human", "{input}"), # Input is the rephrased question
])

# --- Global Variables for LangChain components (for potential reuse on warm starts) ---
embeddings = None
llm = None
retriever = None
rag_chain = None
initialization_error = None

# --- Initialization Function ---
# This runs potentially only once per "warm" function instance
def initialize_components():
    global embeddings, llm, retriever, rag_chain, initialization_error

    # If already initialized in this instance, skip
    if rag_chain:
        logger.info("Components already initialized (warm start).")
        return True

    logger.info("Performing cold start initialization of LangChain components...")
    initialization_error = None # Reset error on new attempt

    if not GOOGLE_API_KEY:
        initialization_error = "GOOGLE_API_KEY environment variable not set."
        logger.critical(initialization_error)
        return False

    try:
        # 1. Initialize Embeddings
        logger.info(f"Initializing Embeddings: {GOOGLE_EMBEDDING_MODEL_NAME}")
        embeddings = GoogleGenerativeAIEmbeddings(
            model=GOOGLE_EMBEDDING_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            # Add task_type="RETRIEVAL_DOCUMENT" if appropriate for your embedding model version
        )
        logger.info("Embeddings initialized successfully.")

        # 2. Load FAISS Index (from packaged zip)
        logger.info(f"Loading FAISS index from packaged zip: {FAISS_INDEX_ZIP_PATH}")
        # Get the directory where the function code is running
        script_dir = Path(__file__).parent.resolve()
        zip_full_path = script_dir / FAISS_INDEX_ZIP_PATH

        if not zip_full_path.is_file():
             # Log the directory content for debugging if file not found
             logger.error(f"FAISS zip file not found at expected path: {zip_full_path}")
             logger.error(f"Contents of {script_dir}: {list(script_dir.iterdir())}")
             raise FileNotFoundError(f"FAISS zip file not found at expected path: {zip_full_path}")

        temp_extract_dir = tempfile.mkdtemp()
        vector_store = None
        try:
            with zipfile.ZipFile(zip_full_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)
            logger.info(f"Extracted index to temporary directory: {temp_extract_dir}")

            index_path = temp_extract_dir # FAISS needs the directory path
            # Check if the necessary files exist after extraction
            expected_faiss = Path(index_path) / "index.faiss"
            expected_pkl = Path(index_path) / "index.pkl"
            if not expected_faiss.is_file() or not expected_pkl.is_file():
                 logger.error(f"index.faiss or index.pkl not found in extracted dir: {index_path}")
                 logger.error(f"Contents of {index_path}: {list(Path(index_path).iterdir())}")
                 raise FileNotFoundError(f"index.faiss or index.pkl not found in extracted dir: {index_path}")

            vector_store = FAISS.load_local(
                index_path,
                embeddings, # Use the initialized embeddings object
                allow_dangerous_deserialization=True # Required by FAISS load_local
            )
            # Define retriever (adjust k as needed)
            retriever = vector_store.as_retriever(search_kwargs={'k': 6})
            logger.info("FAISS index loaded and retriever created successfully.")

        finally:
            # Clean up the temporary directory
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
        history_aware_retriever_chain = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        question_answer_chain = create_stuff_documents_chain(llm, persona_qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever_chain, question_answer_chain)
        logger.info("RAG chain created successfully.")
        return True

    except Exception as e:
        initialization_error = f"Failed during LangChain components initialization: {e}"
        logger.error(initialization_error, exc_info=True)
        # Ensure globals are reset if initialization fails
        embeddings = llm = retriever = rag_chain = None
        return False

# --- Netlify Function Handler ---
# This function is called by Netlify for each incoming HTTP request
def handler(event, context):
    global rag_chain, initialization_error # Allow modification of globals if needed

    # --- Allow CORS preflight requests (OPTIONS) ---
    # Browsers send OPTIONS request first for cross-origin POST
    if event.get('httpMethod', 'GET').upper() == 'OPTIONS':
        logger.info("Responding to OPTIONS preflight request")
        return {
            'statusCode': 204, # No Content
            'headers': {
                'Access-Control-Allow-Origin': '*', # Allow requests from any origin
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Max-Age': '86400' # Cache preflight response for 1 day
            },
            'body': ''
        }

    # --- Ensure components are initialized (handles cold starts) ---
    if not rag_chain and not initialize_components():
        logger.error(f"Initialization failed, cannot process request. Error: {initialization_error}")
        # Return error response with CORS headers
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*" # Crucial for error responses too
            },
            "body": json.dumps({"error": f"Server initialization failed: {initialization_error or 'Unknown error'}"})
        }
    elif initialization_error: # If initialization failed on a previous cold start within this instance
         logger.error(f"Returning error due to previous initialization failure: {initialization_error}")
         return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": f"Server initialization failed: {initialization_error}"})
        }

    # --- Handle actual POST request ---
    if event.get('httpMethod', 'GET').upper() != 'POST':
        logger.warning(f"Received non-POST request: {event.get('httpMethod')}")
        return {
            'statusCode': 405, # Method Not Allowed
            'headers': {'Allow': 'POST', "Access-Control-Allow-Origin": "*"},
            'body': json.dumps({'error': 'Method Not Allowed. Please use POST.'})
        }

    # --- Process Request Body ---
    try:
        body = json.loads(event.get('body', '{}'))
        user_input = body.get('message')
        # Gradio chat history format: [[user_msg, ai_msg], ...]
        history = body.get('history', [])

        if not user_input:
            logger.warning("Request received with missing 'message' field.")
            return {
                "statusCode": 400, # Bad Request
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": "Missing 'message' in request body"})
            }

        logger.info(f"Received message for processing: '{user_input}'")
        logger.debug(f"Received history (length {len(history)}): {history}")

        # Convert Gradio history to LangChain Message format
        lc_history = []
        for user_msg, ai_msg in history:
            # Ensure messages are strings and not None before adding
            if user_msg is not None:
                lc_history.append(HumanMessage(content=str(user_msg)))
            if ai_msg is not None:
                lc_history.append(AIMessage(content=str(ai_msg)))

        # Limit history length if desired (e.g., last 10 messages / 5 turns)
        lc_history = lc_history[-10:]
        logger.debug(f"Converted LangChain history (last {len(lc_history)} messages): {lc_history}")

        # --- Invoke the RAG chain ---
        logger.info("Invoking RAG chain...")
        # Netlify Python functions are synchronous, use invoke() not ainvoke()
        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": lc_history
        })

        answer = response.get('answer', "").strip() # Default to empty string if no answer
        if not answer:
            answer = "I looked into that but couldn't form a specific answer based on my knowledge."
            logger.warning("RAG chain returned empty answer, using default.")
        else:
             logger.info("RAG chain invocation successful, answer generated.")
             logger.debug(f"Generated answer preview: '{answer[:100]}...'")

        # --- Send Success Response ---
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*", # Allow requests from frontend
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "POST, OPTIONS"
             },
            "body": json.dumps({"answer": answer}) # Send back JSON with the answer
        }

    # --- Handle Errors During Processing ---
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON body.", exc_info=True)
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": "Invalid JSON in request body"})
        }
    except FileNotFoundError as e: # Catch specific error during FAISS loading potentially
         logger.error(f"File not found error during processing (likely FAISS related): {e}", exc_info=True)
         return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": f"Internal Server Error: Could not load necessary files. Details: {e}"})
        }
    except Exception as e:
        logger.error(f"Unhandled error processing request: {e}", exc_info=True)
        return {
            "statusCode": 500, # Internal Server Error
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": f"An internal server error occurred."}) # Avoid exposing raw error details
        }