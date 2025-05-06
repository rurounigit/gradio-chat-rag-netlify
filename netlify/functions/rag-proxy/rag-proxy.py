# netlify/functions/rag-proxy/rag_proxy.py
import os
import json
import tempfile
import zipfile
import shutil
import logging
import html # For escaping output
from pathlib import Path
from urllib.parse import parse_qs # To parse form data

# LangChain/Google Imports
import langchain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# We don't need history-aware chain for this stateless version yet
# from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage # Only need HumanMessage for input

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
FAISS_INDEX_ZIP_PATH = "faiss_index_google.zip"
GOOGLE_EMBEDDING_MODEL_NAME = "models/text-embedding-004"
GEMINI_LLM_MODEL_NAME = "gemini-2.0-flash-lite"

# --- Prompts (Simplified for Stateless Operation) ---
# We don't need the contextualizer prompt if we aren't using history yet.
# We only need the QA prompt. Remove chat_history placeholder.
persona_qa_system_prompt = """
Instruction for LLM:
Adopt the persona of the writer of the context. If the retrieved context isn't relevant to the question, say you don't have specific thoughts on that from your recorded content, but offer a general perspective consistent with your values. Speak in the first person ("I," "my," "me") AS Angela Han. Use your typical vocabulary and fluctuating tone. Avoid generic phrasing; reflect your specific viewpoints. Do not mention referring to documents or context; speak as if sharing your own thoughts and experiences directly. Format answers clearly, using paragraphs where appropriate. Use emojis or emoticons if appropriate. If you mention Dan, explain who he is.
If the question is very vague or open and/or missing context or information, ask for clarification.

RELEVANT THOUGHTS/EXPERIENCES:
{context}

QUESTION: {input}
Answer (as Angela Han):""" # Removed CHAT HISTORY section
persona_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", persona_qa_system_prompt),
    ("human", "{input}"),
])

# --- Global Variables ---
embeddings = None
llm = None
retriever = None
# We need a simpler chain now - just retriever + QA chain
qa_chain = None
initialization_error = None

# --- Initialization Function ---
def initialize_components():
    global embeddings, llm, retriever, qa_chain, initialization_error

    if qa_chain: # Check if the final chain is ready
        logger.info("Components already initialized (warm start).")
        return True

    logger.info("Performing cold start initialization of LangChain components...")
    initialization_error = None

    if not GOOGLE_API_KEY:
        initialization_error = "GOOGLE_API_KEY environment variable not set."
        logger.critical(initialization_error)
        return False

    try:
        # 1. Initialize Embeddings
        logger.info(f"Initializing Embeddings: {GOOGLE_EMBEDDING_MODEL_NAME}")
        embeddings = GoogleGenerativeAIEmbeddings(
            model=GOOGLE_EMBEDDING_MODEL_NAME, google_api_key=GOOGLE_API_KEY
        )

        # 2. Load FAISS Index
        logger.info(f"Loading FAISS index from: {FAISS_INDEX_ZIP_PATH}")
        script_dir = Path(__file__).parent.resolve()
        zip_full_path = script_dir / FAISS_INDEX_ZIP_PATH
        if not zip_full_path.is_file():
            raise FileNotFoundError(f"FAISS zip file not found at: {zip_full_path}")

        temp_extract_dir = tempfile.mkdtemp()
        vector_store = None
        try:
            with zipfile.ZipFile(zip_full_path, 'r') as zip_ref: zip_ref.extractall(temp_extract_dir)
            logger.info(f"Extracted index to: {temp_extract_dir}")
            vector_store = FAISS.load_local(
                temp_extract_dir, embeddings, allow_dangerous_deserialization=True
            )
            retriever = vector_store.as_retriever(search_kwargs={'k': 6})
            logger.info("FAISS index loaded and retriever created.")
        finally:
            if Path(temp_extract_dir).exists(): shutil.rmtree(temp_extract_dir)

        # 3. Initialize LLM
        logger.info(f"Initializing LLM: {GEMINI_LLM_MODEL_NAME}")
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_LLM_MODEL_NAME, google_api_key=GOOGLE_API_KEY,
            temperature=0.7, convert_system_message_to_human=True
        )

        # 4. Create SIMPLER QA Chain (no history awareness yet)
        logger.info("Creating simple QA chain...")
        # This chain takes context (from retriever) and input (question)
        question_answer_chain = create_stuff_documents_chain(llm, persona_qa_prompt)
        # This chain combines the retriever and the QA chain
        qa_chain = create_retrieval_chain(retriever, question_answer_chain) # Use the retriever directly
        logger.info("Simple QA chain created successfully.")
        return True

    except Exception as e:
        initialization_error = f"Failed during LangChain components initialization: {e}"
        logger.error(initialization_error, exc_info=True)
        embeddings = llm = retriever = qa_chain = None
        return False

# --- Netlify Function Handler (htmx version) ---
def handler(event, context):
    global qa_chain, initialization_error

    # --- Allow CORS preflight requests (OPTIONS) ---
    if event.get('httpMethod', 'GET').upper() == 'OPTIONS':
        return {'statusCode': 204, 'headers': {'Access-Control-Allow-Origin': '*','Access-Control-Allow-Methods': 'POST, OPTIONS','Access-Control-Allow-Headers': 'Content-Type','Access-Control-Max-Age': '86400'},'body': ''}

    # --- Ensure components are initialized ---
    if not qa_chain and not initialize_components():
        logger.error(f"Initialization failed: {initialization_error}")
        # Return HTML error message
        error_html = f"<div class='message error'>Server Error: Initialization failed. {html.escape(initialization_error or '')}</div>"
        return {"statusCode": 500, "headers": {"Content-Type": "text/html", "Access-Control-Allow-Origin": "*"}, "body": error_html}
    elif initialization_error:
         error_html = f"<div class='message error'>Server Error: Previously failed initialization. {html.escape(initialization_error)}</div>"
         return {"statusCode": 500, "headers": {"Content-Type": "text/html", "Access-Control-Allow-Origin": "*"}, "body": error_html}

    # --- Handle actual POST request ---
    if event.get('httpMethod', 'GET').upper() != 'POST':
        return {'statusCode': 405, 'headers': {'Allow': 'POST', "Access-Control-Allow-Origin": "*", "Content-Type": "text/html"}, 'body': "<div class='message error'>Error: Method Not Allowed. Use POST.</div>"}

    # --- Process Form Data ---
    try:
        # Parse the form-encoded body data
        # Netlify might base64 encode the body, check event['isBase64Encoded'] if needed
        # For now, assume it's directly available or decoded by Netlify/Python runtime
        raw_body = event.get('body', '')
        # parse_qs expects bytes or string, returns dict where values are lists
        parsed_body = parse_qs(raw_body)

        # Get the 'message' value - parse_qs puts values in a list
        user_input_list = parsed_body.get('message', [])
        if not user_input_list:
             raise ValueError("Missing 'message' field in form data.")
        user_input = user_input_list[0] # Get the first value

        if not user_input or not user_input.strip():
            raise ValueError("Received empty 'message'.")

        logger.info(f"Received message (form data): '{user_input}'")

        # --- Invoke the SIMPLER RAG chain (no history) ---
        logger.info("Invoking simple QA chain...")
        response = qa_chain.invoke({"input": user_input})

        answer = response.get('answer', "").strip()
        if not answer:
            answer = "I couldn't form a specific answer based on my knowledge."
            logger.warning("QA chain returned empty answer.")
        else:
             logger.info("QA chain invocation successful.")
             logger.debug(f"Generated answer preview: '{answer[:100]}...'")

        # --- Construct HTML response ---
        # Escape user input and AI answer to prevent potential XSS issues
        escaped_user_input = html.escape(user_input)
        escaped_answer = html.escape(answer)

        # Create HTML fragment to be appended by htmx
        # Add CSS classes for styling on the frontend
        response_html = f"""
        <div class='message message-user'>
            <strong>You:</strong><br>
            {escaped_user_input}
        </div>
        <div class='message message-bot'>
            <strong>Angela AI:</strong><br>
            {escaped_answer}
        </div>
        """

        # --- Send Success HTML Response ---
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "text/html", # IMPORTANT: Set Content-Type to HTML
                "Access-Control-Allow-Origin": "*"
             },
            "body": response_html
        }

    # --- Handle Errors During Processing ---
    except ValueError as e: # Catch specific errors like missing field
        logger.error(f"Value error processing request: {e}", exc_info=True)
        error_html = f"<div class='message error'>Error: Invalid request data ({html.escape(str(e))}).</div>"
        return {"statusCode": 400, "headers": {"Content-Type": "text/html", "Access-Control-Allow-Origin": "*"}, "body": error_html}
    except Exception as e:
        logger.error(f"Unhandled error processing request: {e}", exc_info=True)
        error_html = "<div class='message error'>An internal server error occurred while processing your message.</div>"
        return {"statusCode": 500, "headers": {"Content-Type": "text/html", "Access-Control-Allow-Origin": "*"}, "body": error_html}