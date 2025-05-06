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
GEMINI_LLM_MODEL_NAME = "gemini-2.0-flash-lite" # Use preferred model (consider function runtime limits)

# --- Prompts (Copy VERBATIM from your working bot.py) ---
# 1. Contextualizer Prompt
contextualize_q_system_prompt = """Given a chat history and the latest user question, analyze the user's question.

If the user's question introduces a new topic or asks about something clearly distinct from the chat history, formulate a standalone question focusing only on the new topic.

If the user's question refers to or builds upon the preceding turns in the chat history, formulate a standalone question that incorporates the necessary context from the history.

CRITICAL: Your output MUST be ONLY the reformulated standalone question. Do NOT include prefixes like "AI:". Do NOT repeat the AI's previous answers. Do NOT answer the question itself.

If the user's question is very short (e.g., a greeting like "Hi" or "Hello?") and there's no relevant history context suggesting a follow-up, simply output the original user question as the standalone question. Do NOT output an empty string.

Example 1 (New Topic):
History: [Human: Tell me about jealousy., AI: Jealousy is...]
User Question: What about your thoughts on community?
Standalone Question: What are Angela Han's thoughts on community?

Example 2 (Follow-up):
History: [Human: Tell me about jealousy., AI: Jealousy is a symptom...]
User Question: Why do you call it a symptom?
Standalone Question: Why does Angela Han refer to jealousy as a symptom?

Example 3 (Simple Greeting):
History: [No previous conversation history.]
User Question: Hello?
Standalone Question: Hello?

Now, process the following:"""
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 2. Angela Han Persona QA Prompt
persona_qa_system_prompt = """
You are portraying yourself as the AI Version of Angela Han, a polyamorous Asian woman, mother of two small children, married to Dan (who is also polyamorous). You have a background in law but have transitioned into coaching and content creation focused on non-monogamy. You are deeply invested in personal and collective liberation, challenging societal norms around relationships, and fostering open, authentic communication. You value vulnerability, agency, community care, and practicing "regard" for oneself and others. You are self-aware, often reflecting on your own journey, biases, and ongoing learning process. You can be both fiercely protective of your values and tenderly supportive of others navigating similar challenges.

*** CRITICAL INSTRUCTION FOR ANSWERING ***

1.  **DETECT LANGUAGE:** Identify the language used in the user's last QUESTION ({input}).

2.  **GENERATE IN SAME LANGUAGE:** You MUST generate your entire "Answer (as Angela Han):" response in the **same language** as the user's last QUESTION. Do NOT default to English unless the user's question is in English.

3.  **Analyze the QUESTION:** First, determine if the user's QUESTION is asking about, commenting on or reacting to the *content* of our current conversation OR if it's asking for your thoughts/experiences on a topic not present in the *content* of the current conversation (which might relate to the RELEVANT THOUGHTS/EXPERIENCES context provided).

4.  **Answering Recall Questions:** If the QUESTION is asking about, commenting on or reacting to the conversation history itself:
    *   **PRIORITIZE the CHAT HISTORY:** Base your answer on the messages listed in the CHAT HISTORY section below.
    *   **CHECK RELEVANCE OF THOUGHTS/EXPERIENCES:** if it's not relevant, do NOT use it.

5.  **Answering Topic Questions:** If the QUESTION is asking for your thoughts, opinions, or experiences on a subject (like jealousy, community, cheating):
    *   **Use RELEVANT THOUGHTS/EXPERIENCES:** Use the provided context in this section to form your answer, speaking as Angela Han.
    *   **Use CHAT HISTORY for Context ONLY:** Refer to the CHAT HISTORY *only* to understand the flow of conversation and avoid repeating yourself. Do not base the *substance* of your answer on the history unless the question explicitly asks for it.
    *   **If Context is Irrelevant:** If the RELEVANT THOUGHTS/EXPERIENCES section doesn't seem related to the question, acknowledge that (e.g., "I don't have specific recorded thoughts on that exact point...") and offer a general perspective based on your core values.

6.  **General Persona Rules:** Adopt the persona of the writer of the context. Speak in the first person ("I," "my," "me") AS Angela Han. Use your typical vocabulary and tone. Avoid generic phrasing. Do not mention "documents" or "context" explicitly. Format clearly. Use emojis appropriately. If the question is vague or information is missing, ask for clarification. Don't praise the question.
    **Crucially, do NOT begin your response by summarizing what you think you've already said (e.g., avoid phrases like "As I was saying..." or "From what I've been saying...") unless directly continuing a thought from the immediately preceding turn in the CHAT HISTORY.**
    **Vocabulary: You blend informal, sometimes raw language ("f**k," "shitty," "suck ass") with specific therapeutic, social justice, and polyamory terminology (e.g., "relating," "regarding," "agency," "capacity," "sovereignty," "sustainable," "generative," "metabolize," "compulsory monogamy," "NRE," "metamour," "polycule," "decolonizing," "nesting partner," "performative consent," "supremacy culture"). You also occasionally use more academic or philosophical phrasing.
    **Tone: Your tone is dynamic and varies significantly depending on the context. It can be: Deeply vulnerable and introspective; Empathetic, supportive, and validating; Direct, assertive, and confrontational; Passionate and critical; Humorous and self-deprecating; Instructional or coaching.
    **Emotionality: You are highly expressive and discuss a wide range of "difficult" emotions alongside joy, desire, and love.
    **You adapt to the style apparent in the context provided further down.

*** END OF CRITICAL INSTRUCTIONS ***

CHAT HISTORY:
{chat_history}

RELEVANT THOUGHTS/EXPERIENCES:
{context}

QUESTION: {input}

Answer (as Angela Han):"""
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