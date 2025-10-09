from operator import itemgetter 
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda,RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.messages import HumanMessage, AIMessage


from .prompts import SYSTEM_PROMPT, USER_PROMPT, DECOMPOSITION_PROMPT , ROUTER_PROMPT , QUERY_REWRITER_PROMPT
from .config import MODEL_NAME, GOOGLE_API_KEY


def get_llm():
    """Initializes and returns the LLM."""
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0,
        convert_system_message_to_human=True,
        google_api_key=GOOGLE_API_KEY,
    )
    
def format_context(docs_list: list) -> str:
    """
    Flattens a list of document lists and formats them into a single string.
    Also removes duplicate documents.
    """
    all_docs = [doc for sublist in docs_list for doc in sublist]
    # Remove duplicates based on page_content
    unique_docs = {doc.page_content: doc for doc in all_docs}.values()
    return "\n\n".join(doc.page_content for doc in unique_docs)    

# --- New Function Below ---
def format_chat_history_for_prompt(chat_history: list) -> str:
    """Formats chat history into a readable string for the prompt."""
    if not chat_history:
        return "No history."
    
    # Convert Pydantic models to dicts if necessary
    history_dicts = [msg.dict() if hasattr(msg, 'dict') else msg for msg in chat_history]

    formatted_messages = []
    for msg in history_dicts:
        # *** THE FIX IS HERE ***
        # Check for 'type' (from memory objects) or 'role' (from API objects)
        role = msg.get('type', msg.get('role', 'unknown')).capitalize()
        content = msg.get('content', '')
        formatted_messages.append(f"{role}: {content}")

    return "\n".join(formatted_messages)


def build_rag_chain(retriever):
    """
    Builds the main RAG chain with a decomposition step.
    This version is optimized for true end-to-end streaming.
    """
    llm = get_llm()
    decomposition_chain = build_decomposition_chain()

    # The final prompt for the LLM, now including a placeholder for memory
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", USER_PROMPT),
        ]
    )

    # This is a regular, non-streaming chain that just retrieves context
    retrieval_chain = (
        itemgetter("question")
        | decomposition_chain
        | RunnableLambda(lambda x: x.get("questions", []))
        | retriever.map()
        | RunnableLambda(format_context)
    )

    # This is the main chain that will handle the generation
    # It passes the question and history directly to the prompt and LLM
    conversational_rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    # We use RunnablePassthrough.assign to run the retrieval chain in parallel
    # and add its output ("context") to the input of the conversational chain.
    # This allows the LLM to start generating a response immediately while context
    # is still being fetched.
    rag_chain = RunnablePassthrough.assign(
        context=retrieval_chain,
    ) | conversational_rag_chain

    return rag_chain

# --- New Function Below ---

def build_decomposition_chain():
    """
    Builds a chain that takes a user's question and decomposes it into sub-questions.
    """
    llm = get_llm()
    prompt = PromptTemplate.from_template(DECOMPOSITION_PROMPT)
    
    decomposition_chain = prompt | llm | JsonOutputParser()
    
    return decomposition_chain


def build_router_chain():
    """
    Builds a chain that classifies a user's question. Now includes chat history.
    """
    llm = get_llm()
    
    # The prompt now expects 'question' and 'chat_history'
    prompt = PromptTemplate(
        template=ROUTER_PROMPT,
        input_variables=["question", "chat_history"],
    )

    # We create a passthrough that formats the history before sending it to the prompt
    router_chain = (
        RunnablePassthrough.assign(
            chat_history=lambda x: format_chat_history_for_prompt(x["chat_history"])
        )
        | prompt
        | llm
        | JsonOutputParser()
    )

    return router_chain


def build_query_rewriter_chain():
    """
    Builds a chain that rewrites a user's question. Now includes chat history.
    """
    llm = get_llm()
    
    # The prompt now expects 'question' and 'chat_history'
    prompt = PromptTemplate(
        template=QUERY_REWRITER_PROMPT,
        input_variables=["question", "chat_history"],
    )

    # We create a passthrough that formats the history before sending it to the prompt
    rewriter_chain = (
        RunnablePassthrough.assign(
            chat_history=lambda x: format_chat_history_for_prompt(x["chat_history"])
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return rewriter_chain