from operator import itemgetter 
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda,RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


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


def build_rag_chain(retriever):
    """
    Builds the main RAG chain with a decomposition step.
    This chain is now memory-aware.
    """
    llm = get_llm()
    decomposition_chain = build_decomposition_chain()

    # We build the final prompt for the LLM, now including a placeholder for memory
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", USER_PROMPT),
        ]
    )

    # The full chain is defined as a single sequence
    rag_chain = (
        # Step 1: Decompose the question and pass along the original input
        RunnablePassthrough.assign(
            sub_questions=itemgetter("question") | decomposition_chain
        )
        # Step 2: Use the output of Step 1 to retrieve context
        | RunnablePassthrough.assign(
            context=itemgetter("sub_questions")
            | RunnableLambda(lambda x: x.get("questions", []))
            | retriever.map()
            | RunnableLambda(format_context)
        )
        # Step 3: Use the context and original question to generate an answer
        | prompt
        | llm
        | StrOutputParser()
    )

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
    Builds a chain that classifies a user's question into one of three routes.
    """
    llm = get_llm()
    prompt = PromptTemplate.from_template(ROUTER_PROMPT)

    router_chain = prompt | llm | JsonOutputParser()

    return router_chain


def build_query_rewriter_chain():
    """
    Builds a chain that rewrites a user's question to include temporal context.
    """
    llm = get_llm()
    prompt = PromptTemplate.from_template(QUERY_REWRITER_PROMPT)

    # We don't use a JSON parser here because we expect a simple string output
    rewriter_chain = prompt | llm | StrOutputParser()

    return rewriter_chain