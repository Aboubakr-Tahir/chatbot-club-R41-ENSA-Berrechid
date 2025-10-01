from operator import itemgetter 
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnableLambda,RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


from prompts import SYSTEM_PROMPT, USER_PROMPT, DECOMPOSITION_PROMPT
from config import MODEL_NAME, GOOGLE_API_KEY


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
    """
    llm = get_llm()
    decomposition_chain = build_decomposition_chain()

    # 1. The main input is the original question
    # 2. We pass it to the decomposition chain to get sub-questions
    # 3. We also keep the original question using RunnablePassthrough
    decomposed_step = {
        "sub_questions": decomposition_chain,
        "original_question": RunnablePassthrough(),
    }

    # 4. We retrieve documents for each sub-question
    # 5. We format the retrieved documents into a single context string
    retrieval_step = {
        "context": itemgetter("sub_questions")
        | RunnableLambda(lambda x: x.get("questions", [])) # Extract the list of questions
        | retriever.map() # Run retriever for each question in the list
        | RunnableLambda(format_context), # Combine and format the context
        "question": itemgetter("original_question"),
    }


    # 6. We build the final prompt for the LLM
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )

    # 7. We combine everything into the final chain
    rag_chain = (
        decomposed_step
        | RunnablePassthrough.assign(
            context=retrieval_step["context"],
            question=retrieval_step["question"],
        )
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