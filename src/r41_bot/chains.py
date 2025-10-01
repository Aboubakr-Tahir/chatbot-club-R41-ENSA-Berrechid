from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.multi_query import MultiQueryRetriever

from prompts import SYSTEM_PROMPT, USER_PROMPT
from config import MODEL_NAME, GOOGLE_API_KEY


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs) if docs else ""


def build_rag_chain(base_retriever):
    # LLM used to generate alternative queries (paraphrases)
    query_llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,
        max_output_tokens=128,
    )

    # Custom prompt to request exactly 4 rewrites
    mq_prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "Rewrite the user's question into 4 distinct, helpful search queries. "
            "Keep language the same as the original. One query per line.\n\n"
            "Original question: {question}"
        ),
    )

    # Build MultiQuery retriever (no num_queries kwarg; we use the prompt)
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=query_llm,
        prompt=mq_prompt,
        include_original=True,  # keep the user’s original query too
    )

    # Normal RAG prompt + answer LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])

    answer_llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,
        max_output_tokens=256,
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | answer_llm
        | StrOutputParser()
    )
    return chain
