import os
from operator import itemgetter
from typing import TypedDict
from dotenv import load_dotenv
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load env variables
load_dotenv()

# Retrieve vector data from PGVector
vector_store = PGVector(
    collection_name=os.getenv("PG_COLLECTION"),
    connection_string=os.getenv("POSTGRES_URL"),
    embedding_function=OpenAIEmbeddings()
)

# Create template
template = """
Answer given the following context:
{context}

Question: {question}
"""

# Create templated prompts from chatbot
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

# Use OpenAI gpt-4-1106-preview
llm = ChatOpenAI(temperature=0, model='gpt-4-1106-preview', streaming=True)


class RagInput(TypedDict):
    question: str


# Create final chain
final_chain = (
    {
        "context": itemgetter("question") | vector_store.as_retriever(),
        "question": itemgetter("question")
    }
    | ANSWER_PROMPT
    | llm
    | StrOutputParser()
).with_types(input_type=RagInput)
