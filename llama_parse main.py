from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
import os
from PyPDF2 import PdfReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from prompts import context
from dotenv import load_dotenv
load_dotenv()


llm = Ollama(model="llama2", request_timeout=120)

parser = LlamaParse(result_type="markdown")

file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

embed_model = resolve_embed_model("local:BAAI/bge-m3")

# Create the vector index from documents
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Create a query engine
query_engine = vector_index.as_query_engine(llm=llm)

tools = [
    QueryEngineTool(query_engine,
                        metadata=ToolMetadata(name="api_documentations",
                                            description="this gives documentation about code for an API. Use this for reading docs for the API",
                        ),
                    ),

]

code_llm =Ollama(model="codellama")
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)