from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
import os
from PyPDF2 import PdfReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from prompts import context
from code_reader import code_reader
# Initialize the LLM
llm = Ollama(model="llama2", request_timeout=3600)

# Define a parser function for PDF files
def parse_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Specify the correct path to your PDF file
pdf_directory = "./data"  # Replace with the actual directory containing your PDF files

# Ensure the directory exists
if not os.path.exists(pdf_directory):
    raise ValueError(f"Directory {pdf_directory} does not exist.")

# Define a file extractor for .pdf files
file_extractor = {".pdf": parse_pdf}

# Load documents
documents = SimpleDirectoryReader(pdf_directory, file_extractor=file_extractor).load_data()

# Resolve the embedding model
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
    code_reader,

]

code_llm =Ollama(model="codellama", request_timeout=3600)
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)