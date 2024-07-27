from llama_index.llms.ollama import Ollama

llm = Ollama(model="llama2", request_timeout=120)

result = llm.complete("Hello World")

print(result)
