import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

# 设置你的OpenAI API密钥
os.environ["OPENAI_API_KEY"] = "sk-"

# 加载数据并构建索引
# 假设你已经有了一些文本数据存储在"data"文件夹中
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 创建查询引擎
query_engine = index.as_query_engine()

# 查询你的数据
response = query_engine.query("谁是XXX")
print(response)