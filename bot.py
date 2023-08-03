from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import SKLearnVectorStore
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
import pandas as pd

from model import Model

loader = PyPDFLoader("/home/julian/datasets/pdf_data/paper_1.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=64,
    separators=['\n\n', '\n', '(?=>\. )', ' ', '']
)
texts = text_splitter.split_documents(pages)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vector_db_path = "./paper_1.parquet"
vector_db = SKLearnVectorStore.from_documents(
    texts,
    embedding=embedding,
    persist_path=vector_db_path,
    serializer="parquet"
)

vector_db.persist()
vector_df = pd.read_parquet(vector_db_path)

# Load the Model
llm_model = Model()
generate_text = llm_model.load()

hf_pipeline = HuggingFacePipeline(pipeline=generate_text)

qa = RetrievalQA.from_chain_type(
    llm=hf_pipeline,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    verbose=False
)

query = input("[USER]: ")
result = qa({"query":query})
print('[BOT]: ',result['result'])
