from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
import sentence_transformers
from torch import cuda
from InstructorEmbedding import INSTRUCTOR
from langchain.vectorstores import FAISS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

import os, sys, warnings
warnings.filterwarnings('ignore')

HUGGINGFACEHUB_API_TOKEN = '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'
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

def generate_answer(query):
    embedding_model_id = 'hku-nlp/instructor-base'
    llm_model = 'google/flan-t5-xxl'
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    loader = CSVLoader(file_path = 'bigBasketProducts.csv', encoding = 'utf-8')
    data = loader.load()

    embedding_model = HuggingFaceInstructEmbeddings(model_name = embedding_model_id,
                                                    model_kwargs = {'device': device},
                                                    encode_kwargs = {'device': device, 'batch_size': 32})
    
    if (os.path.exists('FAISS_Index')):
        vectorstore = FAISS.load_local('FAISS_Index', embedding_model)
    else:
        vectorstore = FAISS.from_documents(data, embedding_model)
        vectorstore.save_local('FAISS_Index')

    callbacks = [StreamingStdOutCallbackHandler()]
    llm = HuggingFaceHub(repo_id = 'google/flan-t5-xxl', 
                         model_kwargs = {'temperature': 0.5, 'max_length': 128, 'device': device})
    
    qa_pipeline = RetrievalQA.from_chain_type(llm = llm, 
                                              chain_type = 'stuff', 
                                              retriever = vectorstore.as_retriever(), 
                                              callbacks = callbacks)
    
    return qa_pipeline(query)['result']
                                                
print(generate_answer('Best cooking oil?'))
    
    