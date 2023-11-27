# ChaabiAssignmentIITGuwahati_200108002

## Introduction
The Flask application in ```run_api.py``` serves as an API that generates answers to a query using knowledge from a data source. The application leverages open-source embedding models, vector databases and LLMs (Large Language Models) to build an end-to-end query engine that takes a query from the user and returns the answer from a database. We use Hugging Face's transformers library to load a ```hku-nlp/instructor-base``` model for generating sentence embeddings, and use ```Huggingface Hub``` to load **Flan**, an open-source LLM by Google. The vector embeddings are stored in **FAISS (Facebook AI for Similarity Search)**, which is a vector database developed by Facebook AI. The pipeline involves obtaining the query from the user, generating the corresponding embeddings, obtaining the relevant documents from the vector database using similarity search measures, passing them to the LLM and returning the answer provided by the LLM based on the context extracted from the documents to the user. The pipeline looks like something as follows:

<a href="https://ibb.co/7yYLxsS"><img src="https://i.ibb.co/x5CdKx2/Screenshot-217.png" alt="Screenshot-217" border="0" /></a>

The ```llm-qa``` file initializes the embedding model, vector database, LLM and retrieval chain. The ```run_api.py``` file obtains user from the input, passes it through the pipeline and returns the generated text as a JSON response.

## Requirements
The following dependencies need to be installed to run the code:

transformers==4.35.2 \
sentence_transformers==2.2.2 \
flask==2.2.5 \
langchain==0.0.340 \
faiss==1.7.2 \
huggingface_hub==0.19.4 

Install all the missing dependencies using  
```bash 
pip install --r requirements.txt
```

## Usage

Here are the steps to run the API:
1) Run the ```llm_qa.py``` file to load the embedding model, LLM and create the vector database
2) Run the ```run_api.py``` file to load the API on local host to provide query input and obtain query output

Note that loading models, generating embeddings and creating the vector database may take some time. It is recommended to use any available GPUs to speed up computations.

## Results
We build a query engine on the data source (**bigBasketProducts.csv**) and tried a few questions relevant to it to test the pipeline. The results are as given below:

<a href="https://imgbb.com/"><img src="https://i.ibb.co/3kNjYxb/Screenshot-218.png" alt="Screenshot-218" border="0"></a>

## Colab Support

The Colab notebook contains detailed descriptions and step-by-step instructions for building the query engine. In addition to downloading the notebook from the repository, you can access the Colab notebook at [ChaabiAssigment](https://colab.research.google.com/drive/1B9_zwWApkKAEAWYdVs2HqifMKoxY8ceU?usp=sharing)
![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)
