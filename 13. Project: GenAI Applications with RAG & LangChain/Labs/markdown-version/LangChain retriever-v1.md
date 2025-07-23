# **Develop a Retriever to Fetch Document Segments Based on Queries**


Estimated time needed: **40** minutes


## Overview


Imagine you are working on a project that involves processing a large collection of text documents, such as research papers, legal documents, or customer service logs. Your task is to develop a system that can quickly retrieve the most relevant segments of text based on a user's query. Traditional keyword-based search methods might not be sufficient, as they often fail to capture the nuanced meanings and contexts within the documents. To address this challenge, you can use different types of retrievers based on LangChain.

Using retrievers is crucial for several reasons:

- Efficiency: Retrievers enable fast and efficient retrieval of relevant information from large datasets, saving time and computational resources.
- Accuracy: By leveraging advanced retrieval techniques, these tools can provide more accurate and contextually relevant results compared to traditional search methods.
- Versatility: Different retrievers can be tailored to specific use cases, making them adaptable to various types of text data and query requirements.
- Context awareness: Some retrievers, like the Parent Document Retriever, can consider the broader context of the document, enhancing the relevance of the retrieved segments.


<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/EUODrOFxvSSNL935zpwh9A/retriever.png" width="50%" alt="retriever"/>


In this lab, you will learn how to use various retrievers to efficiently extract relevant document segments from text using LangChain. 
You will learn about four types of retrievers: `Vector Store-backed Retriever`, `Multi-Query Retriever`, `Self-Querying Retriever`, and `Parent Document Retriever`. You will also learn the differences between these retrievers and understand the appropriate situations in which to use each one. By the end of this lab, you will be equipped with the skills to implement and utilize these retrievers in your projects.


## __Table of Contents__

<ol>
    <li><a href="#Objectives">Objectives</a></li>
    <li>
        <a href="#Setup">Setup</a>
        <ol>
            <li><a href="#Installing-required-libraries">Installing required libraries</a></li>
            <li><a href="#Defining-helper-functions">Defining helper functions</a>
                <ol>
                    <li><a href="#Build-LLM">Build LLM</a></li>
                    <li><a href="#Text-splitter">Text splitter</a></li>
                    <li><a href="#Embedding-model">Embedding model</a></li>
                </ol>
            </li>
        </ol>
    </li>
    <li>
        <a href="#Retrievers">Retrievers</a>
        <ol>
            <li><a href="#Vector-Store-Backed-Retriever">Vector Store-Backed Retriever</a></li>
            <li><a href="#Multi-Query-Retriever">Multi-Query Retriever</a></li>
            <li><a href="#Self-Querying-Retriever">Self-Querying Retriever</a></li>
            <li><a href="#Parent-Document-Retriever">Parent Document Retriever</a></li>
        </ol>

   
            
<li><a href="#Exercises">Exercises</a>
<ol>
<li><a href="#Retrieve-top-2-results-using-vector-store-backed-retriever">Retrieve top 2 results using vector store-backed retriever</a></li>
<li><a href="#Self-Querying-Retriever-for-a-query">Self querying retriever for a query</a></li>
</ol>
</li>


## Objectives

After completing this lab, you will be able to:

- Use various types of retrievers to efficiently extract relevant document segments from text, leveraging LangChain's capabilities.
- Apply the Vector Store-backed Retriever to solve problems involving semantic similarity and relevance in large text datasets.
- Utilize the Multi-Query Retriever to address situations where multiple query variations are needed to capture comprehensive results.
- Implement the Self-Querying Retriever to automatically generate and refine queries, enhancing the accuracy of information retrieval.
- Employ the Parent Document Retriever to maintain context and relevance by considering the broader context of the parent document.


----


## Setup


For this lab, you will use the following libraries:

*   [`ibm-watson-ai`](https://ibm.github.io/watsonx-ai-python-sdk/index.html) for using LLMs from IBM's watsonx.ai.
*   [`langchain`, `langchain-ibm`, `langchain-community`](https://www.langchain.com/) for using relevant features from LangChain.
*   [`pypdf`](https://pypi.org/project/pypdf/)is an open-source pure Python PDF library capable of splitting, merging, cropping, and transforming the pages of PDF files.
*   [`chromadb`](https://www.trychroma.com/) is an open-source vector database used to store embeddings.
*   [`lark`](https://pypi.org/project/lark/) is a general-purpose parsing library for Python. It is necessary for a Self-Querying Retriever.


### Installing required libraries

The following required libraries are __not__ preinstalled in the Skills Network Labs environment. __You must run the following cell__ to install them:

**Note:** The version is being pinned here to specify the version. It's recommended that you do this as well. Even if the library is updated in the future, the installed library could still support this lab work.

This might take approximately 1-2 minutes. 



```python
# Please restart the kernel and run all cells after executing this cell.
# !pip install --user "ibm-watsonx-ai==1.1.2" | tail -n 1
# !pip install --user "langchain==0.2.1" | tail -n 1
# !pip install --user "langchain-ibm==0.1.11" | tail -n 1
# !pip install --user "langchain-community==0.2.1" | tail -n 1
# !pip install --user "chromadb==0.4.24" | tail -n 1
# !pip install --user "pypdf==4.3.1" | tail -n 1
# !pip install --user "lark==1.1.9" | tail -n 1
```

After you install the libraries, restart your kernel. You can do that by clicking the **Restart the kernel** icon.

<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/QrUNwLZfVySxQ9xvbOJgyQ/restart.png" width="80%" alt="Restart kernel">


### Defining helper functions

_Use the following code to define some helper functions to reduce the repeat work in the notebook:_



```python
# You can use this section to suppress warnings generated by your code:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')
```

The following functions are prerequisite knowledge for understanding the topic of this project—retrievers. These functions include:

- Building LLMs
- Splitting documents into chunks
- Building an embedding model

The relevant knowledge and details of these functions have been covered in previous lessons.


#### Build LLM



```python
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models.extensions.langchain import WatsonxLLM
```

`mixtral-8x7b-instruct-v01` is used as the base foundational LLM.



```python
def llm():
    model_id = 'mistralai/mixtral-8x7b-instruct-v01'
    
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
        GenParams.TEMPERATURE: 0.5, # this randomness or creativity of the model's responses
    }
    
    credentials = {
        "url": "https://us-south.ml.cloud.ibm.com"
    }
    
    
    project_id = "skills-network"
    
    model = ModelInference(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id
    )
    
    mixtral_llm = WatsonxLLM(model = model)
    return mixtral_llm
```


```python
llm = llm()
llm
```




    WatsonxLLM(model=<ibm_watsonx_ai.foundation_models.inference.model_inference.ModelInference object at 0x72bbe2b54860>)



#### Text splitter



```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
```


```python
def text_splitter(data, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks
```

#### Embedding model


The following code demonstrates how to build an embedding model using the `watsonx.ai` package.

For this project, the `ibm/slate-125m-english-rtrvr` embedding model is used.



```python
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings
```


```python
def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    
    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )
    return watsonx_embedding
```

## Retrievers


A retriever is an interface designed to return documents based on an unstructured query. Unlike a vector store, which stores and retrieves documents, a retriever's primary function is to find and return relevant documents. While vector stores can serve as the backbone of a retriever, there are various other types of retrievers that can be used as well.


Retrievers take a string `query` as input and output a list of `Documents`.


### Vector Store-Backed Retriever


A vector store retriever is a type of retriever that utilizes a vector store to fetch documents. It acts as a lightweight wrapper around the vector store class, enabling it to conform to the retriever interface. This retriever leverages the search methods implemented by the vector store, such as similarity search and Maximum Marginal Relevance (MMR), to query texts stored within it.


Before demonstrating this retriever, you need to load some example text. A `.txt` document has been prepared for you.



```python
!wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/MZ9z1lm-Ui3YBp3SYWLTAQ/companypolicies.txt"
```

Use `TextLoader` to load the document.



```python
from langchain_community.document_loaders import TextLoader
```


```python
loader = TextLoader("companypolicies.txt")
txt_data = loader.load()
```

Let's take a look at this document. This is a document about different policies in a company.



```python
txt_data
```

Split `txt_data` into chunks. `chunk_size = 200`, `chunk_overlap = 20` has been set.



```python
chunks_txt = text_splitter(txt_data, 200, 20)
```

Store the embeddings into a `ChromaDB`.



```python
from langchain.vectorstores import Chroma
```


```python
vectordb = Chroma.from_documents(chunks_txt, watsonx_embedding())
```

#### Simple similarity search


Here is an example of a simple similarity search based on the vector database.

For this demonstration, the query has been set to "email policy".



```python
query = "email policy"
retriever = vectordb.as_retriever()
```


```python
docs = retriever.invoke(query)
```

By default, the number of retrieval results is four, and they are ranked by similarity level.



```python
docs
```

You can also specify `search kwargs` like `k` to limit the retrieval results.



```python
retriever = vectordb.as_retriever(search_kwargs={"k": 1})
docs = retriever.invoke(query)
docs
```

#### MMR retrieval


MMR in vector stores is a technique used to balance the relevance and diversity of retrieved results. It selects documents that are both highly relevant to the query and minimally similar to previously selected documents. This approach helps to avoid redundancy and ensures a more comprehensive coverage of different aspects of the query.


The following code is showing how to conduct an MMR search in a vector database. You just need to sepecify `search_type="mmr"`.



```python
retriever = vectordb.as_retriever(search_type="mmr")
docs = retriever.invoke(query)
docs
```

#### Similarity score threshold retrieval


You can also set a retrieval method that defines a similarity score threshold, returning only documents with a score above that threshold.



```python
retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.4}
)
docs = retriever.invoke(query)
docs
```

### Multi-Query Retriever


Distance-based vector database retrieval represents queries in high-dimensional space and finds similar embedded documents based on "distance". However, retrieval results may vary with subtle changes in query wording or if the embeddings do not accurately capture the data's semantics.

The `MultiQueryRetriever` addresses this by using an LLM to generate multiple queries from different perspectives for a given user input query. For each query, it retrieves a set of relevant documents and then takes the unique union of these results to form a larger set of potentially relevant documents. By generating multiple perspectives on the same question, the `MultiQueryRetriever` can potentially overcome some limitations of distance-based retrieval, resulting in a richer and more diverse set of results.


The following picture shows the difference between retrievers solely based on distance and the Multi-Query Retriever.


<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/NCZCJ26bp3uKTa0gp8Agwg/multiquery.png" width="40%" alt="multiquery"/>


Let's consider the query sentence, `"I like cats"`.

On the upper side of the picture, you can see a retriever that relies solely on distance. This retriever calculates the distance between the query and the documents in the vector store, returning the document with the closest match.

On the lower side, you can see a multi-query retriever. It first uses an LLM to generate multiple queries from different perspectives based on the user's input query. For each generated query, it retrieves relevant documents and then returns the union of these results.


A PDF document has been prepared to demonstrate this Multi-Query Retriever.



```python
from langchain_community.document_loaders import PyPDFLoader
```


```python
loader = PyPDFLoader("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ioch1wsxkfqgfLLgmd-6Rw/langchain-paper.pdf")
pdf_data = loader.load()
```

Let's take a look at the first page of this paper. This paper is talking about the framework LangChain.



```python
pdf_data[1]
```

Split document and store the embeddings into a vector database.



```python
# Split
chunks_pdf = text_splitter(pdf_data, 500, 20)

# VectorDB
ids = vectordb.get()["ids"]
vectordb.delete(ids) # We need to delete existing embeddings from previous documents and then store current document embeddings in.
vectordb = Chroma.from_documents(documents=chunks_pdf, embedding=watsonx_embedding())
```

The `MultiQueryRetriever` function from LangChain is used.



```python
from langchain.retrievers.multi_query import MultiQueryRetriever

query = "What does the paper say about langchain?"

retriever = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm()
)
```

Set logging for the queries.



```python
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
```


```python
docs = retriever.invoke(query)
docs
```

From the log results, you can see that the LLM generated three additional queries from different perspectives based on the given query.

The returned results are the union of the results from each query.


### Self-Querying Retriever


A Self-Querying Retriever, as the name suggests, has the ability to query itself. Specifically, given a natural language query, the retriever uses a query-constructing LLM chain to generate a structured query. It then applies this structured query to its underlying vector store. This enables the retriever to not only use the user-input query for semantic similarity comparison with the contents of stored documents but also to extract and apply filters based on the metadata of those documents.


The following code demonstrates how to use a Self-Querying Retriever.



```python
from langchain_core.documents import Document
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from lark import lark
```

A couple of document pieces have been prepared where the `page_content` contains descriptions of movies, and the `meta_data` includes different attributes for each movie, such as `year`, `rating`, `genre`, and `director`. These attributes are crucial in the Self-Querying Retriever, as the LLM will use the metadata information to apply filters during the retrieval process.



```python
docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "thriller",
            "rating": 9.9,
        },
    ),
]
```

Now you can instantiate your retriever. To do this, you'll need to provide some upfront information about the metadata fields that your documents support, as well as a brief description of the document contents.



```python
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]
```

Store the document's embeddings into a vector database.



```python
vectordb = Chroma.from_documents(docs, watsonx_embedding())
```

Use the `SelfQueryRetriever`.



```python
document_content_description = "Brief summary of a movie."

retriever = SelfQueryRetriever.from_llm(
    llm(),
    vectordb,
    document_content_description,
    metadata_field_info,
)
```

Now you can actually try using your retriever.



```python
# This example only specifies a filter
retriever.invoke("I want to watch a movie rated higher than 8.5")
```


```python
# This example specifies a query and a filter
retriever.invoke("Has Greta Gerwig directed any movies about women")
```

When running the following cell, you might encounter some errors or blank content. This is because the LLM cannot get the answer at first. Don't worry; if you re-run it several times, you will get the answer.



```python
# This example specifies a composite filter
retriever.invoke("What's a highly rated (above 8.5) science fiction film?")
```

### Parent Document Retriever


When splitting documents for retrieval, there are often conflicting desires:

1. You may want to have small documents so that their embeddings can most accurately reflect their meaning. If the documents are too long, the embeddings can lose meaning.
2. You want to have long enough documents so that the context of each chunk is retained.

The `ParentDocumentRetriever` strikes that balance by splitting and storing small chunks of data. During retrieval, it first fetches the small chunks but then looks up the parent IDs for those chunks and returns those larger documents.



```python
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain.storage import InMemoryStore
```


```python
# Set two splitters. One is with big chunk size (parent) and one is with small chunk size (child)
parent_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=20, separator='\n')
child_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20, separator='\n')
```


```python
vectordb = Chroma(
    collection_name="split_parents", embedding_function=watsonx_embedding()
)

# The storage layer for the parent documents
store = InMemoryStore()
```


```python
retriever = ParentDocumentRetriever(
    vectorstore=vectordb,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
```


```python
retriever.add_documents(chunks_txt)
```

These are the number of large chunks.



```python
len(list(store.yield_keys()))
```

Let's make sure the underlying vector store still retrieves the small chunks.



```python
sub_docs = vectordb.similarity_search("smoking policy")
```


```python
print(sub_docs[0].page_content)
```

Then, retrieve the relevant large chunk.



```python
retrieved_docs = retriever.invoke("smoking policy")
print(retrieved_docs[0].page_content)
```

# Exercises


### Exercise 1
### Retrieve top 2 results using vector store-backed retriever

Can you retrieve the top two results for the company policy document for the query "smoking policy" using the Vector Store-Backed Retriever?



```python
# Your code here
```

<details>
    <summary>Click here for Solution</summary>

```python

vectordb = Chroma.from_documents(documents=chunks_txt, embedding=watsonx_embedding())
retriever = vectordb.as_retriever(search_kwargs={"k": 2})
query = "smoking policy"
docs = retriever.invoke(query)
docs
```

</details>


### Exercise 2
### Self-Querying Retriever for a query


Can you use the Self Querying Retriever to invoke a query with a filter?



```python
# Your code here
```

<details>
    <summary>Click here for a Solution</summary>

```python

# You might encouter some errors or blank content when run the following code.
# It is becasue LLM cannot get the answer at first. Don't worry, re-run it several times you will get the answer.

vectordb = Chroma.from_documents(docs, watsonx_embedding())

retriever = SelfQueryRetriever.from_llm(
    llm(),
    vectordb,
    document_content_description,
    metadata_field_info,
)

# This example specifies a query with filter
retriever.invoke(
    "I want to watch a movie directed by Christopher Nolan"
)
```

</details>


## Authors


[Kang Wang](https://author.skills.network/instructors/kang_wang)

Kang Wang is a Data Scientist in IBM. He is also a PhD Candidate in the University of Waterloo.


### Other Contributors


[Joseph Santarcangelo](https://author.skills.network/instructors/joseph_santarcangelo)

Joseph has a Ph.D. in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.


```{## Change Log}
```


```{|Date (YYYY-MM-DD)|Version|Changed By|Change Description||-|-|-|-||2024-07-29|0.1|Kang Wang|Create the lab|}
```


Copyright © IBM Corporation. All rights reserved.

