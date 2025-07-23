# **Create and Configure a Vector Database to Store Document Embeddings**


Estimated time needed: **30** minutes


## Overview


Imagine you are working in a customer support center that receives a high volume of inquiries and tickets every day. Your task is to create a system that can quickly provide support agents with the most relevant information to resolve customer issues. Traditional methods of searching through FAQs or support documents can be slow and inefficient, leading to delayed responses and dissatisfied customers.

To address this challenge, you will use embedding models to convert support documents and past inquiry responses into numerical vectors that capture their semantic content. These vectors will be stored in a vector database, enabling fast and accurate similarity searches. For example, when a support agent receives a new inquiry about a product issue, the system can instantly retrieve similar past inquiries and their resolutions, helping the agent to provide a quicker and more accurate response.


<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/veZYoygp9GqZrIw5f6SD0g/vector%20db.png" width="50%" alt="vector db"/>


In this lab, you will learn how to use vector databases to store embeddings generated from textual data using LangChain. The focus will be on two popular vector databases: Chroma DB and FAISS (Facebook AI Similarity Search). You will also learn how to perform similarity searches in these databases based on a query, enabling efficient retrieval of relevant information. By the end of this lab, you will be able to effectively use vector databases to store and query embeddings, enhancing your data analysis and retrieval capabilities.


## __Table of Contents__

<ol>
    <li><a href="#Objectives">Objectives</a></li>
    <li>
        <a href="#Setup">Setup</a>
        <ol>
            <li><a href="#Installing-required-libraries">Installing required libraries</a></li>
            <li><a href="#Load-text">Load text</a></li>
            <li><a href="#Split-data">Split data</a></li>
            <li><a href="#Embedding model">Embedding model</a></li>
        </ol>
    </li>
    <li>
        <a href="#Vector-store">Vector store</a>
        <ol>
            <li><a href="#Chroma-DB">Chroma DB</a></li>
            <li><a href="#FIASS-DB">FIASS DB</a></li>
            <li><a href="#Managing-vector-store:-adding,-updating,-and-deleting-entries">Managing vector store: adding, updating, and deleting entries</a></li>
        </ol>
    </li>
</ol>

<a href="#Exercises">Exercises</a>
<ol>
    <li><a href="#Exercise-1---Use-another-query-to-conduct-similarity-search.">Exercise 1. Use another query to conduct similarity search.</a></li>
</ol>


## Objectives

After completing this lab you will be able to:

- Prepare and preprocess documents for embeddings.
- Generate embeddings using watsonx.ai's embedding model.
- Store these embeddings in Chroma DB and FAISS.
- Perform similarity searches to retrieve relevant documents based on new inquiries.


----


## Setup


For this lab, you will use the following libraries:

* [`ibm-watson-ai`](https://ibm.github.io/watsonx-ai-python-sdk/) for using LLMs from IBM's watsonx.ai.
* [`langchain`, `langchain-ibm`, `langchain-community`](https://www.langchain.com/) for using relevant features from Langchain.
* [`chromadb`](https://www.trychroma.com/) is a open-source vector database used to store embeddings.
* [`faiss-cpu`](https://pypi.org/project/faiss-cpu/) is used to support the using of FAISS vector database.


### Installing required libraries

The following required libraries are __not__ preinstalled in the Skills Network Labs environment. __You must run the following cell__ to install them:

**Note:** The version is being pinned here to specify the version. It's recommended that you do this as well. Even if the library is updated in the future, the installed library could still support this lab work.

This might take approximately 1-2 minutes. 

As `%%capture` is used to capture the installation, you won't see the output process. After the installation is completed, you will see a number beside the cell.



```python
%%capture
!pip install --user"ibm-watsonx-ai==1.0.4"
!pip install  --user "langchain==0.2.1" 
!pip install  --user "langchain-ibm==0.1.7"
!pip install  --user "langchain-community==0.2.1"
!pip install --user "chromadb==0.4.24"
!pip install  --user "faiss-cpu==1.8.0"
```

After you install the libraries, restart your kernel. You can do that by clicking the **Restart the kernel** icon.

<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/build-a-hotdog-not-hotdog-classifier-guided-project/images/Restarting_the_Kernel.png" width="50%" alt="Restart kernel">


-----


The following steps are prerequisite tasks for conducting this project's topic - vector store. These steps include:

- Loading the source document.
- Splitting the document into chunks.
- Building an embedding model.
  
The details of these steps have been introduced in previous lessons.


### Load text


A text file has been prepared as the source document for the downstream vector database task.

Now, let's download and load it using LangChain's `TextLoader`.



```python
!wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/BYlUHaillwM8EUItaIytHQ/companypolicies.txt"
```


```python
from langchain_community.document_loaders import TextLoader
```


```python
loader = TextLoader("companypolicies.txt")
data = loader.load()
```

You can have a look at this document.



```python
data
```

### Split data


The next step is to split the document using LangChain's text splitter. Here, you will use the `RecursiveCharacterTextSplitter, which is well-suited for this generic text. The following parameters have been set:

- `chunk_size = 100`
- `chunk_overlap = 20`
- `length_function = len`



```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
```


```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)
```


```python
chunks = text_splitter.split_documents(data)
```

Let's take a look at how many chunks you get.



```python
len(chunks)
```

So, in total, you get 215 chunks.


### Embedding model


The following code demonstrates how to build an embedding model using the `watsonx.ai` package.

For this project, the `ibm/slate-125m-english-rtrvr` embedding model will be used.



```python
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings
```


```python
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
```

The embedding model is formed into the `watsonx_embedding` object.


## Vector store


In this section, you will be guided on how to use two commonly used vector databases: Chroma DB and FAISS DB. You will also see how to perform a similarity search based on an input query using these databases.


### Chroma DB


#### Build the database


First, you need to import `Chroma` from Langchain vector stores.



```python
from langchain.vectorstores import Chroma
```

Next, you need to create an ID list that will be used to assign each chunk a unique identifier, allowing you to track them later in the vector database. The length of this list should match the length of the chunks.

Note: The IDs should be in string format.



```python
ids = [str(i) for i in range(0, len(chunks))]
```

The next step is to use the embedding model to create embeddings for each chunk and then store them in the Chroma database.

The following code demonstrates how to do this.



```python
vectordb = Chroma.from_documents(chunks, watsonx_embedding, ids=ids)
```

Now that you have built the vector store named `vectordb`, you can use the method `.collection.get()` to print some of the chunks indexed by their IDs.

Note: Although the chunks are stored in the database in embedding format, when you retrieve and print them by their IDs, the database will return the chunk text information instead of the embedding vectors.



```python
for i in range(3):
    print(vectordb._collection.get(ids=str(i)))
```

You can also use the method `._collection.count()` to see the length of the vector database, which should be the same as the length of chunks.



```python
vectordb._collection.count()
```

#### Similarity search


Similarity search in a vector database involves finding items that are most similar to a given query item based on their vector representations.

In this process, data objects are converted into vectors (which you've already done), and the search algorithm identifies and retrieves those with the closest vector distances to the query, enabling efficient and accurate identification of similar items in large datasets.


LangChain supports similarity search in vector stores using the method `.similarity_search()`.

The following is an example of how to perform a similarity search based on the query "Email policy."

By default, it will return the top four closest vectors to the query.



```python
query = "Email policy"
docs = vectordb.similarity_search(query)
docs
```

You can specify `k = 1` to just retrieve the top one result.



```python
vectordb.similarity_search(query, k = 1)
```

### FIASS DB


FIASS is another vector database that is supported by LangChain.

The process of building and using FAISS is similar to Chroma DB.

However, there may be differences in the retrieval results between FAISS and Chroma DB.


#### Build the database


Build the database and store the embeddings to the database here.



```python
from langchain_community.vectorstores import FAISS
```


```python
faissdb = FAISS.from_documents(chunks, watsonx_embedding, ids=ids)
```

Next, print the first three information pieces in the database based on IDs.



```python
for i in range(3):
    print(vectordb._collection.get(ids=str(i)))
```

#### Similarity search


Let's do a similarity search again using FIASS DB on the same query.



```python
query = "Email policy"
docs = faissdb.similarity_search(query)
docs
```

The retrieve results based on the similarity search seem to be the same as with the Chroma DB.

You can try with other queries or documents to see if they follow the same situation.


### Managing vector store: Adding, updating, and deleting entries


There might be situations where new documents come into your RAG application that you want to add to the current vector database, or you might need to delete some existing documents from the database. Additionally, there may be updates to some of the documents in the database that require updating.

The following sections will guide you on how to perform these tasks. You will use the Chroma DB as an example.


#### Add


Imagine you have a new piece of text information that you want to add to the vector database. First, this information should be formatted into a document object.



```python
text = "Instructlab is the best open source tool for fine-tuning a LLM."
```


```python
from langchain_core.documents import Document
```

Form the text into a `Document` object named `new_chunk`.



```python
new_chunk =  Document(
    page_content=text,
    metadata={
        "source": "ibm.com",
        "page": 1
    }
)
```

Then, the new chunk should be put into a list as the vector database only accepts documents in a list.



```python
new_chunks = [new_chunk]
```

Before you add the document to the vector database, since there are 215 chunks with IDs from 0 to 214, if you print ID 215, the document should show no values. Let's validate it.



```python
print(vectordb._collection.get(ids=['215']))
```

Next, you can use the method `.add_documents()` to add this `new_chunk`. In this method, you should assign an ID to the document. Since there are already IDs from 0 to 214, you can assign ID 215 to this document. The ID should be in string format and placed in a list.



```python
vectordb.add_documents(
    new_chunks,
    ids=["215"]
)
```

Now you can count the length of the vector database again to see if it has increased by one.



```python
vectordb._collection.count()
```

You can then print this newly added document from the database by its ID.



```python
print(vectordb._collection.get(ids=['215']))
```

#### Update


Imagine you want to update the content of a document that is already stored in the database. The following code demonstrates how to do this.


Still, you need to form the updated text into a `Document` object.



```python
update_chunk =  Document(
    page_content="Instructlab is a perfect open source tool for fine-tuning a LLM.",
    metadata={
        "source": "ibm.com",
        "page": 1
    }
)
```

Then, you can use the method `.update_document()` to update the specific stored information indexing by its ID.



```python
vectordb.update_document(
    '215',
    update_chunk,
)
```


```python
print(vectordb._collection.get(ids=['215']))
```

As you can see, the document information has been updated.


#### Delete


If you want to delete documents from the vector database, you can use the method `_collection.delete()` and specify the document ID to delete it.



```python
vectordb._collection.delete(ids=['215'])
```


```python
print(vectordb._collection.get(ids=['215']))
```

As you can see, now that document is empty.


# Exercises


### Exercise 1 - Use another query to conduct similarity search.

Can you use another query to conduct the similarity search?



```python
# Your code here
```

<details>
    <summary>Click here for solution</summary>

```python
query = "Smoking policy"
docs = vectordb.similarity_search(query)
docs
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


```{|Date (YYYY-MM-DD)|Version|Changed By|Change Description||-|-|-|-||2024-07-24|0.1|Kang Wang|Create the lab|}
```



Copyright © IBM Corporation. All rights reserved.

