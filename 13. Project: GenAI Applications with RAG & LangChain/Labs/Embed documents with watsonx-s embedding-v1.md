# **Embed documents using watsonx's embedding model**


Estimated time needed: **30** minutes


## Overview


Imagine you work in a company that handles a vast amount of text data, including documents, emails, and reports. Your task is to build an intelligent search system that can quickly and accurately retrieve relevant documents based on user queries. Traditional keyword-based search methods often fail to understand the context and semantics of the queries, leading to poor search results.

To address this challenge, you can use embedding models to convert documents into numerical vectors. These vectors capture the semantic meaning of the text, enabling more accurate and context-aware search capabilities. Document embedding is a powerful technique to convert textual data into numerical vectors, which can then be used for various downstream tasks such as search, classification, clustering, and more.


<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/O_XVADBArH0ck4Wf6uuoBA/embeddings.png" width="60%" alt="embeddings">


In this lab, you will learn how to use embedding models from watsonx.ai and Hugging Face to embed documents. By the end of this lab, you will be able to effectively use these embedding models to transform and utilize textual data in your projects.


## __Table of Contents__

<ol>
    <li><a href="#Objectives">Objectives</a></li>
    <li>
        <a href="#Setup">Setup</a>
        <ol>
            <li><a href="#Installing-required-ribraries">Installing required libraries</a></li>
            <li><a href="#Load-data">Load data</a></li>
            <li><a href="#Split data">Split data</a></li>
        </ol>
    </li>
    <li>
        <a href="#Watsonx-embedding-model">Watsonx embedding model</a>
        <ol>
            <li><a href="#Model-description">Model description</a></li>
            <li><a href="#Build-model">Build model</a></li>
            <li><a href="#Query-embeddings">Query embeddings</a></li>
            <li><a href="#Document-embeddings">Document embeddings</a></li>
        </ol>
    </li>
    <li>
        <a href="#HuggingFace embedding model">HuggingFace embedding models</a>
        <ol>
            <li><a href="#Model-description">Model description</a></li>
            <li><a href="#Build-model">Build model</a></li>
            <li><a href="#Query-embeddings">Query embeddings</a></li>
            <li><a href="#Document-embeddings">Document embeddings</a></li>
        </ol>
    </li>
</ol>

<a href="#Exercises">Exercises</a>
<ol>
    <li><a href="#Exercise-1---Using-another-watsonx-embedding-model">Exercise 1. Using another watsonx embedding model</a></li>
</ol>


## Objectives

After completing this lab, you will be able to:

 - Prepare and preprocess documents for embedding
 - Use watsonx.ai and Hugging Face embedding models to generate embeddings for your documents


----


## Setup


For this lab, you will use the following libraries:

* [`ibm-watson-ai`](https://ibm.github.io/watsonx-ai-python-sdk/fm_embeddings.html#EmbeddingModels) for using embedding models from IBM's watsonx.ai.
* [`langchain`, `langchain-ibm`, `langchain-community`](https://www.langchain.com/) for using relevant features from LangChain.
* [`sentence-transformers`](https://huggingface.co/sentence-transformers) for using embedding models from HuggingFace.


### Installing required libraries

The following required libraries are __not__ preinstalled in the Skills Network Labs environment. __You need to run the following cell__ to install them:

**Note:** The version is being pinned here to specify the version. It's recommended that you do this as well. Even if the library is updated in the future, the installed library could still support this lab work.

This might take around 1-2 minutes. 

As `%%capture` is used to capture the installation, you won't see the output process. But after the installation completes, you will see a number beside the cell.



```python
%%capture
#After executing the cell,please RESTART the kernel and run all the cells.
!pip install --user "ibm-watsonx-ai==1.1.2"
!pip install --user "langchain==0.2.11"
!pip install --user "langchain-ibm==0.1.11"
!pip install --user "langchain-community==0.2.10"
!pip install --user "sentence-transformers==3.0.1"
```

After you install the libraries, restart your kernel. You can do that by clicking the **Restart the kernel** icon.

<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/build-a-hotdog-not-hotdog-classifier-guided-project/images/Restarting_the_Kernel.png" width="50%" alt="Restart kernel">


## Load data


A text file has been prepared as the source document for the downstream embedding task.

Now, let's download and load it using LangChain's `TextLoader`.



```python
!wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/i5V3ACEyz6hnYpVq6MTSvg/state-of-the-union.txt"
```


```python
from langchain_community.document_loaders import TextLoader
```


```python
loader = TextLoader("state-of-the-union.txt")
data = loader.load()
```

Let's take a look at the document.



```python
data
```

## Split data


Since the embedding model has a maximum input token limit, you cannot input the entire document at once. Instead, you need to split it into chunks.

The following code shows how to use LangChain's `RecursiveCharacterTextSplitter` to split the document into chunks.
- Use the default separator list, which is `["\n\n", "\n", " ", ""]`.
- Chunk size is set to `100`. This should be set to less than the model's maximum input token.
- Chunk overlap is set to `20`.
- The length function is `len`.



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
chunks = text_splitter.split_text(data[0].page_content)
```

Let's see how many chunks you get.



```python
len(chunks)
```

Let's also see what these chunks looks like.



```python
chunks
```

## Watsonx embedding model


### Model description


In this section, you will use IBM `slate-125m-english-rtrvr` model as an example embedding model.

The slate.125m.english.rtrvr model is a [standard sentence](https://www.sbert.net/) transformers model based on bi-encoders. The model produces an embedding for a given input, e.g., query, passage, document, etc. At a high level, the model is trained to maximize the cosine similarity between two input pieces of text, e.g., text A (query text) and text B (passage text), which results in the sentence embeddings q and p.These sentence embeddings can be compared using cosine similarity, which measures the distance between sentences by calculating the distance between their embeddings.


<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/NDCHhZfcC96jggb2hMdJhg/fm-slate-125m-english-rtrvr-cosine.jpg" width="50%">


The embedding model, `slate.125m.english` formerly known as WatBERT, has the same architecture as a RoBERTa base transformer model and has ~125 million parameters and an embedding dimension of `768`.


|Model name|API model_id|Maximum input tokens|Number of dimensions|More information|
|-|-|-|-|-|
|slate-125m-english-rtrvr|ibm/slate-125m-english-rtrvr|512|768|[model card](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-slate-125m-english-rtrvr-model-card.html?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Embed+documents+with+watsonx%E2%80%99s+embedding_v1_1721662184&context=wx)|


### Build model


The following code shows how to build the `slate-125m-english-rtrvr` model from IBM watsonx.ai API.


First, import the necessary dependencies. 
- `WatsonxEmbeddings` is a class/dependence that can be used to form an embedding model object.
- `EmbedTextParamsMetaNames` is a dependence that controls the embedding parameters.



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

### Query embeddings


Now, create an embedding based on a single sentence, which can be treated as a query.


Use the `embed_query` method.



```python
query = "How are you?"

query_result = watsonx_embedding.embed_query(query)
```

Let's see the length/dimension of this embedding.



```python
len(query_result)
```

It has a dimension of `768`, which aligns with the model description. 


Next, take a look at the first five results from the embeddings.



```python
query_result[:5]
```

### Document embeddings


After creating the query embeddings, you will be guided on how to create embeddings from documents, which are a list a text chunks.


Use `embed_documents`. The parameter `chunks` should be a list of text. Here, chunks is a list of documents you get from before after splitting the whole document.



```python
doc_result = watsonx_embedding.embed_documents(chunks)
```

As each piece of text is embedded into a vector, so the length of the `doc_result` should be the same as the length of chunks.



```python
len(doc_result)
```

Now, take a look at the first five results from the embeddings of the first piece of text.



```python
doc_result[0][:5]
```

Check the embedding dimension to see if it is also 768.



```python
len(doc_result[0])
```

## Hugging Face embedding model


### Model description


In this section, you will use the `all-mpnet-base-v2` from HuggingFace as an example embedding model.

It is a sentence-transformers model. It maps sentences and paragraphs to a 768-dimensional dense vector space and can be used for tasks like clustering or semantic search. It used the pre-trained `Microsoft/money-base` model and fine-tuned it on a 1B sentence pairs dataset. For more information, please refer to [here](https://huggingface.co/sentence-transformers/all-mpnet-base-v2).


### Build model


To build the model, you need to import the `HuggingFaceEmbeddings` dependence first.



```python
from langchain_community.embeddings import HuggingFaceEmbeddings
```

Then, you specify the model name.



```python
model_name = "sentence-transformers/all-mpnet-base-v2"
```

Here we create a embedding model object.



```python
huggingface_embedding = HuggingFaceEmbeddings(model_name=model_name)
```

### Query embeddings


Let's create the embeddings from the same sentence, but using the Hugging Face embedding model. 



```python
query = "How are you?"
```


```python
query_result = huggingface_embedding.embed_query(query)
```


```python
query_result[:5]
```

Do you see the differences between embeddings that are created by the watsonx embedding model and the Hugging Face embedding model?


### Document embeddings


Next, you can do the same for creating embeddings from documents.



```python
doc_result = huggingface_embedding.embed_documents(chunks)
doc_result[0][:5]
```


```python
len(doc_result[0])
```

# Exercises


### Exercise 1 - Using another watsonx embedding model
Watsonx.ai also supports other embedding models, for which you can find more information [here](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-slate-30m-english-rtrvr-model-card.html?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Embed+documents+with+watsonx%E2%80%99s+embedding_v1_1721662184&context=wx). Can you try to use another embedding model to create embeddings for the document?



```python
# Your code here
```

<details>
    <summary>Click here for Solution</summary>

```python

from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings

embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}

watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-30m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=embed_params,
)

doc_result = watsonx_embedding.embed_documents(chunks)

doc_result[0][:5]

```

</details>


## Authors


[Kang Wang](https://author.skills.network/instructors/kang_wang)

Kang Wang is a Data Scientist in IBM. He is also a PhD Candidate in the University of Waterloo.


### Other Contributors


[Joseph Santarcangelo](https://author.skills.network/instructors/joseph_santarcangelo)

Joseph has a Ph.D. in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.

[Hailey Quach](https://author.skills.network/instructors/hailey_quach)

Hailey is a Data Scientist Intern at IBM. She is also pursuing a BSc in Computer Science, Honors at Concordia University, Montreal.


```{## Change Log}
```


```{|Date (YYYY-MM-DD)|Version|Changed By|Change Description||-|-|-|-||2024-07-22|0.1|Kang Wang|Create the lab|}
```


Copyright © IBM Corporation. All rights reserved.


