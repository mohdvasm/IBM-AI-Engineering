# **Apply Text Splitting Techniques to Enhance Model Responsiveness**


Estimated time needed: **30** minutes


## Overview


In many data processing tasks, especially those involving large documents, breaking down text into smaller, more manageable chunks is essential. Text splitters are tools specifically designed to accomplish this, ensuring that lengthy texts are divided into coherent segments. This division is crucial for maintaining the integrity and readability of the information, making it easier to handle and process. Effective text splitting helps prevent overwhelming systems with large, unwieldy blocks of text and ensures that each segment remains meaningful and contextually relevant.

The significance of text splitters becomes even more apparent in the context of retrieval-augmented generation (RAG). RAG involves fetching relevant pieces of information from a large dataset and using them to generate accurate and context-aware responses. Without properly split text, the retrieval process can become inefficient, potentially missing critical pieces of information or returning irrelevant data. By using text splitters to create well-defined chunks, the retrieval process can be streamlined, ensuring that the most relevant information is easily accessible. This not only enhances the efficiency of data retrieval but also improves the quality and relevance of the generated responses, making text splitters an important tool in the RAG workflow.


<figure>
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/3Y4eB0v7LXU5cSVbeN1b8A/text-splitter.png" width="50%" alt="langchain">
    <figcaption><a>source: DALL-E</a></figcaption>
</figure>


This lab will guide you about how to use some commonly used text splitters from LangChain to split your source document.


## __Table of Contents__

<ol>
    <li><a href="#Objectives">Objectives</a></li>
    <li>
        <a href="#Setup">Setup</a>
        <ol>
            <li><a href="#Installing-required-libraries">Installing required libraries</a></li>
        </ol>
    </li>
    <li>
        <a href="#Text-splitters">Text splitters</a>
        <ol>
            <li><a href="#Key-parameters">Key parameters</a></li>
            <li><a href="#Prepare-the-document">Prepare the document</a></li>
            <li><a href="#Split-by-Character">Split by Character</a></li>
            <li><a href="#Recursively-Split-by-Character">Recursively Split by Character</a></li>
            <li><a href="#Split-Code">Split Code</a></li>
            <li><a href="#Markdown-Header-Text-Splitter">Markdown Header Text Splitter</a></li>
            <li><a href="#Split-by-HTML">Split by HTML</a></li>
        </ol>
    </li>
</ol>

<a href="#Exercises">Exercises</a>
<ol>
    <li><a href="#Exercise-1---Changing-separator-for-CharacterTextSplitter">Exercise 1. Changing separator for CharacterTextSplitter</a></li>
    <li><a href="#Exercise-2---Splitting-Latex-code">Exercise 2. Splitting Latex code</a></li>
</ol>


## Objectives

After completing this lab you will be able to:

 - Use commonly used text splitters from LangChain.
 - Split source documents into chunks for downstream use in RAG


----


## Setup


For this lab, you will use the following libraries:

*   [`langchain`, `langchain-text-splitters`](https://www.langchain.com/) for using relevant features and text splitters from Langchain.
*   [`lxml`](https://pypi.org/project/lxml/) for libxml2 and libxslt libraries, which is used for splitting html text.


### Installing required libraries

The following required libraries are __not__ preinstalled in the Skills Network Labs environment. __You must run the following cell__ to install them:

**Note:** The version is being pinned here to specify the version. It's recommended that you do this as well. Even if the library is updated in the future, the installed library could still support this lab work.

This might take approximately 1 minute. 

As `%%capture` is used to capture the installation, you won't see the output process. But after the installation completes, you will see a number beside the cell.



```python
%%capture
!pip install "langchain==0.2.7"
!pip install "langchain-core==0.2.20"
!pip install "langchain-text-splitters==0.2.2"
!pip install "lxml==5.2.2"
```

After you install the libraries, restart your kernel. You can do that by clicking the **Restart the kernel** icon.

<p style="text-align:left">
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/1_Bd_EvpEzLH9BbxRXXUGQ/screenshot-to-replace.png" width="50%"/>
    </a>
</p>


## Text splitters


### Key parameters


When using the splitter, you can customize several key parameters to fit your needs:
- **separator**: Define the characters that will be used for splitting the text.
- **chunk_size**: Specify the maximum size of your chunks to ensure they are as granular or broad as needed.
- **chunk_overlap**: Maintain context between chunks by setting the `chunk_overlap` parameter, which determines the number of characters that overlap between consecutive chunks. This helps ensure that information isn't lost at the chunk boundaries.
- **length_function**: Define how the length of chunks is calculated.


### Prepare the document


A long document has been prepared for this project to demonstrate the performance of each splitter. Run the following code to download it.



```python
!wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/YRYau14UJyh0DdiLDdzFcA/companypolicies.txt"
```

Let's take a look at what the document looks like.



```python
# This is a long document you can split up.
with open("companypolicies.txt") as f:
    companypolicies = f.read()
```


```python
print(companypolicies)
```

It is a long document about a company's policies.


### Document object


Before introducing the splitters, let's take a look at the document object in LangChain, which is a data structure used to represent and manage text data in RAG process.

A Document object in LangChain contains information about some data. It has two attributes:

- `page_content: str`: The content of this document. Currently is only a string.
- `metadata: dict`: Arbitrary metadata associated with this document. Can track the document id, file name, etc.


Here, you can use an example to guide you through creating a document object. Langchain uses this object type to deal with text/documents.



```python
from langchain_core.documents import Document
Document(page_content="""Python is an interpreted high-level general-purpose programming language. 
                        Python's design philosophy emphasizes code readability with its notable use of significant indentation.""",
         metadata={
             'my_document_id' : 234234,
             'my_document_source' : "About Python",
             'my_document_create_time' : 1680013019
         })
```

### Split by Character


This is the simplest method, which splits the text based on characters (by default `"\n\n"`) and measures chunk length by the number of characters.
- **How the text is split**: By single character.
- **How the chunk size is measured**: By number of characters.


Let's see how to implement this method using code.


In the following code, you will use `CharacterTextSplitter` to split the document by character. 
- Separator: Set to `''`, meaning that any character can act as a separator once the chunk size reaches the set limit.
- Chunk size: Set to `200`, meaning that once a chunk reaches 200 characters, it will be split.
- Chunk overlap: Set to `20`, meaning there will be `20` characters overlapping between chunks.
- Length function: Set to `len`.



```python
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
)
```

You will use `split_text` function to operate the split.



```python
texts = text_splitter.split_text(companypolicies)
```

Let's take a look how the document has been split.



```python
texts
```

After the split, you'll see that the document has been divided into multiple chunks, with some character overlaps between the chunks.

You can see how many chunks you get.



```python
len(texts)
```

You get `87` chunks.


You can also use the following code to add metadata to the text, forming it into a `document` object using LangChain.



```python
texts = text_splitter.create_documents([companypolicies], metadatas=[{"document":"Company Policies"}])  # pass the metadata as well
```


```python
texts[0]
```

### Recursively Split by Character


This text splitter is the recommended one for generic text. It is parameterized by a list of characters, and it tries to split on them in order until the chunks are small enough. The default list is `["\n\n", "\n", " ", ""]`.

It processes the large text by attempting to split it by the first character, `\n\n`. If the first split by \n\n results in chunks that are still too large, it moves to the next character, `\n`, and attempts to split by it. This process continues through the list of characters until the chunks are less than the specified chunk size.

This method aims to keep all paragraphs (then sentences, then words) together as much as possible, as these are generally the most semantically related pieces of text.

- **How the text is split**: by list of characters.
- **How the chunk size is measured**: by number of characters.


The following code is showing how to implement it.


The `RecursiveCharacterTextSplitter` class from LangChain is used to implement it.
- You use the default separator list, which is `["\n\n", "\n", " ", ""]`.
- Chunk size is set to `100`.
- Chunk overlap is set to `20`.
- And the length function is `len`.



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

Here, you are using the same document "companypolicies.txt" from earlier as an example to show the performance of `RecursiveCharacterTextSplitter`.



```python
texts = text_splitter.create_documents([companypolicies])
```


```python
texts
```

From the split results, you can see that the splitter uses recursion as the core strategy to divide the document into chunks.


You can also see how many chunks you get.



```python
len(texts)
```

You get `215` chunks.


### Split Code


The `CodeTextSplitter` allows you to split your code, supporting multiple programming languages. It is based on the `RecursiveCharacterTextSplitter` strategy. Simply import enum `Language` and specify the language.



```python
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
```

Use the following to see a list of codes it supports.



```python
[e.value for e in Language]
```

Use the following code to see what default separators it uses, for example, for Python.



```python
RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)
```

#### Python


The following demonstrates how to split Python code using the `RecursiveCharacterTextSplitter` class.

The main difference between splitting code and using the original `RecursiveCharacterTextSplitter` is that you need to call `.from_language` after the `RecursiveCharacterTextSplitter` and specify the `language`. The other parameter settings remain the same as before.



```python
PYTHON_CODE = """
    def hello_world():
        print("Hello, World!")
    
    # Call the function
    hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
python_docs = python_splitter.create_documents([PYTHON_CODE])
python_docs
```

#### Javascript


Let's see the separators for JSON language.



```python
RecursiveCharacterTextSplitter.get_separators_for_language(Language.JS)
```

The following code is used to separate the JSON language code.



```python
JS_CODE = """
    function helloWorld() {
      console.log("Hello, World!");
    }
    
    // Call the function
    helloWorld();
"""

js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, chunk_size=60, chunk_overlap=0
)
js_docs = js_splitter.create_documents([JS_CODE])
js_docs
```

For more information about applying to other languages, you can refer [here](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/code_splitter/).


### Markdown Header Text Splitter


As mentioned, chunking often aims to keep text with a common context together. With this in mind, you might want to specifically honor the structure of the document itself. For example, a Markdown file is organized by headers. Creating chunks within specific header groups is an intuitive approach.

To address this challenge, you can use `MarkdownHeaderTextSplitter`. This splitter will divide a Markdown file based on a specified set of headers.



```python
from langchain.text_splitter import MarkdownHeaderTextSplitter
```

For example, if you want to split this markdown:



```python
md = "# Foo\n\n## Bar\n\nHi this is Jim\n\nHi this is Joe\n\n### Boo \n\nHi this is Lance \n\n## Baz\n\nHi this is Molly"
```

You can specify the headers to split on:



```python
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
```


```python
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(md)
md_header_splits
```

From the split results, you can see that the Markdown file is divided into several chunks formatted as document objects. The `page_content` contains the text under the headings, and the `metadata` contains the header information corresponding to the `page_content`.


If you want the headers appears in the page_content as well, you can specify `strip_headers=False` when you call the `MarkdownHeaderTextSplitter`.



```python
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
md_header_splits = markdown_splitter.split_text(md)
md_header_splits
```

### Split by HTML


#### Split by HTML header


Similar in concept to the `MarkdownHeaderTextSplitter`, the HTMLHeaderTextSplitter is a "structure-aware" chunker that splits text at the element level and adds metadata for each header "relevant" to any given chunk. It can return chunks element by element or combine elements with the same metadata, with the objectives of (a) keeping related text grouped (more or less) semantically and (b) preserving context-rich information encoded in document structures. It can be used with other text splitters as part of a chunking pipeline.



```python
from langchain_text_splitters import HTMLHeaderTextSplitter
```

Assume you have the following HTML code that you want to split.



```python
html_string = """
    <!DOCTYPE html>
    <html>
    <body>
        <div>
            <h1>Foo</h1>
            <p>Some intro text about Foo.</p>
            <div>
                <h2>Bar main section</h2>
                <p>Some intro text about Bar.</p>
                <h3>Bar subsection 1</h3>
                <p>Some text about the first subtopic of Bar.</p>
                <h3>Bar subsection 2</h3>
                <p>Some text about the second subtopic of Bar.</p>
            </div>
            <div>
                <h2>Baz</h2>
                <p>Some text about Baz</p>
            </div>
            <br>
            <p>Some concluding text about Foo</p>
        </div>
    </body>
    </html>
"""
```

Set up the header to split.



```python
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]
```

Split the HTML string using `HTMLHeaderTextSplitter`.



```python
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
html_header_splits = html_splitter.split_text(html_string)
html_header_splits
```

From the split results, you can see that the context under the headings is extracted and put in the `page_content` parameter. The `metatdata` contains the header information.


#### Split by HTML section


Similar to the `HTMLHeaderTextSplitter`, the `HTMLSectionSplitter` is also a "structure-aware" chunker that splits text section by section based on headings.


The following code is used to implement it.



```python
from langchain_text_splitters import HTMLSectionSplitter

html_string = """
    <!DOCTYPE html>
    <html>
    <body>
        <div>
            <h1>Foo</h1>
            <p>Some intro text about Foo.</p>
            <div>
                <h2>Bar main section</h2>
                <p>Some intro text about Bar.</p>
                <h3>Bar subsection 1</h3>
                <p>Some text about the first subtopic of Bar.</p>
                <h3>Bar subsection 2</h3>
                <p>Some text about the second subtopic of Bar.</p>
            </div>
            <div>
                <h2>Baz</h2>
                <p>Some text about Baz</p>
            </div>
            <br>
            <p>Some concluding text about Foo</p>
        </div>
    </body>
    </html>
"""

headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")]

html_splitter = HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)
html_header_splits = html_splitter.split_text(html_string)
html_header_splits
```

# Exercises


### Exercise 1 - Changing separator for CharacterTextSplitter


Try to change to use another separator, for example `"\n"` to see how it affect the split and chunks.



```python
# Your code here
```

<details>
    <summary>Click here for Solution</summary>

```python
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
)
texts = text_splitter.split_text(companypolicies)
texts
```

</details>


### Exercise 2 - Splitting Latex code


Here is an example of Latex code. Try to split it.



```python
latex_text = """
    \documentclass{article}
    
    \begin{document}
    
    \maketitle
    
    \section{Introduction}
    Large language models (LLMs) are a type of machine learning model that can be trained on vast amounts of text data to generate human-like language. In recent years, LLMs have made significant advances in a variety of natural language processing tasks, including language translation, text generation, and sentiment analysis.
    
    \subsection{History of LLMs}
    The earliest LLMs were developed in the 1980s and 1990s, but they were limited by the amount of data that could be processed and the computational power available at the time. In the past decade, however, advances in hardware and software have made it possible to train LLMs on massive datasets, leading to significant improvements in performance.
    
    \subsection{Applications of LLMs}
    LLMs have many applications in industry, including chatbots, content creation, and virtual assistants. They can also be used in academia for research in linguistics, psychology, and computational linguistics.
    
    \end{document}
"""
```


```python
# Your code here
```

<details>
    <summary>Click here for Solution</summary>

```python
latex_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.LATEX, chunk_size=60, chunk_overlap=0
)
latex_docs = latex_splitter.create_documents([latex_text])
latex_docs
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


```{Date (YYYY-MM-DD)|Version|Changed By|Change Description||-|-|-|-||2024-07-16|0.1|Kang Wang|Create the lab|}
```


© Copyright IBM Corporation. All rights reserved.

