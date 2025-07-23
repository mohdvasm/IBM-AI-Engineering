```python
# !pip -q install gradio ibm-watsonx-ai langchain langchain-community langchain-ibm chromadb pypdf pydantic 
```


```python
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import gradio as gr
from pathlib import Path
import os 

# Importing environment variables 
from dotenv import load_dotenv

load_dotenv()
```




    True




```python
IBM_PROJECT_API_KEY = os.getenv("IBM_PROJECT_API_KEY")
IBM_PROJECT_URL = os.getenv("IBM_PROJECT_URL")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID")
```

### **Task 1**

(This task corresponds with Exercise 1 in the lab “Load Documents Using LangChain for Different Sources” from Module 1)

Capture a screenshot (saved as pdf_loader) that displays both the code used and the first 1000 characters of the content after loading the paper link.


```python
## Document loader
def document_loader(file):
    loader = PyPDFLoader(file)
    loaded_document = loader.load()
    return loaded_document

## Text splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks
```


```python
file = Path("/home/vasim/Khatir/IBM-AI-Engineering/13. Project: GenAI Applications with RAG & LangChain/Notes/A_Comprehensive_Review_of_Low_Rank_Adaptation_in_Large_Language_Models_for_Efficient_Parameter_Tuning-1.pdf")
```


```python
documents = document_loader(file)
print(f"Length of docuemtns: {len(documents)}")
print(f"First 1000 characters in first document:\n")
# print(documents[0].page_content)
print(documents[0].page_content[:1000])
```

    Length of docuemtns: 11
    First 1000 characters in first document:
    
    A Comprehensive Review of Low-Rank
    Adaptation in Large Language Models for
    Efficient Parameter Tuning
    September 10, 2024
    Abstract
    Natural Language Processing (NLP) often involves pre-training large
    models on extensive datasets and then adapting them for specific tasks
    through fine-tuning. However, as these models grow larger, like GPT-3
    with 175 billion parameters, fully fine-tuning them becomes computa-
    tionally expensive. We propose a novel method called LoRA (Low-Rank
    Adaptation) that significantly reduces the overhead by freezing the orig-
    inal model weights and only training small rank decomposition matrices.
    This leads to up to 10,000 times fewer trainable parameters and reduces
    GPU memory usage by three times. LoRA not only maintains but some-
    times surpasses fine-tuning performance on models like RoBERTa, De-
    BERTa, GPT-2, and GPT-3. Unlike other methods, LoRA introduces
    no extra latency during inference, making it more efficient for practical
    applications. All relevant code an


### **Task 2**

(This task corresponds with Exercise 2 in the lab, “Apply text splitting techniques to enhance model responsiveness.”)

Submit a screenshot (saved as ‘code_splitter.png’) that displays the code used to split the following LATEX code and its corresponding results.


```python
latex_text = r"""
\documentclass{article}

\begin{document}

\maketitle

\section{Introduction}

Large language models (LLMs) are a type of machine learning model that can be trained on vast amounts of text data to generate human-like language. In recent years, LLMs have made significant advances in various natural language processing tasks, including language translation, text generation, and sentiment analysis.

\subsection{History of LLMs}

The earliest LLMs were developed in the 1980s and 1990s, but they were limited by the amount of data that could be processed and the computational power available at the time. In the past decade, however, advances in hardware and software have made it possible to train LLMs on massive datasets, leading to significant improvements in performance.

\subsection{Applications of LLMs}

LLMs have many applications in the industry, including chatbots, content creation, and virtual assistants. They can also be used in academia for research in linguistics, psychology, and computational linguistics.

\end{document}
"""
```


```python
from langchain.text_splitter import LatexTextSplitter

text_splitter = LatexTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
        length_function=len,
    )
```


```python
chunks = text_splitter.split_text(latex_text)
chunks
```




    ['\\documentclass{article}\n\n\\begin{document}\n\n\\maketitle\n\n\\section{Introduction}\n\nLarge language',
     'language models (LLMs) are a type of machine learning model that can be trained on vast amounts of',
     'of text data to generate human-like language. In recent years, LLMs have made significant advances',
     'advances in various natural language processing tasks, including language translation, text',
     'text generation, and sentiment analysis.\n\n\\subsection{History of LLMs}\n\nThe earliest LLMs were',
     'LLMs were developed in the 1980s and 1990s, but they were limited by the amount of data that could',
     'could be processed and the computational power available at the time. In the past decade, however,',
     'however, advances in hardware and software have made it possible to train LLMs on massive datasets,',
     'datasets, leading to significant improvements in performance.\n\n\\subsection{Applications of',
     'of LLMs}\n\nLLMs have many applications in the industry, including chatbots, content creation, and',
     'and virtual assistants. They can also be used in academia for research in linguistics, psychology,',
     'and computational linguistics.\n\n\\end{document}']



### **Task 3**

(This task corresponds with Exercise 1 in the lab “Embed documents using watsonx’s embedding model.”)
Submit a screenshot (saved as ‘embedding.png’) that displays the code used to embed the following sentence and its corresponding results, which display the first five embedding numbers.

query = "How are you?"


```python
## Embedding model
def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url=IBM_PROJECT_URL,
        project_id=IBM_PROJECT_ID,
        apikey=IBM_PROJECT_API_KEY,
        params=embed_params,
    )
    return watsonx_embedding

embedding_model = watsonx_embedding()
```


```python
query = "How are you?"
embedding_model.embed_query("How are you?")
```




    [-0.06722455,
     -0.023730014,
     0.017487874,
     -0.013195301,
     -0.03958462,
     0.035013206,
     0.026268989,
     -0.016346067,
     0.0017328461,
     0.057110325,
     -0.07210769,
     -0.017297756,
     -0.049611095,
     -0.017423034,
     -0.028412396,
     -0.008028814,
     0.017073661,
     -0.005280184,
     0.025419915,
     -0.033137515,
     0.009124627,
     0.0023437547,
     0.0027022033,
     -0.02301458,
     -0.012872511,
     0.011373145,
     -0.009063827,
     -0.033579566,
     0.0021336055,
     0.039742704,
     -0.007008723,
     0.017540192,
     0.013501204,
     -0.02065632,
     0.03194266,
     0.01087044,
     -0.008687383,
     0.023959257,
     0.0007889831,
     -0.007972035,
     -0.028410012,
     -0.00958747,
     -0.027156787,
     -0.022130527,
     -0.002896206,
     -0.01690262,
     0.077346265,
     0.010192934,
     0.044747256,
     0.016257787,
     0.06652342,
     -0.009298826,
     0.05224959,
     0.0096178325,
     0.047355823,
     -0.038022693,
     0.016534382,
     0.018556956,
     -0.010481163,
     0.0061104945,
     -0.01024595,
     0.051244814,
     -0.05585934,
     -0.008510807,
     0.028069673,
     0.011818047,
     -0.040075816,
     0.029689154,
     -0.08787459,
     0.024186041,
     -0.06456677,
     -0.007604542,
     0.016543629,
     -0.05843332,
     -0.03482226,
     0.027720012,
     -0.032682337,
     -0.027742345,
     -0.024907205,
     -0.0065533523,
     0.023166396,
     -0.0041369293,
     0.015799554,
     0.0019043316,
     0.01980828,
     -0.0018761457,
     -0.018335285,
     -0.02799592,
     -0.03493484,
     -0.013229756,
     0.0328185,
     -0.035363022,
     0.02849952,
     -0.051070668,
     0.01573663,
     0.009241733,
     0.023862498,
     0.043013517,
     0.030826723,
     0.024911858,
     0.06197857,
     -0.021643346,
     0.036210053,
     -0.031883724,
     0.028464895,
     0.052406855,
     -0.034159165,
     0.026003513,
     -0.012133356,
     -0.0033279047,
     0.0723308,
     -0.07607224,
     -0.045392875,
     0.026521059,
     0.047667827,
     0.038230803,
     0.00895217,
     -0.05222689,
     0.041690882,
     -0.04328276,
     -0.07080701,
     -0.04203314,
     -0.07632123,
     -0.06832437,
     0.054295316,
     0.029673688,
     0.041617572,
     0.03529861,
     0.021892177,
     -0.010355221,
     -0.0073439484,
     0.016643204,
     0.020906968,
     -0.01809326,
     -0.005790205,
     -0.020870486,
     0.015880644,
     0.042151492,
     -0.019949654,
     -0.011566116,
     0.017166026,
     -0.0039207647,
     0.02139079,
     -0.0005141056,
     -0.028978357,
     0.028017879,
     0.020853555,
     -0.07844372,
     -0.03266978,
     -0.0041859616,
     -0.037841916,
     -0.052457806,
     0.021955173,
     0.060466055,
     -0.027981348,
     -0.02544453,
     0.020927139,
     0.045826852,
     0.006870324,
     -0.015889449,
     0.030123828,
     0.0077857315,
     -0.04010277,
     0.007152381,
     0.032464925,
     0.012487702,
     -0.034773868,
     -0.018295867,
     -0.0012087274,
     -0.0716737,
     0.024570704,
     -0.009726623,
     -0.038413882,
     -0.019947482,
     -0.03316492,
     -0.07943936,
     0.010327834,
     -0.04096532,
     0.018062698,
     0.040272932,
     0.09916281,
     -0.014621517,
     -0.025262313,
     0.01583892,
     -0.02092535,
     0.06710146,
     -0.045359038,
     -0.005166119,
     0.017859137,
     0.017828148,
     -0.036870822,
     0.01870715,
     0.02599318,
     0.004818808,
     0.037456136,
     -0.010864479,
     0.030485863,
     -0.021875963,
     0.0009316719,
     -0.021685522,
     -0.030767532,
     0.03251513,
     -0.04346945,
     -0.02479172,
     0.018508064,
     -0.025016826,
     -0.035899576,
     0.037892688,
     0.04362186,
     -0.05932239,
     -0.004409164,
     -0.0032594816,
     -0.015706502,
     -0.28357172,
     -0.017650302,
     0.016408194,
     -0.13232796,
     -0.00540266,
     0.009985626,
     0.0086471625,
     -0.017288657,
     0.011719155,
     -0.009293179,
     0.005329977,
     -0.014740604,
     -0.032484923,
     0.029534033,
     -0.04015033,
     -0.04166391,
     0.026231213,
     0.018610734,
     -0.029331751,
     0.019705856,
     0.023222715,
     -0.02647245,
     -0.062451817,
     0.0402493,
     -0.009222628,
     -0.017783199,
     0.028875403,
     -0.043873124,
     -0.009172994,
     -0.000115537616,
     0.064451724,
     0.018526984,
     -0.00066964555,
     -0.0072787115,
     -0.013408208,
     0.0119286375,
     -0.054093853,
     -0.0187989,
     -0.013029432,
     -0.24088807,
     -0.0006449119,
     -0.023697024,
     0.045447577,
     0.022037921,
     0.020504616,
     0.024958545,
     -0.0034134348,
     0.017239017,
     0.0021879787,
     -0.009537018,
     0.031147791,
     -0.04702478,
     -0.0036239019,
     0.017840983,
     0.0034490011,
     0.03137267,
     0.029044393,
     -0.04737335,
     0.0102209905,
     0.06280561,
     0.06503778,
     0.009405432,
     -0.03909948,
     0.031956792,
     -0.0130366,
     -0.0137541955,
     0.003991437,
     0.028008552,
     -0.003198378,
     0.03407933,
     -0.019345915,
     -0.022447893,
     -0.040087443,
     0.019696336,
     -0.022922149,
     -3.405552e-05,
     0.026170118,
     -0.020740952,
     0.0028476492,
     0.029027529,
     -0.033788316,
     0.02567077,
     0.0548394,
     0.0065655876,
     -0.016915213,
     -0.03594947,
     0.010433701,
     -0.009507896,
     -0.043013282,
     0.028793935,
     -0.0011624726,
     -0.02362253,
     -0.03344909,
     0.023946533,
     -0.008226001,
     -0.014730333,
     0.029443325,
     0.0072977426,
     -0.03704814,
     -0.04662152,
     -0.015566327,
     0.026013682,
     -0.017892005,
     0.006063944,
     0.002325516,
     0.020134458,
     0.0037855518,
     -0.02025993,
     0.019973226,
     0.06203648,
     0.046641752,
     -0.020340301,
     0.021675797,
     0.07472237,
     0.025873095,
     0.042626023,
     -0.007866032,
     -0.07048584,
     -0.007841892,
     -0.0017895318,
     0.0060411384,
     0.025519198,
     -0.026332607,
     0.021141766,
     -0.019905437,
     -0.03667538,
     0.034737784,
     0.022205692,
     0.05096014,
     0.002102702,
     -0.02878163,
     -0.030223254,
     -0.008009609,
     -0.00859737,
     -0.08181258,
     0.022011725,
     0.037582982,
     0.06818832,
     -0.022083152,
     0.009199552,
     0.012032961,
     -0.036478195,
     -0.06414286,
     0.013720091,
     0.027904183,
     -0.035400543,
     -0.02965274,
     -0.013544993,
     0.03268436,
     0.018884374,
     -0.035893142,
     0.040075712,
     -0.0042887535,
     0.0065071136,
     -0.03513518,
     0.019213328,
     -0.0051839775,
     -0.022963962,
     -0.020977192,
     0.025904866,
     0.022376198,
     -0.027651096,
     -0.014076232,
     0.030928884,
     -0.07935061,
     -0.012231404,
     0.0020059056,
     0.0033196122,
     -0.047333054,
     0.04103873,
     -0.027747354,
     -0.062386185,
     0.03442664,
     -0.0006287391,
     -0.021185411,
     0.016059734,
     0.019775214,
     -0.016278243,
     0.10593479,
     0.024622368,
     0.047323518,
     0.057055604,
     -0.032622695,
     0.08374051,
     0.013158502,
     -0.017449643,
     -0.009003906,
     -0.002525663,
     0.04281074,
     -0.043502785,
     -0.013352289,
     -0.01252995,
     0.08813049,
     -0.084339894,
     -0.014972828,
     0.113745205,
     -0.03298367,
     0.009955621,
     -0.031106584,
     0.024170699,
     0.03699106,
     0.0028586816,
     -0.0026328918,
     0.037273373,
     0.031983916,
     0.04092413,
     0.00657442,
     -0.0489463,
     -0.016951311,
     0.017962592,
     0.0017097823,
     0.21778235,
     -0.012243278,
     -0.050046455,
     -0.0020600178,
     -0.05312466,
     -0.003115375,
     0.039001543,
     -0.033622406,
     0.023183865,
     -0.02117485,
     0.096117474,
     0.009511881,
     -0.052500606,
     -0.018172758,
     0.037292566,
     -0.05774915,
     -0.013294491,
     -0.068218686,
     0.016727602,
     -0.034375824,
     0.02118825,
     0.022696508,
     0.037687942,
     -0.003933483,
     -5.9311005e-05,
     -0.023479603,
     0.028875552,
     0.015842678,
     0.0132222725,
     0.04549442,
     0.029369421,
     -0.016189896,
     -0.013251936,
     0.010400359,
     -0.009519275,
     -0.030444251,
     -0.018722821,
     -0.019748418,
     -0.029346976,
     0.017578185,
     0.058790468,
     0.031586166,
     -0.026485054,
     -0.05637338,
     0.007970383,
     -0.0067538964,
     -0.035401482,
     -0.016126117,
     0.07254416,
     0.037870593,
     -0.047262035,
     -0.021215025,
     -0.0141087035,
     0.043487888,
     0.0525844,
     0.0068743783,
     -0.02619157,
     0.044459403,
     -0.017976051,
     -0.032685455,
     0.033352964,
     0.04940618,
     -0.012074308,
     -0.00035745287,
     -0.06456157,
     -0.044894412,
     -0.018132498,
     -0.007603064,
     -0.08239569,
     0.001174774,
     0.027255824,
     0.01614235,
     0.0025581785,
     0.02316907,
     0.008250368,
     0.0047441176,
     -0.022313109,
     -0.047295317,
     -0.006546519,
     0.029824222,
     -0.0048065954,
     0.038411144,
     0.01914943,
     0.031204311,
     -0.021600937,
     -0.047022417,
     0.0071235015,
     0.012847805,
     -0.00096813374,
     0.05473874,
     -0.049797658,
     0.02755442,
     0.027115408,
     0.011577001,
     -0.039570205,
     0.010787669,
     -0.0129262665,
     0.02622099,
     0.0051348624,
     0.0041144057,
     -0.005466008,
     -0.019787865,
     -0.00021865712,
     -0.02575863,
     -0.041323543,
     -0.029870346,
     0.009251507,
     -0.00676688,
     0.016422227,
     -0.01558209,
     -0.0032229177,
     0.021139063,
     0.002345135,
     0.04187644,
     -0.031707183,
     -0.020851452,
     0.004446392,
     -0.00826386,
     -0.021481778,
     -0.08407272,
     -0.024300965,
     0.00894761,
     0.08880225,
     -0.024432613,
     0.023595663,
     0.05086921,
     -0.018015018,
     -0.031196216,
     0.009017821,
     -7.595544e-05,
     -0.013442741,
     0.030178037,
     0.0033020433,
     0.023084441,
     0.053742934,
     -0.037115026,
     0.043991532,
     0.0058427486,
     -0.004270561,
     -0.006344164,
     0.004707068,
     0.037882797,
     -0.016168537,
     -0.036739457,
     -0.003819318,
     -0.021599479,
     0.031486053,
     -0.034679573,
     -0.013894724,
     0.0013890621,
     0.008054395,
     -0.033693124,
     0.0054222117,
     0.024470165,
     -0.014058812,
     0.014632633,
     0.003609711,
     0.006467272,
     0.0020671189,
     0.02342717,
     -0.085343845,
     0.0149695305,
     -0.009706586,
     0.021453405,
     -0.02290488,
     -0.023744095,
     -0.060693976,
     -0.036001198,
     0.04096357,
     0.017321398,
     2.0451593e-05,
     -0.027393999,
     -0.04250415,
     -0.03714641,
     0.024184689,
     -0.036342625,
     0.035094433,
     0.012959198,
     -0.06930901,
     -0.056883268,
     -0.022972742,
     0.048250962,
     -0.007026202,
     0.02460966,
     0.03518498,
     0.04548681,
     -0.024265407,
     -0.052501883,
     -0.008089454,
     -0.026447037,
     0.008933223,
     -0.008084308,
     -0.013452935,
     -0.02580882,
     0.03758563,
     -0.004602806,
     0.01112405,
     0.0058883782,
     0.025275705,
     -0.003333785,
     -0.022864748,
     0.043936163,
     0.020140208,
     -0.00065803836,
     0.02142301,
     0.019339848,
     0.012094271,
     0.03681956,
     0.011538933,
     -0.024688143,
     -0.016712397,
     -0.058771435,
     -0.0070213242,
     -0.030203633,
     -0.06518787,
     0.0058104848,
     0.041439813,
     0.0025484608,
     0.006904121,
     0.007701046,
     0.027719216,
     0.03659662,
     -0.015744874,
     -0.009165034,
     -0.022093333,
     -0.025826506,
     0.037204366,
     0.057949778,
     0.010409679,
     0.00042051065,
     -0.022143342,
     0.027365457,
     0.027921513,
     -0.045516424,
     -0.04994737,
     0.029813508,
     0.006173958,
     -0.02974992,
     0.015433339,
     0.030925905,
     0.014590269,
     -0.016110683,
     -0.011238454,
     -0.007279702,
     -0.02537937,
     0.01513152,
     -0.035147116,
     0.03519993,
     -0.024340604,
     -1.9054027e-05,
     -0.07206142,
     -0.0243227,
     0.013835538,
     0.06857519,
     0.046647206,
     -0.00018952311,
     -0.039867446,
     0.00706003,
     0.010219546,
     0.06681887,
     0.00025775243,
     0.024534605,
     0.041032437,
     0.044350326,
     -0.008519211,
     -0.03631305,
     -0.014887802,
     0.003934027,
     0.018266248,
     0.0100357765,
     -0.025854273,
     0.004913154,
     -0.06198882,
     -0.0129223745,
     0.027375495,
     0.06313827,
     0.025960622,
     -0.03840752,
     -0.01370126,
     -0.032558013,
     -0.023163652,
     -0.014939728,
     0.03540768,
     -0.006067803,
     0.011539442,
     -0.025558287,
     -0.055061538,
     -0.018397756,
     -0.009948869,
     -0.014422928,
     -0.0018951045,
     -0.04359461,
     0.019127095,
     -0.0038726127,
     -0.011145273,
     0.0043897317,
     -0.020771585,
     -0.044244785,
     -0.0510516,
     0.022646587,
     0.016734786,
     0.0048565995,
     -0.06837032,
     -0.016216282,
     -0.033660695,
     0.04109432,
     -0.029574582,
     0.017772945,
     -0.03189441,
     -0.018271543,
     0.016013876,
     0.008760026,
     0.0036045122,
     0.026603378,
     -0.014957655,
     -0.0071086567,
     -0.019532666,
     0.050095726,
     -0.038911704,
     -6.5375214e-05,
     -0.029691748,
     0.010813699,
     0.022713967,
     0.01322954,
     0.0034845378,
     -0.016643845,
     0.02070205,
     0.032483213,
     0.0014380713,
     -0.010250538,
     0.056960832,
     -0.030362243,
     -0.014216856,
     -0.03664275,
     -0.027183406,
     0.0145952385,
     -0.02570729,
     -0.02683875,
     0.004370176,
     0.042175207,
     0.006016668,
     0.019694686,
     0.015346909,
     0.0040553934]



### **Task 4**

(This task corresponds with Exercise 1 in the lab “Create and Configure a Vector Database to Store Document Embeddings”)

Submit a screenshot (saved as ‘vectordb.png’) that displays the code used to create a Chroma vector database that stores the embeddings of the document [new-Policies.txt](13. Project: GenAI Applications with RAG & LangChain/Notes/new-Policies.txt) and then conduct a similarity search for the following query with the top 5 results used.

query = "Smoking policy"


```python
from langchain_community.document_loaders.text import TextLoader
```


```python
## Document loader
def text_loader(file):
    loader = TextLoader(file.name)
    loaded_document = loader.load()
    return loaded_document

## Text splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks

## Vector db
def vector_database(chunks):
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

## Retriever
def create_retriever(file, k=1):
    print("Starting...")
    splits = text_loader(file)
    print("Data is splitted")
    chunks = text_splitter(splits)
    print("Data is chunked")
    vectordb = vector_database(chunks)
    print("Data is add into vectorstore")
    search_kwargs = {"k": k}
    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
    print("Retriever is created.")
    return retriever
```


```python
retriever = create_retriever(Path("new-Policies.txt"), k=5)
```

    Starting...
    Data is splitted
    Data is chunked
    Data is add into vectorstore
    Retriever is created.



```python
query = "Smoke Policy"
information = retriever.invoke(query)
information
```




    [Document(metadata={'source': 'new-Policies.txt'}, page_content='This policy promotes the safe and responsible use of digital communication tools in line with our values and legal obligations. Employees must understand and comply with this policy. Regular reviews will ensure it remains relevant with changing technology and security standards.\n\n4. Mobile Phone Policy\n\nOur Mobile Phone Policy defines standards for responsible use of mobile devices within the organization to ensure alignment with company values and legal requirements.\n\nAcceptable Use: Mobile devices are primarily for work-related tasks. Limited personal use is allowed if it does not disrupt work responsibilities.\n\nSecurity: Secure your mobile device and credentials. Be cautious with app downloads and links from unknown sources, and report any security issues promptly.\n\nConfidentiality: Avoid sharing sensitive company information via unsecured messaging apps or emails. Exercise caution when discussing company matters in public.'),
     Document(metadata={'source': 'new-Policies.txt'}, page_content='This policy promotes the safe and responsible use of digital communication tools in line with our values and legal obligations. Employees must understand and comply with this policy. Regular reviews will ensure it remains relevant with changing technology and security standards.\n\n4. Mobile Phone Policy\n\nOur Mobile Phone Policy defines standards for responsible use of mobile devices within the organization to ensure alignment with company values and legal requirements.\n\nAcceptable Use: Mobile devices are primarily for work-related tasks. Limited personal use is allowed if it does not disrupt work responsibilities.\n\nSecurity: Secure your mobile device and credentials. Be cautious with app downloads and links from unknown sources, and report any security issues promptly.\n\nConfidentiality: Avoid sharing sensitive company information via unsecured messaging apps or emails. Exercise caution when discussing company matters in public.'),
     Document(metadata={'source': 'new-Policies.txt'}, page_content='Safety: We prioritize the safety of our employees, clients, and the community. We encourage a culture of safety, including reporting any unsafe practices or conditions.\n\nEnvironmental Responsibility: We strive to reduce our environmental impact and promote sustainable practices.\n\nThis Code of Conduct is the cornerstone of our organizational culture. We expect every employee to uphold these principles and act as role models, ensuring our reputation for ethical conduct, integrity, and social responsibility.\n\n2. Recruitment Policy\n\nOur Recruitment Policy is dedicated to attracting, selecting, and integrating the most qualified and diverse candidates into our organization. The success of our company depends on the talent, skills, and commitment of our employees.'),
     Document(metadata={'source': 'new-Policies.txt'}, page_content='Safety: We prioritize the safety of our employees, clients, and the community. We encourage a culture of safety, including reporting any unsafe practices or conditions.\n\nEnvironmental Responsibility: We strive to reduce our environmental impact and promote sustainable practices.\n\nThis Code of Conduct is the cornerstone of our organizational culture. We expect every employee to uphold these principles and act as role models, ensuring our reputation for ethical conduct, integrity, and social responsibility.\n\n2. Recruitment Policy\n\nOur Recruitment Policy is dedicated to attracting, selecting, and integrating the most qualified and diverse candidates into our organization. The success of our company depends on the talent, skills, and commitment of our employees.'),
     Document(metadata={'source': 'new-Policies.txt'}, page_content='Security: Protect your login credentials and avoid sharing passwords. Be cautious with email attachments and links from unknown sources, and promptly report any unusual online activity or potential security threats.\n\nConfidentiality: Use email for confidential information, trade secrets, and sensitive customer data only with encryption. Be careful when discussing company matters on public platforms or social media.\n\nHarassment and Inappropriate Content: Internet and email must not be used for harassment, discrimination, or the distribution of offensive content. Always communicate respectfully and sensitively online.\n\nCompliance: Adhere to all relevant laws and regulations concerning internet and email use, including copyright and data protection laws.\n\nMonitoring: The company reserves the right to monitor internet and email usage for security and compliance purposes.\n\nConsequences: Violations of this policy may lead to disciplinary action, including potential termination.')]



### **Task 5**

(This task corresponds with Exercise 1 in the lab “Develop a Retriever to Fetch Document Segments based on Queries.”)

Submit a screenshot (saved as ‘retriever.png’) that displays the code used to use ChromaDB as a retriever and conduct a similarity search with the top 2 return results. 

The document you can use is [new-Policies.txt](13. Project: GenAI Applications with RAG & LangChain/Notes/new-Policies.txt). 

The query you can use is:

```python
query = "Email policy"
```


```python
retriever = create_retriever(Path("new-Policies.txt"), k=2)
```

    Starting...
    Data is splitted
    Data is chunked
    Data is add into vectorstore
    Retriever is created.



```python
query = "Email Policy"
information = retriever.invoke(query)
information
```




    [Document(metadata={'source': 'new-Policies.txt'}, page_content='This policy promotes the safe and responsible use of digital communication tools in line with our values and legal obligations. Employees must understand and comply with this policy. Regular reviews will ensure it remains relevant with changing technology and security standards.\n\n4. Mobile Phone Policy\n\nOur Mobile Phone Policy defines standards for responsible use of mobile devices within the organization to ensure alignment with company values and legal requirements.\n\nAcceptable Use: Mobile devices are primarily for work-related tasks. Limited personal use is allowed if it does not disrupt work responsibilities.\n\nSecurity: Secure your mobile device and credentials. Be cautious with app downloads and links from unknown sources, and report any security issues promptly.\n\nConfidentiality: Avoid sharing sensitive company information via unsecured messaging apps or emails. Exercise caution when discussing company matters in public.'),
     Document(metadata={'source': 'new-Policies.txt'}, page_content='This policy promotes the safe and responsible use of digital communication tools in line with our values and legal obligations. Employees must understand and comply with this policy. Regular reviews will ensure it remains relevant with changing technology and security standards.\n\n4. Mobile Phone Policy\n\nOur Mobile Phone Policy defines standards for responsible use of mobile devices within the organization to ensure alignment with company values and legal requirements.\n\nAcceptable Use: Mobile devices are primarily for work-related tasks. Limited personal use is allowed if it does not disrupt work responsibilities.\n\nSecurity: Secure your mobile device and credentials. Be cautious with app downloads and links from unknown sources, and report any security issues promptly.\n\nConfidentiality: Avoid sharing sensitive company information via unsecured messaging apps or emails. Exercise caution when discussing company matters in public.')]



### **Task 6**

(This task corresponds with the lab “Construct a QA Bot That Leverages the LangChain and LLM to Answer Questions from Loaded Documents.”)

Submit a screenshot (saved as ‘QA_bot.png’) that displays the QA bot interface you created based on the lab “Construct a QA Bot That Leverages the LangChain and LLM to Answer Questions from Loaded Documents.” Also, the picture should display that you uploaded a PDF and are asking a query to the bot. 

The PDF you can use is available [here](13. Project: GenAI Applications with RAG & LangChain/Notes/A_Comprehensive_Review_of_Low_Rank_Adaptation_in_Large_Language_Models_for_Efficient_Parameter_Tuning-1.pdf).

The query you can use is:

```python
query = "What this paper is talking about?
```



