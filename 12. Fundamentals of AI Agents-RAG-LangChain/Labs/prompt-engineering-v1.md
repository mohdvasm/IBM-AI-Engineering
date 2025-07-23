# **In-Context Engineering and Prompt Templates**


Estimated time needed: **30** minutes


## Overview


You're stepping into the world of prompt engineering, where each command you craft has the power to guide intelligent LLM systems toward specific outcomes. In this tutorial, you will explore the foundational aspects of prompt engineering, dive into advanced techniques of in-context learning, such as few-shot and self-consistent learning, and learn how to effectively use tools like Langchain.

Start by understanding the basics—how to formulate prompts that communicate effectively with AI. From there, we'll explore how the Langchain prompt template can simplify and enhance this process, making it more structured and efficient.

As you progress, you'll learn to apply these skills in practical scenarios, creating sophisticated applications like QA bots and text summarization tools. By using the Langchain prompt template, you'll see firsthand how structured prompting can streamline the development of these applications, transforming complex requirements into clear, concise tasks for AI.


<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ai8G4tOU4mksEYfv5wsghA/prompt%20engineering.png" width="50%" alt="indexing"/>


By the end of this tutorial, you'll not only master the different techniques of prompt engineering but also acquire hands-on experience in applying these techniques to real-world problems, ensuring you're well-prepared to harness the full potential of AI in various settings.


## __Table of Contents__

<ol>
    <li><a href="#Objectives">Objectives</a></li>
    <li>
        <a href="#Setup">Setup</a>
        <ol>
            <li><a href="#Installing-required-libraries">Installing required libraries</a></li>
            <li><a href="#Importing-required-libraries">Importing required libraries</a></li>
            <li><a href="#Setup-LLM">Setup LLM</a></li>
        </ol>
    </li>
    <li>
        <a href="#Prompt-engineering">Prompt engineering</a>
        <ol>
            <li><a href="#First-basic-prompt">First basic prompt</a></li>
            <li><a href="#Zero-shot-prompt">Zero-shot prompt</a></li>
            <li><a href="#One-shot-prompt">One-shot prompt</a></li>
            <li><a href="#Few-shot-prompt">Few-shot prompt</a></li>
            <li><a href="#Chain-of-thought-(CoT)-prompting">Chain-of-thought (CoT) prompting</a></li>
            <li><a href="#Self-consistency">Self-consistency</a></li>
        </ol>
    </li>
    <li>
        <a href="#Applications">Applications</a>
        <ol>
            <li><a href="#Prompt-template">Prompt template</a></li>
            <li><a href="#Text-summarization">Text summarization</a></li>
            <li><a href="#Question-answering">Question answering</a></li>
            <li><a href="#Text-classification">Text classification</a></li>
            <li><a href="#Code-generation">Code generation</a></li>
            <li><a href="#Role-playing">Role playing</a></li>
        </ol>
    </li>
</ol>

<a href="#Exercises">Exercises</a>
<ol>
    <li><a href="#Exercise-1:-Change-parameters-for-the-LLM">Exercise 1: Change parameters for the LLM</a></li>
    <li><a href="#Exercise-2:-Observe-how-LLM-thinks">Exercise 2: Observe how LLM thinks</a></li>
    <li><a href="#Exercise-3:-Revise-the-text-classification-agent-to-one-shot-learning">Exercise 3: Revise the text classification agent to one-shot learning</a></li>
</ol>


## Objectives

After completing this lab, you will be able to:

- **Understand the basics of prompt engineering**: Gain a solid foundation in how to effectively communicate with LLM using prompts, setting the stage for more advanced techniques.

- **Master advanced prompt techniques**: Learn and apply advanced prompt engineering techniques such as few-shot and self-consistent learning to optimize the LLM's response.

- **Utilize LangChain prompt template**: Become proficient in using LangChain's prompt template to structure and optimize your interactions with LLM.

- **Develop practical LLM agents**: Acquire the skills to create and implement agents such as QA bots and text summarization using the Langchain prompt template, translating theoretical knowledge into practical solutions.


----


## Setup


For this lab, you will be using the following libraries:

*   [`ibm-watsonx-ai`](https://ibm.github.io/watson-machine-learning-sdk/index.html) for using LLMs from IBM's watsonx.ai.
*   [`langchain`](https://www.langchain.com/) for using langchain's different chain and prompt functions.
*   [`langchain-ibm`](https://python.langchain.com/v0.1/docs/integrations/llms/ibm_watsonx/) provides integration between langchain and ibm-watsonx-ai.


### Installing required libraries

The following required libraries are __not__ preinstalled in the Skills Network Labs environment. __You must run the following cell__ to install them:

**Note:** The version has been pinned here to specify the version. It's recommended that you do this as well. Even though the library will be updated in the future, the library could still support this lab work.

This might take approximately 1-2 minutes.

`%%capture` has been used to capture the installation, you won't see the output process. But once the installation is done, you will see a number beside the cell.



```python
%%capture
!pip install "ibm-watsonx-ai==0.2.6"
!pip install "langchain==0.1.16" 
!pip install "langchain-ibm==0.1.4"
```

### Importing required libraries

_It is recommended that you import all required libraries in one place (here):_



```python
# You can also use this section to suppress warnings generated by your code:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
```

### Setup LLM


In this section, you will build an LLM model from IBM watsonx.ai. 

The following code initializes a Mixtral model on IBM's watsonx.ai platform and wraps it into a function that could allow repeat use.


Some key parameters are explained here:
- `model_id` specifies which model you want to use. There are various model options available; refer to the [Foundation Models](https://ibm.github.io/watsonx-ai-python-sdk/foundation_models.html) documentation for more options. In this tutorial, you'll use the `mixtral-8x7b-instruct-v01` model.
- `parameters` define the model's configuration. Set five commonly used parameters for this tutorial. To explore additional commonly used parameters, you can run the code `GenParams().get_example_values()` to see. If no custom parameters are passed to the function, the model will use `default_params`.
- `credentials` and `project_id` are necessary parameters to successfully run LLMs from watsonx.ai. (Keep `credentials` and `project_id` as they are now so that you do not need to create your own keys to run models. This supports you in running the model inside this lab environment. However, if you want to run the model locally, refer to this [tutorial](https://medium.com/the-power-of-ai/ibm-watsonx-ai-the-interface-and-api-e8e1c7227358) for creating your own keys.
- `Model()` is used to wrap the parameters for the model and then call it with `WatsonxLLM()`.


Run the following code, you will initialize a LLM model.



```python
def llm_model(prompt_txt, params=None):
    model_id = 'mistralai/mixtral-8x7b-instruct-v01'

    default_params = {
        "max_new_tokens": 256,
        "min_new_tokens": 0,
        "temperature": 0.5,
        "top_p": 0.2,
        "top_k": 1
    }

    if params:
        default_params.update(params)

    parameters = {
        GenParams.MAX_NEW_TOKENS: default_params["max_new_tokens"],  # this controls the maximum number of tokens in the generated output
        GenParams.MIN_NEW_TOKENS: default_params["min_new_tokens"], # this controls the minimum number of tokens in the generated output
        GenParams.TEMPERATURE: default_params["temperature"], # this randomness or creativity of the model's responses
        GenParams.TOP_P: default_params["top_p"],
        GenParams.TOP_K: default_params["top_k"]
    }
    
    credentials = {
        "url": "https://us-south.ml.cloud.ibm.com"
    }
    
    project_id = "skills-network"
    
    model = Model(
        model_id=model_id,
        params=parameters,
        credentials=credentials,
        project_id=project_id
    )
    
    mixtral_llm = WatsonxLLM(model=model)
    response  = mixtral_llm.invoke(prompt_txt)
    return response
```

Let's run the following code to see some other commonly used parameters and their default value.



```python
GenParams().get_example_values()
```

## Prompt engineering


### First basic prompt


In this example, let's introduce a basic prompt that utilizes specific parameters to guide the language model's response. You'll then define a simple prompt and retrieve the model's response,

The prompt used is "The wind is". Let the model generate itself.



```python
params = {
    "max_new_tokens": 128,
    "min_new_tokens": 10,
    "temperature": 0.5,
    "top_p": 0.2,
    "top_k": 1
}

prompt = "The wind is"

response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")
```

As you can see from the response, the model continues generating content following the initial prompt, "The wind is". You might notice that the response appears truncated or incomplete. This is because you have set the `max_new_tokens,` which restricts the number of tokens the model can generate.

Try to adjust the parameters and observe the difference in the response.


### Zero-shot prompt


Here is an example of a zero-shot prompt. 

Zero-shot learning is crucial for testing a model's ability to apply its pre-trained knowledge to new, unseen tasks without additional training. This capability is valuable for gauging the model's generalization skills.

In this example, let's demonstrate a zero-shot learning scenario using a prompt that asks the model to classify a statement without any prior specific training on similar tasks. The prompt requests the model to assess the truthfulness of the statement: "The Eiffel Tower is located in Berlin.". After defining the prompt, you'll execute it with default parameters and print the response.

This approach helps you understand how well the model can handle direct questions based on its underlying knowledge and reasoning abilities.


Try running the prompt to see the model's capacity to correctly analyze and respond to factual inaccuracies.



```python
prompt = """Classify the following statement as true or false: 
            'The Eiffel Tower is located in Berlin.'

            Answer:
"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")
```

The model responds with the 'False' answer, which is correct. It also gives the reason for it.


### One-shot prompt


Here is a one-shot learning example where the model is given a single example to help guide its translation from English to French.

The prompt provides a sample translation pairing, "How is the weather today?" translated to "Comment est le temps aujourd'hui?" This example serves as a guide for the model to understand the task context and desired format. The model is then tasked with translating a new sentence, "Where is the nearest supermarket?" without further guidance.



```python
params = {
    "max_new_tokens": 20,
    "temperature": 0.1,
}

prompt = """Here is an example of translating a sentence from English to French:

            English: “How is the weather today?”
            French: “Comment est le temps aujourd'hui?”
            
            Now, translate the following sentence from English to French:
            
            English: “Where is the nearest supermarket?”
            
"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")
```

The model's response shows how it applies the structure and context provided by the initial example to translate the new sentence.


Consider experimenting with different sentences or adjusting the parameters to see how these changes impact the model's translations.


### Few-shot prompt


Here is an example of few-shot learning by classifying emotions from text statements. 

Let's provide the model with three examples, each labeled with an appropriate emotion—joy, frustration, and sadness—to establish a pattern or guideline on how to categorize emotions in statements.

After presenting these examples, let's challenge the model with a new statement: "That movie was so scary I had to cover my eyes." The task for the model is to classify the emotion expressed in this new statement based on the learning from the provided examples. 



```python
 #parameters  `max_new_tokens` to 10, which constrains the model to generate brief responses

params = {
    "max_new_tokens": 10,
}

prompt = """Here are few examples of classifying emotions in statements:

            Statement: 'I just won my first marathon!'
            Emotion: Joy
            
            Statement: 'I can't believe I lost my keys again.'
            Emotion: Frustration
            
            Statement: 'My best friend is moving to another country.'
            Emotion: Sadness
            
            Now, classify the emotion in the following statement:
            Statement: 'That movie was so scary I had to cover my eyes.’
            

"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")
```

The parameters are set with `max_new_tokens` to 10, which constrains the model to generate brief responses, focusing on the essential output without elaboration.


The model's response demonstrates its ability to use the provided few examples to understand and classify the emotion of the new statement effectively following the same pattern in examples.


### Chain-of-thought (CoT) prompting


Here is an example of the Chain-of-Thought (CoT) prompting technique, designed to guide the model through a sequence of reasoning steps to solve a problem. In this example, the problem is a simple arithmetic question: “A store had 22 apples. They sold 15 apples today and received a new delivery of 8 apples. How many apples are there now?”

The CoT technique involves structuring the prompt by instructing the model to “Break down each step of your calculation.” This encourages the model to include explicit reasoning steps, mimicking human-like problem-solving processes.



```python
params = {
    "max_new_tokens": 512,
    "temperature": 0.5,
}

prompt = """Consider the problem: 'A store had 22 apples. They sold 15 apples today and got a new delivery of 8 apples. 
            How many apples are there now?’

            Break down each step of your calculation

"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")
```

From the response of the model, you can see the prompt directs the model to:

1. Add the initial number of apples to the apples received in the new delivery.
2. Subtract the number of apples sold from the sum obtained in the first step.


By breaking down the problem into specific steps, the model is better able to understand the sequence of operations required to arrive at the correct answer.


### Self-consistency


This example demonstrates the self-consistency technique in reasoning through multiple calculations for a single problem. The problem posed is: “When I was 6, my sister was half my age. Now I am 70, what age is my sister?”

The prompt instructs, “Provide three independent calculations and explanations, then determine the most consistent result.” This encourages the model to engage in critical thinking and consistency checking, which are vital for complex decision-making processes.



```python
params = {
    "max_new_tokens": 512,
}

prompt = """When I was 6, my sister was half of my age. Now I am 70, what age is my sister?

            Provide three independent calculations and explanations, then determine the most consistent result.

"""
response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response : {response}\n")
```

The model's response shows that it provides three different calculations and explanations. Each calculation attempts to derive the sister's age using different logical approaches.

Self-consistency can help identify the most accurate and reliable answer in scenarios where multiple plausible solutions exist.


## Applications


In this section, you will show you how to use the prompt template from Langchain to create more structured and reproducible prompts. You will also learn to create some applications based on the prompt template.


### Prompt template


[Prompt template](https://python.langchain.com/v0.2/docs/concepts/#prompt-templates) is a key concept in langchain, it helps to translate user input and parameters into instructions for a language model. This can be used to guide a model's response, helping it understand the context and generate relevant and coherent language-based output.


To use the prompt template, you need to initialize a LLM first.

You can still use the `mixtral-8x7b-instruct-v01` from watsonx.ai.



```python
model_id = 'mistralai/mixtral-8x7b-instruct-v01'

parameters = {
    GenParams.MAX_NEW_TOKENS: 256,  # this controls the maximum number of tokens in the generated output
    GenParams.TEMPERATURE: 0.5, # this randomness or creativity of the model's responses
}

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

project_id = "skills-network"

model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

mixtral_llm = WatsonxLLM(model=model)
mixtral_llm
```

Use the `PromptTemplate` to create a template for a string-based prompt. In this template, you'll define two parameters: `adjective` and `content`. These parameters allow for the reuse of the prompt across different situations. For instance, to adapt the prompt to various contexts, simply pass the relevant values to these parameters.



```python
template = """Tell me a {adjective} joke about {content}.
"""
prompt = PromptTemplate.from_template(template)
prompt 
```

Now, let's take a look at how the prompt has been formatted.



```python
prompt.format(adjective="funny", content="chickens")
```

From the response, you can see that the prompt is formatted according to the specified context.


The following code will wrap the formatted prompt into the LLMChain, and then invoke the prompt to get the response from the LLM.



```python
llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm)
response = llm_chain.invoke(input = {"adjective": "funny", "content": "chickens"})
print(response["text"])
```

From the response, you can see the LLM came up with a funny joke about chickens.

To use this prompt in another context, simply replace the variables accordingly



```python
response = llm_chain.invoke(input = {"adjective": "sad", "content": "fish"})
print(response["text"])
```

In the following sections, you will learn how to create agents capable of completing various tasks using prompt templates.


### Text summarization


Here is a text summarization agent designed to help summarize the content you provide to the LLM. 

You can store the content to be summarized in a variable, allowing for repeated use of the prompt.



```python
content = """
        The rapid advancement of technology in the 21st century has transformed various industries, including healthcare, education, and transportation. 
        Innovations such as artificial intelligence, machine learning, and the Internet of Things have revolutionized how we approach everyday tasks and complex problems. 
        For instance, AI-powered diagnostic tools are improving the accuracy and speed of medical diagnoses, while smart transportation systems are making cities more efficient and reducing traffic congestion. 
        Moreover, online learning platforms are making education more accessible to people around the world, breaking down geographical and financial barriers. 
        These technological developments are not only enhancing productivity but also contributing to a more interconnected and informed society.
"""

template = """Summarize the {content} in one sentence.
"""
prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm)
response = llm_chain.invoke(input = {"content": content})
print(response["text"])
```

### Question answering


Here is a Q&A agent. 

This agent enables the LLM to learn from the provided content and answer questions based on what it has learned. Occasionally, if the LLM does not have sufficient information, it might generate a speculative answer. To manage this, you'll specifically instruct it to respond with "Unsure about the answer" if it is uncertain about the correct response.



```python
content = """
        The solar system consists of the Sun, eight planets, their moons, dwarf planets, and smaller objects like asteroids and comets. 
        The inner planets—Mercury, Venus, Earth, and Mars—are rocky and solid. 
        The outer planets—Jupiter, Saturn, Uranus, and Neptune—are much larger and gaseous.
"""

question = "Which planets in the solar system are rocky and solid?"

template = """
            Answer the {question} based on the {content}.
            Respond "Unsure about answer" if not sure about the answer.
            
            Answer:
            
"""
prompt = PromptTemplate.from_template(template)
output_key = "answer"

llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm, output_key=output_key)
response = llm_chain.invoke(input = {"question":question ,"content": content})
print(response["answer"])
```

### Text classification


Here is a text classification agent designed to categorize text into predefined categories. This example employs zero-shot learning, where the agent classifies text without prior exposure to related examples.

Can you revise it to the one-shot learning or few-shot learning in the exercises?



```python
text = """
        The concert last night was an exhilarating experience with outstanding performances by all artists.
"""

categories = "Entertainment, Food and Dining, Technology, Literature, Music."

template = """
            Classify the {text} into one of the {categories}.
            
            Category:
            
"""
prompt = PromptTemplate.from_template(template)
output_key = "category"

llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm, output_key=output_key)
response = llm_chain.invoke(input = {"text":text ,"categories": categories})
print(response["category"])
```

### Code generation


Here is an example of an SQL code generation agent. This agent is designed to generate SQL queries based on given descriptions. It interprets the requirements from your input and translates them into executable SQL code.



```python
description = """
        Retrieve the names and email addresses of all customers from the 'customers' table who have made a purchase in the last 30 days. 
        The table 'purchases' contains a column 'purchase_date'
"""

template = """
            Generate an SQL query based on the {description}
            
            SQL Query:
            
"""
prompt = PromptTemplate.from_template(template)
output_key = "query"

llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm, output_key=output_key)
response = llm_chain.invoke(input = {"description":description})
print(response["query"])
```

### Role playing


You can also configure the LLM to assume specific roles as defined by us, enabling it to follow predetermined rules and behave like a task-oriented chatbot.

For example, the code below configures the LLM to act as a game master. In this role, the LLM answers questions about games while maintaining an engaging and immersive tone, enhancing the user experience.


Run the following code to create the prompt template and create a LLMChian to wrap the prompt.



```python
role = """
        game master
"""

tone = "engaging and immersive"

template = """
            You are an expert {role}. I have this question {question}. I would like our conversation to be {tone}.
            
            Answer:
            
"""
prompt = PromptTemplate.from_template(template)
output_key = "answer"

llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm, output_key=output_key)
```

The following code will create a game master chatbot that takes your questions as input and provides responses from the model.

Run the code below to launch the bot.


You can test the bot by asking the question, "Who are you?" The bot will respond with "I am a game master," indicating it has assumed the role that you predefined.

The function is written within a while loop, allowing continuous interaction. To exit the loop and terminate the conversation, type "quit," "exit," or "bye" into the input box.



```python
while True:
    query = input("Question: ")
    
    if query.lower() in ["quit","exit","bye"]:
        print("Answer: Goodbye!")
        break
        
    response = llm_chain.invoke(input = {"role": role, "question": query, "tone": tone})
    
    print("Answer: ", response["answer"])
```

Great! You finish the lab. Now let's take some exercises.


# Exercises


### Exercise 1: Change parameters for the LLM


Experiment with changing the parameters of the LLM to observe how different settings impact the responses. Adjusting parameters such as `max_new_tokens`, `temperature`, or `top_p` can significantly alter the behavior of the model. Try different configurations to see how each variation influences the output.



```python
# TODO
```

<details>
    <summary>Click here for Solution</summary>

```python
params = {
    "max_new_tokens": 128,
    "min_new_tokens": 100,
    "temperature": 1,
    "top_p": 0.1,
    "top_k": 1
}

prompt = "The wind is"

response = llm_model(prompt, params)
print(response)

```

</details>


### Exercise 2: Observe how LLM thinks


You can set `verbose=True` in the `LLMChain()` to observe the thought process of the LLM, gaining insights into how it formulates its responses. Can you make it any agent you created before to observe it?



```python
# TODO
```

<details>
    <summary>Click here for Solution</summary>

```python
content = """
        The solar system consists of the Sun, eight planets, their moons, dwarf planets, and smaller objects like asteroids and comets. 
        The inner planets—Mercury, Venus, Earth, and Mars—are rocky and solid. 
        The outer planets—Jupiter, Saturn, Uranus, and Neptune—are much larger and gaseous.
"""

question = "Which planets in the solar system are rocky and solid?"

template = """
            Answer the {question} based on the {content}.
            Respond "Unsure about answer" if not sure about the answer.
            
            Answer:
            
"""
prompt = PromptTemplate.from_template(template)
output_key = "answer"

llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm, output_key=output_key, verbose=True)
response = llm_chain.invoke(input = {"question":question ,"content": content})
print(response["answer"])

```

</details>


### Exercise 3: Revise the text classification agent to one-shot learning


You were using zero-shot learning when you created the text classification agent. Can you revise it to use one-shot learning?



```python
# TODO
```

<details>
    <summary>Click here for Solution</summary>

```python
example_text = """
               Last week's book fair was a delightful gathering of authors and readers, featuring discussions and book signings.
               """

example_category = "Literature"

text = """
       The concert last night was an exhilarating experience with outstanding performances by all artists.
       """

categories = "Entertainment, Food and Dining, Technology, Literature, Music."

template = """
           Example:
           Text: {example_text}
           Category: {example_category}

           Now, classify the following text into one of the specified categories: {categories}
           
           Text: {text}
           
           Category:
           
           """
prompt = PromptTemplate.from_template(template)
output_key = "category"

llm_chain = LLMChain(prompt=prompt, llm=mixtral_llm, output_key=output_key)
response = llm_chain.invoke(input = {"example_text": example_text, "example_category":example_category ,"categories": categories, "text":text})
print(response["category"])

```

</details>


## Authors


[Kang Wang](https://author.skills.network/instructors/kang_wang)

Kang Wang is a Data Scientist in IBM. He is also a PhD Candidate in the University of Waterloo.


© Copyright IBM Corporation. All rights reserved.

