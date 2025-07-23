## Objective


In this lab, you will learn how to load images and manipulate them for training using Keras ImageDataGenerator.


## Table of Contents

<div class="alert alert-block alert-info" style="margin-top: 20px">

<font size = 3>    

1. <a href="#item22">Import Libraries and Packages</a> 
2. <a href="#item21">Download Data</a> 
3. <a href="#item23">Construct an ImageDataGenerator Instance</a>  
4. <a href="#item24">Visualize Batches of Images</a>
5. <a href="#item25">Questions</a>    
</font>
    
</div>


   


<a id="item1"></a>


<a id='item21'></a>


## Import Libraries and Packages


Before we proceed, let's import the libraries and packages that we will need to complete the rest of this lab.



```python
import os
import numpy as np
import matplotlib.pyplot as plt
import skillsnetwork
# import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

    2025-03-01 19:40:15.684670: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1740838215.714715   58236 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1740838215.722410   58236 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2025-03-01 19:40:15.751190: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


## Download Data


For your convenience, I have placed the data on a server which you can retrieve and unzip easily using the **skillsnetwork.prepare** command. So let's run the following line of code to get the data. Given the large size of the image dataset, it might take some time depending on your internet speed.



```python
# await skillsnetwork.prepare("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/concrete_data_week2.zip",path = "data", overwrite=True)
```

Now, you should see two folders appear in the left pane: *Positive* and *Negative*. *Negative* is the negative class like we defined it earlier and it represents the concrete images with no cracks. *Positive* on the other hand is the positive class and represents the concrete images with cracks.


**Important Note**: There are thousands and thousands of images in each folder, so please don't attempt to double click on the *Negative* and *Positive* folders. This may consume all of your memory and you may end up with a **50*** error. So please **DO NOT DO IT**.


You can check the content of <code>./concrete_data_week2</code> by running the following:



```python
!ls data/concrete_data_week2
```

    Negative  Positive


or the following:



```python
os.listdir('data/concrete_data_week2')
```




    ['.DS_Store', 'Negative', 'Positive']



## Construct an ImageDataGenerator Instance


In this section, you will learn how to define a Keras ImageDataGenerator instance and use it to load and manipulate data for building a deep learning model.


Before we proceed, let's define a variable that represents the path to the folder containing our data which is <code>concrete_data_week2</code> in this case.



```python
dataset_dir = 'data/concrete_data_week2'
```

Keras ImageDataGenerator requires images be arranged in a certain folder hierarchy, where the main directory would contain folders equal to the number of classes in your problem. Since in this case we are trying to build a classifier of two classes, then our main directory, which is <code>concrete_data_week2</code>, should contain two folders, one for each class. This has already been done for you as the negative images are in one folder and the positive images are in another folder.


Let's go ahead and define an instance of the Keras ImageDataGenerator. 


#### Standard ImageDataGenerator


You can define a standard one like this, where you are simply using the ImageDataGenerator to train your model in batches.



```python
# instantiate your image data generator
data_generator = ImageDataGenerator()
```

Next, you use the <code>flow_from_directory</code> methods to loop through the images in batches. In this method, you pass the directory where the images reside, the size of each batch, *batch_size*, and since batches are sampled randomly, then you can also specify a random seed, *seed*, if you would like to reproduce the batch sampling. In case you would like to resize your images, then you can using the *target_size* argument to accomplish that.



```python
image_generator = data_generator.flow_from_directory(
    dataset_dir,
    batch_size=4,
    class_mode='categorical',
    seed=24
    )
```

    Found 40000 images belonging to 2 classes.


What is great about this method, is it prints a summary of it found in the directory passed. Here, it found 40,000 images in total belonging to 2 classes.


Now, to access the batches, you use the <code>next</code> method as follows:



```python
# first_batch = image_generator.next()
# first_batch = next(image_generator)
# first_batch
```

As you can see, this returned the images along with their labels. Therefore, the following returns the images only,



```python
# first_batch_images = image_generator.next()[0]
first_batch_images = next(image_generator)[0]
second_batch_images = next(image_generator)[0]
third_batch_images = next(image_generator)[0]
fourth_batch_images = next(image_generator)[0]
fifth_batch_images = next(image_generator)[0]
# first_batch_images
```

and the following returns the labels only.



```python
# first_batch_labels = next(image_generator)[1]
# first_batch_labels
```

#### Custom ImageDataGenerator


You can also specify some transforms, like scaling, rotations, and flips, that you would like applied to the images when you define an ImageDataGenerator object. Say you want to normalize your images, then you can define your ImageDataGenerator instance as follows:



```python
# instantiate your image data generator
data_generator = ImageDataGenerator(
    rescale=1./255
)
```

And then you proceed with defining your *image_generator* using the *flow_from_directory* method, just like before.



```python
image_generator = data_generator.flow_from_directory(
    dataset_dir,
    batch_size=4,
    class_mode='categorical',
    seed=24
    )
```

    Found 40000 images belonging to 2 classes.


However, now we explore the first batch using the *next* method, 



```python
first_batch = next(image_generator)
first_batch
```




    (array([[[[0.5921569 , 0.58431375, 0.5882353 ],
              [0.6       , 0.5921569 , 0.59607846],
              [0.60784316, 0.6       , 0.6039216 ],
              ...,
              [0.6       , 0.58431375, 0.5803922 ],
              [0.6       , 0.58431375, 0.5803922 ],
              [0.6       , 0.58431375, 0.5803922 ]],
     
             [[0.59607846, 0.5882353 , 0.5921569 ],
              [0.6039216 , 0.59607846, 0.6       ],
              [0.6117647 , 0.6039216 , 0.60784316],
              ...,
              [0.6039216 , 0.5882353 , 0.58431375],
              [0.6039216 , 0.5882353 , 0.58431375],
              [0.6039216 , 0.5882353 , 0.58431375]],
     
             [[0.6       , 0.5921569 , 0.59607846],
              [0.60784316, 0.6       , 0.6039216 ],
              [0.6117647 , 0.6039216 , 0.60784316],
              ...,
              [0.6117647 , 0.59607846, 0.5921569 ],
              [0.6117647 , 0.59607846, 0.5921569 ],
              [0.6117647 , 0.59607846, 0.5921569 ]],
     
             ...,
     
             [[0.5568628 , 0.54901963, 0.56078434],
              [0.5568628 , 0.54901963, 0.56078434],
              [0.5568628 , 0.54901963, 0.56078434],
              ...,
              [0.54509807, 0.5294118 , 0.53333336],
              [0.5568628 , 0.5411765 , 0.54509807],
              [0.5686275 , 0.5529412 , 0.5568628 ]],
     
             [[0.5568628 , 0.54901963, 0.56078434],
              [0.5568628 , 0.54901963, 0.56078434],
              [0.5568628 , 0.54901963, 0.56078434],
              ...,
              [0.54901963, 0.53333336, 0.5372549 ],
              [0.5568628 , 0.5411765 , 0.54509807],
              [0.57254905, 0.5568628 , 0.56078434]],
     
             [[0.5568628 , 0.54901963, 0.56078434],
              [0.5568628 , 0.54901963, 0.56078434],
              [0.5568628 , 0.54901963, 0.56078434],
              ...,
              [0.5529412 , 0.5372549 , 0.5411765 ],
              [0.5647059 , 0.54901963, 0.5529412 ],
              [0.5764706 , 0.56078434, 0.5647059 ]]],
     
     
            [[[0.7411765 , 0.7294118 , 0.70980394],
              [0.7490196 , 0.7372549 , 0.7176471 ],
              [0.7607844 , 0.7490196 , 0.7294118 ],
              ...,
              [0.7294118 , 0.7254902 , 0.70980394],
              [0.72156864, 0.7176471 , 0.7019608 ],
              [0.7176471 , 0.7137255 , 0.69803923]],
     
             [[0.7372549 , 0.7254902 , 0.7058824 ],
              [0.74509805, 0.73333335, 0.7137255 ],
              [0.7607844 , 0.7490196 , 0.7294118 ],
              ...,
              [0.7294118 , 0.7254902 , 0.70980394],
              [0.7254902 , 0.72156864, 0.7058824 ],
              [0.72156864, 0.7176471 , 0.7019608 ]],
     
             [[0.7294118 , 0.7176471 , 0.69803923],
              [0.7411765 , 0.7294118 , 0.70980394],
              [0.7568628 , 0.74509805, 0.7254902 ],
              ...,
              [0.7294118 , 0.7254902 , 0.70980394],
              [0.7254902 , 0.72156864, 0.7058824 ],
              [0.7254902 , 0.72156864, 0.7058824 ]],
     
             ...,
     
             [[0.7372549 , 0.7254902 , 0.7058824 ],
              [0.73333335, 0.72156864, 0.7019608 ],
              [0.7294118 , 0.7176471 , 0.69803923],
              ...,
              [0.6666667 , 0.6666667 , 0.63529414],
              [0.6666667 , 0.6666667 , 0.63529414],
              [0.67058825, 0.67058825, 0.6392157 ]],
     
             [[0.7372549 , 0.7254902 , 0.7058824 ],
              [0.73333335, 0.72156864, 0.7019608 ],
              [0.7294118 , 0.7176471 , 0.69803923],
              ...,
              [0.6784314 , 0.6666667 , 0.6392157 ],
              [0.67058825, 0.67058825, 0.6392157 ],
              [0.6745098 , 0.6745098 , 0.6431373 ]],
     
             [[0.7372549 , 0.7254902 , 0.7058824 ],
              [0.73333335, 0.72156864, 0.7019608 ],
              [0.7294118 , 0.7176471 , 0.69803923],
              ...,
              [0.70980394, 0.69803923, 0.67058825],
              [0.7019608 , 0.7019608 , 0.67058825],
              [0.7019608 , 0.7019608 , 0.67058825]]],
     
     
            [[[0.70980394, 0.6784314 , 0.63529414],
              [0.6901961 , 0.65882355, 0.6156863 ],
              [0.65882355, 0.627451  , 0.58431375],
              ...,
              [0.67058825, 0.654902  , 0.6117647 ],
              [0.6666667 , 0.6509804 , 0.60784316],
              [0.6627451 , 0.64705884, 0.6039216 ]],
     
             [[0.7058824 , 0.6745098 , 0.6313726 ],
              [0.6901961 , 0.65882355, 0.6156863 ],
              [0.6666667 , 0.63529414, 0.5921569 ],
              ...,
              [0.6745098 , 0.65882355, 0.6156863 ],
              [0.67058825, 0.654902  , 0.6117647 ],
              [0.6666667 , 0.6509804 , 0.60784316]],
     
             [[0.7019608 , 0.67058825, 0.627451  ],
              [0.6901961 , 0.65882355, 0.6156863 ],
              [0.67058825, 0.6392157 , 0.59607846],
              ...,
              [0.68235296, 0.6666667 , 0.62352943],
              [0.6784314 , 0.6627451 , 0.61960787],
              [0.6745098 , 0.65882355, 0.6156863 ]],
     
             ...,
     
             [[0.6862745 , 0.67058825, 0.627451  ],
              [0.69803923, 0.68235296, 0.6392157 ],
              [0.7137255 , 0.69803923, 0.654902  ],
              ...,
              [0.6784314 , 0.6627451 , 0.627451  ],
              [0.6784314 , 0.6627451 , 0.627451  ],
              [0.6784314 , 0.65882355, 0.63529414]],
     
             [[0.6862745 , 0.67058825, 0.627451  ],
              [0.69803923, 0.68235296, 0.6392157 ],
              [0.7137255 , 0.69803923, 0.654902  ],
              ...,
              [0.6862745 , 0.6666667 , 0.6431373 ],
              [0.6862745 , 0.6666667 , 0.6431373 ],
              [0.6862745 , 0.6666667 , 0.6431373 ]],
     
             [[0.6862745 , 0.67058825, 0.627451  ],
              [0.69803923, 0.68235296, 0.6392157 ],
              [0.7137255 , 0.69803923, 0.654902  ],
              ...,
              [0.69411767, 0.6745098 , 0.6509804 ],
              [0.69411767, 0.6745098 , 0.6509804 ],
              [0.69411767, 0.6745098 , 0.6509804 ]]],
     
     
            [[[0.5882353 , 0.58431375, 0.5764706 ],
              [0.5803922 , 0.5764706 , 0.5686275 ],
              [0.57254905, 0.5686275 , 0.56078434],
              ...,
              [0.7803922 , 0.76470596, 0.75294125],
              [0.7607844 , 0.74509805, 0.73333335],
              [0.7411765 , 0.7254902 , 0.7137255 ]],
     
             [[0.6       , 0.59607846, 0.5882353 ],
              [0.59607846, 0.5921569 , 0.58431375],
              [0.5921569 , 0.5882353 , 0.5803922 ],
              ...,
              [0.7686275 , 0.75294125, 0.7411765 ],
              [0.75294125, 0.7372549 , 0.7254902 ],
              [0.7294118 , 0.7137255 , 0.7019608 ]],
     
             [[0.6       , 0.59607846, 0.5882353 ],
              [0.60784316, 0.6039216 , 0.59607846],
              [0.6117647 , 0.60784316, 0.6       ],
              ...,
              [0.75294125, 0.7372549 , 0.7254902 ],
              [0.7372549 , 0.72156864, 0.70980394],
              [0.7137255 , 0.69803923, 0.6862745 ]],
     
             ...,
     
             [[0.7019608 , 0.6862745 , 0.68235296],
              [0.7058824 , 0.6901961 , 0.6862745 ],
              [0.70980394, 0.69411767, 0.6901961 ],
              ...,
              [0.68235296, 0.6666667 , 0.654902  ],
              [0.6784314 , 0.6627451 , 0.6509804 ],
              [0.6784314 , 0.6627451 , 0.6509804 ]],
     
             [[0.7058824 , 0.6901961 , 0.6862745 ],
              [0.7058824 , 0.6901961 , 0.6862745 ],
              [0.7058824 , 0.6901961 , 0.6862745 ],
              ...,
              [0.6901961 , 0.6745098 , 0.6627451 ],
              [0.6862745 , 0.67058825, 0.65882355],
              [0.6784314 , 0.6627451 , 0.6509804 ]],
     
             [[0.7137255 , 0.69803923, 0.69411767],
              [0.70980394, 0.69411767, 0.6901961 ],
              [0.7019608 , 0.6862745 , 0.68235296],
              ...,
              [0.7058824 , 0.6901961 , 0.6784314 ],
              [0.69411767, 0.6784314 , 0.6666667 ],
              [0.68235296, 0.6666667 , 0.654902  ]]]], dtype=float32),
     array([[1., 0.],
            [0., 1.],
            [1., 0.],
            [0., 1.]], dtype=float32))



we find that the values are not integer values anymore, but scaled resolution since the original number are divided by 255.


You can learn more about the Keras ImageDataGeneration class [here](https://keras.io/preprocessing/image/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0321ENSkillsNetwork951-2022-01-01).


<a id='item24'></a>


## Visualize Batches of Images


Let write some code to visualize a batch. We will use subplots in order to make visualizing the images easier.


Recall that we can access our batch images as follows:

<code>first_batch_images = image_generator.next()[0] # first batch</code>

<code>second_batch_images = image_generator.next()[0] # second batch</code>

and so on.



```python
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10)) # define your figure and axes

ind = 0
for ax1 in axs:
    for ax2 in ax1: 
        image_data = first_batch_images[ind].astype(np.uint8)
        ax2.imshow(image_data)
        ind += 1

fig.suptitle('First Batch of Concrete Images') 
plt.show()
```


    
![png](DL0321EN-2-1-Data-Preparation-py-v1.0_files/DL0321EN-2-1-Data-Preparation-py-v1.0_49_0.png)
    


Remember that batches are sampled randomly from the data. In our first batch, we ended up with two negative image and two positive images.


**Important Note**: Because of a bug with the imshow function in Matplotlib, if you are plotting the unscaled RGB images, you have to cast the **image_data** to uint8 before you call the <code>imshow</code> function. So In the code above It looks like this:

image_data = first_batch_images[ind].astype(np.uint8)


<a id='item25'></a>


## Questions


### Question: Create a plot to visualize the images in the third batch.



```python
len(image_generator)
```




    10000




```python
# second_batch_images = next(image_generator)[0]
# third_batch_images = next(image_generator)[0]
# third_batch_images
```


```python
## You can use this cell to type your code to answer the above question
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10)) # define your figure and axes

ind = 0
for ax1 in axs:
    for ax2 in ax1: 
        image_data = third_batch_images[ind].astype(np.uint8)
        ax2.imshow(image_data)
        ind += 1

fig.suptitle('Third Batch of Concrete Images') 
plt.show()

```


    
![png](DL0321EN-2-1-Data-Preparation-py-v1.0_files/DL0321EN-2-1-Data-Preparation-py-v1.0_57_0.png)
    


### Question: How many images from each class are in the fourth batch?



```python
## You can use this cell to type your code to answer the above question
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10)) # define your figure and axes

ind = 0
for ax1 in axs:
    for ax2 in ax1: 
        image_data = fourth_batch_images[ind].astype(np.uint8)
        ax2.imshow(image_data)
        ind += 1

fig.suptitle('Fourth Batch of Concrete Images') 
plt.show()


```


    
![png](DL0321EN-2-1-Data-Preparation-py-v1.0_files/DL0321EN-2-1-Data-Preparation-py-v1.0_59_0.png)
    


### Question: Create a plot to visualize the second image in the fifth batch.



```python
## You can use this cell to type your code to answer the above question
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10)) # define your figure and axes

ind = 0
for ax1 in axs:
    for ax2 in ax1: 
        image_data = fifth_batch_images[ind].astype(np.uint8)
        ax2.imshow(image_data)
        ind += 1

fig.suptitle('Fifth Batch of Concrete Images') 
plt.show()


```


    
![png](DL0321EN-2-1-Data-Preparation-py-v1.0_files/DL0321EN-2-1-Data-Preparation-py-v1.0_61_0.png)
    


### Question: How many images from each class are in the fifth batch?



```python
## You can use this cell to type your code to answer the above question
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10)) # define your figure and axes

ind = 0
for ax1 in axs:
    for ax2 in ax1: 
        image_data = fifth_batch_images[ind].astype(np.uint8)
        ax2.imshow(image_data)
        ind += 1

fig.suptitle('Fifth Batch of Concrete Images') 
plt.show()


```


    
![png](DL0321EN-2-1-Data-Preparation-py-v1.0_files/DL0321EN-2-1-Data-Preparation-py-v1.0_63_0.png)
    


   


Make sure to answer the above questions as the quiz in this module is heavily based on them.


  


   


### Thank you for completing this lab!

This notebook was created by Alex Aklson. I hope you found this lab interesting and educational.


This notebook is part of a course on **Coursera** called *AI Capstone Project with Deep Learning*. If you accessed this notebook outside the course, you can take this course online by clicking [here](https://cocl.us/DL0321EN_Coursera_Week2_LAB1).



## Change Log

|  Date (YYYY-MM-DD) |  Version | Changed By  |  Change Description |
|---|---|---|---|
| 2020-09-18  | 2.0  | Shubham  |  Migrated Lab to Markdown and added to course repo in GitLab |



<hr>

Copyright &copy; 2020 [IBM Developer Skills Network](https://cognitiveclass.ai/?utm_medium=dswb&utm_source=bducopyrightlink&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0321ENSkillsNetwork951-2022-01-01&utm_campaign=bdu). This notebook and its source code are released under the terms of the [MIT License](https://bigdatauniversity.com/mit-license/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0321ENSkillsNetwork951-2022-01-01).

