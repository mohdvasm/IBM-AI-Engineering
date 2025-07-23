<a href="https://colab.research.google.com/github/mohdvasm/IBM-AI-Engineering/blob/main/06.%20AI%20Capstone%20Project%20With%20DL/Labs/DL0321EN-3-1-Pretrained-Models-py-v1.0.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Objective


In this lab, you will learn how to leverage pre-trained models to build image classifiers instead of building a model from scratch.


## Table of Contents

<div class="alert alert-block alert-info" style="margin-top: 20px">

<font size = 3>
    
1. <a href="#item31">Import Libraries and Packages</a>
2. <a href="#item32">Download Data</a>  
3. <a href="#item33">Define Global Constants</a>  
4. <a href="#item34">Construct ImageDataGenerator Instances</a>  
5. <a href="#item35">Compile and Fit Model</a>

</font>
    
</div>


   


<a id='item31'></a>



```python
!pip -q install skillsnetwork
```

    [?25l   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.0/1.6 MB[0m [31m?[0m eta [36m-:--:--[0m[2K   [91m━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.2/1.6 MB[0m [31m10.5 MB/s[0m eta [36m0:00:01[0m[2K   [91m━━━━━━[0m[91m╸[0m[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0.3/1.6 MB[0m [31m4.1 MB/s[0m eta [36m0:00:01[0m[2K   [91m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[91m╸[0m [32m1.6/1.6 MB[0m [31m16.5 MB/s[0m eta [36m0:00:01[0m[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.6/1.6 MB[0m [31m14.2 MB/s[0m eta [36m0:00:00[0m
    [?25h

## Import Libraries and Packages


Let's start the lab by importing the libraries that we will be using in this lab. First we will need the library that helps us to import the data.



```python
import skillsnetwork
```

First, we will import the ImageDataGenerator module since we will be leveraging it to train our model in batches.



```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow
```

In this lab, we will be using the Keras library to build an image classifier, so let's download the Keras library.



```python
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

Finally, we will be leveraging the ResNet50 model to build our classifier, so let's download it as well.



```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
```

<a id='item32'></a>


## Download Data


In this section, you are going to download the data from IBM object storage using **skillsnetwork.prepare** command. skillsnetwork.prepare is a command that's used to download a zip file, unzip it and store it in a specified directory.



```python
## get the data
await skillsnetwork.prepare("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0321EN-SkillsNetwork/concrete_data_week3.zip", overwrite=True)
```


    Downloading concrete_data_week3.zip:   0%|          | 0/97863179 [00:00<?, ?it/s]



      0%|          | 0/30036 [00:00<?, ?it/s]


    Saved to '.'


Now, you should see the folder *concrete_data_week3* appear in the left pane. If you open this folder by double-clicking on it, you will find that it contains two folders: *train* and *valid*. And if you explore these folders, you will find that each contains two subfolders: *positive* and *negative*. These are the same folders that we saw in the labs in the previous modules of this course, where *negative* is the negative class and it represents the concrete images with no cracks and *positive* is the positive class and it represents the concrete images with cracks.


**Important Note**: There are thousands and thousands of images in each folder, so please don't attempt to double click on the *negative* and *positive* folders. This may consume all of your memory and you may end up with a **50** error. So please **DO NOT DO IT**.


<a id='item33'></a>


## Define Global Constants


Here, we will define constants that we will be using throughout the rest of the lab.

1. We are obviously dealing with two classes, so *num_classes* is 2.
2. The ResNet50 model was built and trained using images of size (224 x 224). Therefore, we will have to resize our images from (227 x 227) to (224 x 224).
3. We will training and validating the model using batches of 100 images.



```python
num_classes = 2

image_resize = 224

batch_size_training = 100
batch_size_validation = 100
```

<a id='item34'></a>


## Construct ImageDataGenerator Instances


In order to instantiate an ImageDataGenerator instance, we will set the **preprocessing_function** argument to *preprocess_input* which we imported from **keras.applications.resnet50** in order to preprocess our images the same way the images used to train ResNet50 model were processed.



```python
data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)
```

Next, we will use the *flow_from_directory* method to get the training images as follows:



```python
train_generator = data_generator.flow_from_directory(
    'concrete_data_week3/train',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical')
```

    Found 10001 images belonging to 2 classes.


**Note**: in this lab, we will be using the full data-set of 30,000 images for training and validation.


**Your Turn**: Use the *flow_from_directory* method to get the validation images and assign the result to **validation_generator**.



```python
## Type your answer here
validation_generator = data_generator.flow_from_directory(
    'concrete_data_week3/valid',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_validation,
    class_mode='categorical')

```

    Found 5001 images belonging to 2 classes.


Double-click __here__ for the solution.
<!-- The correct answer is:
validation_generator = data_generator.flow_from_directory(
    'concrete_data_week3/valid',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_validation,
    class_mode='categorical')
-->



<a id='item35'></a>


## Build, Compile and Fit Model


In this section, we will start building our model. We will use the Sequential model class from Keras.



```python
model = Sequential()
```

Next, we will add the ResNet50 pre-trained model to out model. However, note that we don't want to include the top layer or the output layer of the pre-trained model. We actually want to define our own output layer and train it so that it is optimized for our image dataset. In order to leave out the output layer of the pre-trained model, we will use the argument *include_top* and set it to **False**.



```python
model.add(ResNet50(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    ))
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    [1m94765736/94765736[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 0us/step


Then, we will define our output layer as a **Dense** layer, that consists of two nodes and uses the **Softmax** function as the activation function.



```python
model.add(Dense(num_classes, activation='softmax'))
```

You can access the model's layers using the *layers* attribute of our model object.



```python
model.layers
```




    [<Functional name=resnet50, built=True>, <Dense name=dense, built=True>]



You can see that our model is composed of two sets of layers. The first set is the layers pertaining to ResNet50 and the second set is a single layer, which is our Dense layer that we defined above.


You can access the ResNet50 layers by running the following:



```python
model.layers[0].layers
```




    [<InputLayer name=input_layer, built=True>,
     <ZeroPadding2D name=conv1_pad, built=True>,
     <Conv2D name=conv1_conv, built=True>,
     <BatchNormalization name=conv1_bn, built=True>,
     <Activation name=conv1_relu, built=True>,
     <ZeroPadding2D name=pool1_pad, built=True>,
     <MaxPooling2D name=pool1_pool, built=True>,
     <Conv2D name=conv2_block1_1_conv, built=True>,
     <BatchNormalization name=conv2_block1_1_bn, built=True>,
     <Activation name=conv2_block1_1_relu, built=True>,
     <Conv2D name=conv2_block1_2_conv, built=True>,
     <BatchNormalization name=conv2_block1_2_bn, built=True>,
     <Activation name=conv2_block1_2_relu, built=True>,
     <Conv2D name=conv2_block1_0_conv, built=True>,
     <Conv2D name=conv2_block1_3_conv, built=True>,
     <BatchNormalization name=conv2_block1_0_bn, built=True>,
     <BatchNormalization name=conv2_block1_3_bn, built=True>,
     <Add name=conv2_block1_add, built=True>,
     <Activation name=conv2_block1_out, built=True>,
     <Conv2D name=conv2_block2_1_conv, built=True>,
     <BatchNormalization name=conv2_block2_1_bn, built=True>,
     <Activation name=conv2_block2_1_relu, built=True>,
     <Conv2D name=conv2_block2_2_conv, built=True>,
     <BatchNormalization name=conv2_block2_2_bn, built=True>,
     <Activation name=conv2_block2_2_relu, built=True>,
     <Conv2D name=conv2_block2_3_conv, built=True>,
     <BatchNormalization name=conv2_block2_3_bn, built=True>,
     <Add name=conv2_block2_add, built=True>,
     <Activation name=conv2_block2_out, built=True>,
     <Conv2D name=conv2_block3_1_conv, built=True>,
     <BatchNormalization name=conv2_block3_1_bn, built=True>,
     <Activation name=conv2_block3_1_relu, built=True>,
     <Conv2D name=conv2_block3_2_conv, built=True>,
     <BatchNormalization name=conv2_block3_2_bn, built=True>,
     <Activation name=conv2_block3_2_relu, built=True>,
     <Conv2D name=conv2_block3_3_conv, built=True>,
     <BatchNormalization name=conv2_block3_3_bn, built=True>,
     <Add name=conv2_block3_add, built=True>,
     <Activation name=conv2_block3_out, built=True>,
     <Conv2D name=conv3_block1_1_conv, built=True>,
     <BatchNormalization name=conv3_block1_1_bn, built=True>,
     <Activation name=conv3_block1_1_relu, built=True>,
     <Conv2D name=conv3_block1_2_conv, built=True>,
     <BatchNormalization name=conv3_block1_2_bn, built=True>,
     <Activation name=conv3_block1_2_relu, built=True>,
     <Conv2D name=conv3_block1_0_conv, built=True>,
     <Conv2D name=conv3_block1_3_conv, built=True>,
     <BatchNormalization name=conv3_block1_0_bn, built=True>,
     <BatchNormalization name=conv3_block1_3_bn, built=True>,
     <Add name=conv3_block1_add, built=True>,
     <Activation name=conv3_block1_out, built=True>,
     <Conv2D name=conv3_block2_1_conv, built=True>,
     <BatchNormalization name=conv3_block2_1_bn, built=True>,
     <Activation name=conv3_block2_1_relu, built=True>,
     <Conv2D name=conv3_block2_2_conv, built=True>,
     <BatchNormalization name=conv3_block2_2_bn, built=True>,
     <Activation name=conv3_block2_2_relu, built=True>,
     <Conv2D name=conv3_block2_3_conv, built=True>,
     <BatchNormalization name=conv3_block2_3_bn, built=True>,
     <Add name=conv3_block2_add, built=True>,
     <Activation name=conv3_block2_out, built=True>,
     <Conv2D name=conv3_block3_1_conv, built=True>,
     <BatchNormalization name=conv3_block3_1_bn, built=True>,
     <Activation name=conv3_block3_1_relu, built=True>,
     <Conv2D name=conv3_block3_2_conv, built=True>,
     <BatchNormalization name=conv3_block3_2_bn, built=True>,
     <Activation name=conv3_block3_2_relu, built=True>,
     <Conv2D name=conv3_block3_3_conv, built=True>,
     <BatchNormalization name=conv3_block3_3_bn, built=True>,
     <Add name=conv3_block3_add, built=True>,
     <Activation name=conv3_block3_out, built=True>,
     <Conv2D name=conv3_block4_1_conv, built=True>,
     <BatchNormalization name=conv3_block4_1_bn, built=True>,
     <Activation name=conv3_block4_1_relu, built=True>,
     <Conv2D name=conv3_block4_2_conv, built=True>,
     <BatchNormalization name=conv3_block4_2_bn, built=True>,
     <Activation name=conv3_block4_2_relu, built=True>,
     <Conv2D name=conv3_block4_3_conv, built=True>,
     <BatchNormalization name=conv3_block4_3_bn, built=True>,
     <Add name=conv3_block4_add, built=True>,
     <Activation name=conv3_block4_out, built=True>,
     <Conv2D name=conv4_block1_1_conv, built=True>,
     <BatchNormalization name=conv4_block1_1_bn, built=True>,
     <Activation name=conv4_block1_1_relu, built=True>,
     <Conv2D name=conv4_block1_2_conv, built=True>,
     <BatchNormalization name=conv4_block1_2_bn, built=True>,
     <Activation name=conv4_block1_2_relu, built=True>,
     <Conv2D name=conv4_block1_0_conv, built=True>,
     <Conv2D name=conv4_block1_3_conv, built=True>,
     <BatchNormalization name=conv4_block1_0_bn, built=True>,
     <BatchNormalization name=conv4_block1_3_bn, built=True>,
     <Add name=conv4_block1_add, built=True>,
     <Activation name=conv4_block1_out, built=True>,
     <Conv2D name=conv4_block2_1_conv, built=True>,
     <BatchNormalization name=conv4_block2_1_bn, built=True>,
     <Activation name=conv4_block2_1_relu, built=True>,
     <Conv2D name=conv4_block2_2_conv, built=True>,
     <BatchNormalization name=conv4_block2_2_bn, built=True>,
     <Activation name=conv4_block2_2_relu, built=True>,
     <Conv2D name=conv4_block2_3_conv, built=True>,
     <BatchNormalization name=conv4_block2_3_bn, built=True>,
     <Add name=conv4_block2_add, built=True>,
     <Activation name=conv4_block2_out, built=True>,
     <Conv2D name=conv4_block3_1_conv, built=True>,
     <BatchNormalization name=conv4_block3_1_bn, built=True>,
     <Activation name=conv4_block3_1_relu, built=True>,
     <Conv2D name=conv4_block3_2_conv, built=True>,
     <BatchNormalization name=conv4_block3_2_bn, built=True>,
     <Activation name=conv4_block3_2_relu, built=True>,
     <Conv2D name=conv4_block3_3_conv, built=True>,
     <BatchNormalization name=conv4_block3_3_bn, built=True>,
     <Add name=conv4_block3_add, built=True>,
     <Activation name=conv4_block3_out, built=True>,
     <Conv2D name=conv4_block4_1_conv, built=True>,
     <BatchNormalization name=conv4_block4_1_bn, built=True>,
     <Activation name=conv4_block4_1_relu, built=True>,
     <Conv2D name=conv4_block4_2_conv, built=True>,
     <BatchNormalization name=conv4_block4_2_bn, built=True>,
     <Activation name=conv4_block4_2_relu, built=True>,
     <Conv2D name=conv4_block4_3_conv, built=True>,
     <BatchNormalization name=conv4_block4_3_bn, built=True>,
     <Add name=conv4_block4_add, built=True>,
     <Activation name=conv4_block4_out, built=True>,
     <Conv2D name=conv4_block5_1_conv, built=True>,
     <BatchNormalization name=conv4_block5_1_bn, built=True>,
     <Activation name=conv4_block5_1_relu, built=True>,
     <Conv2D name=conv4_block5_2_conv, built=True>,
     <BatchNormalization name=conv4_block5_2_bn, built=True>,
     <Activation name=conv4_block5_2_relu, built=True>,
     <Conv2D name=conv4_block5_3_conv, built=True>,
     <BatchNormalization name=conv4_block5_3_bn, built=True>,
     <Add name=conv4_block5_add, built=True>,
     <Activation name=conv4_block5_out, built=True>,
     <Conv2D name=conv4_block6_1_conv, built=True>,
     <BatchNormalization name=conv4_block6_1_bn, built=True>,
     <Activation name=conv4_block6_1_relu, built=True>,
     <Conv2D name=conv4_block6_2_conv, built=True>,
     <BatchNormalization name=conv4_block6_2_bn, built=True>,
     <Activation name=conv4_block6_2_relu, built=True>,
     <Conv2D name=conv4_block6_3_conv, built=True>,
     <BatchNormalization name=conv4_block6_3_bn, built=True>,
     <Add name=conv4_block6_add, built=True>,
     <Activation name=conv4_block6_out, built=True>,
     <Conv2D name=conv5_block1_1_conv, built=True>,
     <BatchNormalization name=conv5_block1_1_bn, built=True>,
     <Activation name=conv5_block1_1_relu, built=True>,
     <Conv2D name=conv5_block1_2_conv, built=True>,
     <BatchNormalization name=conv5_block1_2_bn, built=True>,
     <Activation name=conv5_block1_2_relu, built=True>,
     <Conv2D name=conv5_block1_0_conv, built=True>,
     <Conv2D name=conv5_block1_3_conv, built=True>,
     <BatchNormalization name=conv5_block1_0_bn, built=True>,
     <BatchNormalization name=conv5_block1_3_bn, built=True>,
     <Add name=conv5_block1_add, built=True>,
     <Activation name=conv5_block1_out, built=True>,
     <Conv2D name=conv5_block2_1_conv, built=True>,
     <BatchNormalization name=conv5_block2_1_bn, built=True>,
     <Activation name=conv5_block2_1_relu, built=True>,
     <Conv2D name=conv5_block2_2_conv, built=True>,
     <BatchNormalization name=conv5_block2_2_bn, built=True>,
     <Activation name=conv5_block2_2_relu, built=True>,
     <Conv2D name=conv5_block2_3_conv, built=True>,
     <BatchNormalization name=conv5_block2_3_bn, built=True>,
     <Add name=conv5_block2_add, built=True>,
     <Activation name=conv5_block2_out, built=True>,
     <Conv2D name=conv5_block3_1_conv, built=True>,
     <BatchNormalization name=conv5_block3_1_bn, built=True>,
     <Activation name=conv5_block3_1_relu, built=True>,
     <Conv2D name=conv5_block3_2_conv, built=True>,
     <BatchNormalization name=conv5_block3_2_bn, built=True>,
     <Activation name=conv5_block3_2_relu, built=True>,
     <Conv2D name=conv5_block3_3_conv, built=True>,
     <BatchNormalization name=conv5_block3_3_bn, built=True>,
     <Add name=conv5_block3_add, built=True>,
     <Activation name=conv5_block3_out, built=True>,
     <GlobalAveragePooling2D name=avg_pool, built=True>]



Since the ResNet50 model has already been trained, then we want to tell our model not to bother with training the ResNet part, but to train only our dense output layer. To do that, we run the following.



```python
model.layers[0].trainable = False
```

And now using the *summary* attribute of the model, we can see how many parameters we will need to optimize in order to train the output layer.



```python
model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ resnet50 (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2048</span>)                │      <span style="color: #00af00; text-decoration-color: #00af00">23,587,712</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>)                   │           <span style="color: #00af00; text-decoration-color: #00af00">4,098</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">23,591,810</span> (90.00 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">4,098</span> (16.01 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">23,587,712</span> (89.98 MB)
</pre>



Next we compile our model using the **adam** optimizer.



```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

Before we are able to start the training process, with an ImageDataGenerator, we will need to define how many steps compose an epoch. Typically, that is the number of images divided by the batch size. Therefore, we define our steps per epoch as follows:



```python
steps_per_epoch_training = len(train_generator)
steps_per_epoch_validation = len(validation_generator)
num_epochs = 2
```

Finally, we are ready to start training our model. Unlike a conventional deep learning training were data is not streamed from a directory, with an ImageDataGenerator where data is augmented in batches, we use the **fit_generator** method.



```python
fit_history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch_training,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
)
```

    /usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
      self._warn_if_super_not_called()


    Epoch 1/2
    [1m101/101[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 509ms/step - accuracy: 0.9699 - loss: 0.1035 - val_accuracy: 0.9962 - val_loss: 0.0117
    Epoch 2/2
    [1m101/101[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m62s[0m 419ms/step - accuracy: 0.9989 - loss: 0.0067 - val_accuracy: 0.9978 - val_loss: 0.0082


Now that the model is trained, you are ready to start using it to classify images.


Since training can take a long time when building deep learning models, it is always a good idea to save your model once the training is complete if you believe you will be using the model again later. You will be using this model in the next module, so go ahead and save your model.



```python
model.save('classifier_resnet_model.h5')
```

    WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 


Now, you should see the model file *classifier_resnet_model.h5* apprear in the left directory pane.


### Thank you for completing this lab!

This notebook was created by Alex Aklson. I hope you found this lab interesting and educational.


This notebook is part of a course on **Coursera** called *AI Capstone Project with Deep Learning*. If you accessed this notebook outside the course, you can take this course online by clicking [here](https://cocl.us/DL0321EN_Coursera_Week3_LAB1).



```python
model_path = 'classifier_resnet_model.h5'
```


```python
loaded_model = tensorflow.keras.models.load_model(model_path, compile=False)
```


```python
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```


```python
model.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ resnet50 (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2048</span>)                │      <span style="color: #00af00; text-decoration-color: #00af00">23,587,712</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>)                   │           <span style="color: #00af00; text-decoration-color: #00af00">4,098</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">23,600,008</span> (90.03 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">4,098</span> (16.01 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">23,587,712</span> (89.98 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Optimizer params: </span><span style="color: #00af00; text-decoration-color: #00af00">8,198</span> (32.03 KB)
</pre>




## Change Log

|  Date (YYYY-MM-DD) |  Version | Changed By  |  Change Description |
|---|---|---|---|
| 2020-09-18  | 2.0  | Shubham  |  Migrated Lab to Markdown and added to course repo in GitLab |
| 2023-01-03  | 3.0  | Artem |  Updated the file import section|



<hr>

Copyright &copy; 2020 [IBM Developer Skills Network](https://cognitiveclass.ai/?utm_source=bducopyrightlink&utm_medium=dswb&utm_campaign=bdu). This notebook and its source code are released under the terms of the [MIT License](https://bigdatauniversity.com/mit-license/).

