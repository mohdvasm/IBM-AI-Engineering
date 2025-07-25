<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" height=300 width=300 />


# Final Project: League of Legends Match Predictor


### Introduction

League of Legends, a popular multiplayer online battle arena (MOBA) game, generates extensive data from matches, providing an excellent opportunity to apply machine learning techniques to real-world scenarios. Perform the following steps to build a logistic regression model aimed at predicting the outcomes of League of Legends matches.

Use the [league_of_legends_data_large.csv](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/rk7VDaPjMp1h5VXS-cUyMg/league-of-legends-data-large.csv) file to perform the tasks.

### Step 1: Data Loading and Preprocessing

#### Task 1: Load the League of Legends dataset and preprocess it for training.

Loading and preprocessing the dataset involves reading the data, splitting it into training and testing sets, and standardizing the features. You will utilize `pandas` for data manipulation, `train_test_split` from `sklearn` for data splitting, and `StandardScaler` for feature scaling.

Note: Please ensure all the required libraries are installed and imported.

1 .Load the dataset:
Use `pd.read_csv()` to load the dataset into a pandas DataFrame.</br>
2. Split data into features and target: Separate win (target) and the remaining columns (features).</br>
   X = data.drop('win', axis=1)</br>
   y = data['win'] </br>
3 .Split the Data into Training and Testing Sets:
Use `train_test_split()` from `sklearn.model_selection` to divide the data. Set `test_size`=0.2 to allocate 20% for testing and 80% for training, and use `random_state`=42 to ensure reproducibility of the split.</br>
4. Standardize the features:
Use `StandardScaler()` from sklearn.preprocessing to scale the features.</br>
5. Convert to PyTorch tensors:
Use `torch.tensor()` to convert the data to PyTorch tensors.

#### Exercise 1:

Write a code to load the dataset, split it into training and testing sets, standardize the features, and convert the data into PyTorch tensors for use in training a PyTorch model.


### Setup
Installing required libraries:

The following required libraries are not pre-installed in the Skills Network Labs environment. You will need to run the following cell to install them:



```python
# !pip install pandas
# !pip install scikit-learn
# !pip install torch
# !pip install matplotlib

```


```python
## Write your code here
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, roc_curve, auc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
```

    cuda:0



```python
# Task 1: Load and Preprocess Data
df = pd.read_csv("/content/league_of_legends_data_large.csv")  # Change filename accordingly

# Define features and target
y = df['win']  # Target variable
X = df.drop(columns=['win'])  # Features

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)

X_train.shape, y_train.shape, X_test.shape, y_test.shape
```




    (torch.Size([800, 8]),
     torch.Size([800, 1]),
     torch.Size([200, 8]),
     torch.Size([200, 1]))



### Step 2: Logistic Regression Model

#### Task 2: Implement a logistic regression model using PyTorch.

Defining the logistic regression model involves specifying the input dimensions, the forward pass using the sigmoid activation function, and initializing the model, loss function, and optimizer.

1 .Define the Logistic Regression Model:</br>
  Create a class LogisticRegressionModel that inherits from torch.nn.Module.</br>
 - In the `__init__()` method, define a linear layer (nn.Linear) to implement the logistic regression model.</br>
- The `forward()` method should apply the sigmoid activation function to the output of the linear layer.</br>

2.Initialize the Model, Loss Function, and Optimizer:</br>
- Set input_dim: Use `X_train.shape[1]` to get the number of features from the training data (X_train).</br>
- Initialize the model: Create an instance of the LogisticRegressionModel class  (e.g., `model = LogisticRegressionModel()`)while passing input_dim as a parameter</br>
- Loss Function: Use `BCELoss()` from torch.nn (Binary Cross-Entropy Loss).</br>
- Optimizer: Initialize the optimizer using `optim.SGD()` with a learning rate of 0.01</br>

#### Exercise 2:

Define the logistic regression model using PyTorch, specifying the input dimensions and the forward pass. Initialize the model, loss function, and optimizer.



```python
## Task: 2

## Write your code here
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

# Initialize model, loss function, and optimizer
model = LogisticRegressionModel(X_train.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

model.state_dict()
```




    OrderedDict([('linear.weight',
                  tensor([[ 0.1997,  0.2898, -0.1500, -0.1232, -0.1473,  0.1343, -0.1207,  0.2071]],
                         device='cuda:0')),
                 ('linear.bias', tensor([-0.2240], device='cuda:0'))])



### Step 3: Model Training

#### Task 3: Train the logistic regression model on the dataset.

The training loop will run for a specified number of epochs. In each epoch, the model makes predictions, calculates the loss, performs backpropagation, and updates the model parameters.

1. Set Number of Epochs:  
   - Define the number of epochs for training to 1000.

2. Training Loop:  
   For each epoch:
   - Set the model to training mode using `model.train()`.
   - Zero the gradients using `optimizer.zero_grad()`.
   - Pass the training data (`X_train`) through the model to get the predictions (`outputs`).
   - Calculate the loss using the defined loss function (`criterion`).
   - Perform backpropagation with `loss.backward()`.
   - Update the model's weights using `optimizer.step()`.

3. Print Loss Every 100 Epochs:  
   - After every 100 epochs, print the current epoch number and the loss value.

4. Model Evaluation:  
   - Set the model to evaluation mode using `model.eval()`.
   - Use `torch.no_grad()` to ensure no gradients are calculated during evaluation.
   - Get predictions on both the training set (`X_train`) and the test set (`X_test`).

5. Calculate Accuracy:  
   - For both the training and test datasets, compute the accuracy by comparing the predicted values with the true values (`y_train`, `y_test`).
   - Use a threshold of 0.5 for classification
   
6. Print Accuracy:  
   - Print the training and test accuracies after the evaluation is complete.

#### Exercise 3:

Write the code to train the logistic regression model on the dataset. Implement the training loop, making predictions, calculating the loss, performing backpropagation, and updating model parameters. Evaluate the model's accuracy on training and testing sets.



```python
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()

    # Evaluate accuracy every 100 epochs
    if epoch % 100 == 0:
        with torch.inference_mode():

            # Convert predictions to 0 or 1 for accuracy calculation
            y_pred_train = (predictions >= 0.5).float()
            train_accuracy = (y_pred_train.eq(y_train).sum().item()) / y_train.size(0)

            model.eval()
            y_pred_test = (model(X_test) >= 0.5).float()
            test_accuracy = (y_pred_test.eq(y_test).sum().item()) / y_test.size(0)
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Train Accuracy={train_accuracy:.4f} Test Accuracy={test_accuracy:.4f}")
```

    Epoch 0: Loss=0.7261, Train Accuracy=0.5000 Test Accuracy=0.4500
    Epoch 100: Loss=0.7109, Train Accuracy=0.5062 Test Accuracy=0.4400
    Epoch 200: Loss=0.7011, Train Accuracy=0.5012 Test Accuracy=0.4450
    Epoch 300: Loss=0.6951, Train Accuracy=0.5100 Test Accuracy=0.4500
    Epoch 400: Loss=0.6913, Train Accuracy=0.5225 Test Accuracy=0.4650
    Epoch 500: Loss=0.6890, Train Accuracy=0.5387 Test Accuracy=0.5050
    Epoch 600: Loss=0.6875, Train Accuracy=0.5387 Test Accuracy=0.5000
    Epoch 700: Loss=0.6867, Train Accuracy=0.5450 Test Accuracy=0.5000
    Epoch 800: Loss=0.6861, Train Accuracy=0.5400 Test Accuracy=0.5050
    Epoch 900: Loss=0.6858, Train Accuracy=0.5450 Test Accuracy=0.5150


### Step 4: Model Optimization and Evaluation

#### Task 4: Implement optimization techniques and evaluate the model's performance.

Optimization techniques such as L2 regularization (Ridge Regression) help in preventing overfitting. The model is retrained with these optimizations, and its performance is evaluated on both training and testing sets.

**Weight Decay** :In the context of machine learning and specifically in optimization algorithms, weight_decay is a parameter used to apply L2 regularization to the model's parameters (weights). It helps prevent the model from overfitting by penalizing large weight values, thereby encouraging the model to find simpler solutions.To use L2 regularization, you need to modify the optimizer by setting the weight_decay parameter. The weight_decay parameter in the optimizer adds the L2 regularization term during training.
For example, when you initialize the optimizer with optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01), the weight_decay=0.01 term applies L2 regularization with a strength of 0.01.

1. Set Up the Optimizer with L2 Regularization:
   - Modify the optimizer to include `weight_decay` for L2 regularization.
   - Example:
     ```python
     optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
     ```
2. Train the Model with L2 Regularization:
    - Follow the same steps as before but use the updated optimizer with regularization during training.
    - Use epochs=1000
   
3. Evaluate the Optimized Model:
   - After training, evaluate the model on both the training and test datasets.
   - Compute the accuracy for both sets by comparing the model's predictions to the true labels (`y_train` and `y_test`).

4. Calculate and Print the Accuracy:
   - Use a threshold of 0.5 to determine whether the model's predictions are class 0 or class 1.
   - Print the training accuracy and test accuracy  after evaluation.


#### Exercise 4:

Implement optimization techniques like L2 regularization and retrain the model. Evaluate the performance of the optimized model on both training and testing sets.



```python
## Write your code here
model = LogisticRegressionModel(X_train.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)

model.state_dict()
```




    OrderedDict([('linear.weight',
                  tensor([[-0.1881, -0.3324,  0.3427, -0.0870,  0.3045,  0.2993, -0.2008, -0.2945]],
                         device='cuda:0')),
                 ('linear.bias', tensor([0.1292], device='cuda:0'))])




```python
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    loss.backward()
    optimizer.step()

    # Evaluate accuracy every 100 epochs
    if epoch % 100 == 0:
        with torch.inference_mode():

            # Convert predictions to 0 or 1 for accuracy calculation
            y_pred_train = (predictions >= 0.5).float()
            train_accuracy = (y_pred_train.eq(y_train).sum().item()) / y_train.size(0)

            model.eval()
            y_pred_test = (model(X_test) >= 0.5).float()
            test_accuracy = (y_pred_test.eq(y_test).sum().item()) / y_test.size(0)
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Train Accuracy={train_accuracy:.4f} Test Accuracy={test_accuracy:.4f}")
```

    Epoch 0: Loss=0.7665, Train Accuracy=0.4963 Test Accuracy=0.5750
    Epoch 100: Loss=0.7353, Train Accuracy=0.4938 Test Accuracy=0.5550
    Epoch 200: Loss=0.7154, Train Accuracy=0.5050 Test Accuracy=0.5550
    Epoch 300: Loss=0.7032, Train Accuracy=0.5075 Test Accuracy=0.5400
    Epoch 400: Loss=0.6959, Train Accuracy=0.5175 Test Accuracy=0.5500
    Epoch 500: Loss=0.6916, Train Accuracy=0.5275 Test Accuracy=0.5500
    Epoch 600: Loss=0.6890, Train Accuracy=0.5375 Test Accuracy=0.5200
    Epoch 700: Loss=0.6875, Train Accuracy=0.5513 Test Accuracy=0.5300
    Epoch 800: Loss=0.6866, Train Accuracy=0.5513 Test Accuracy=0.5150
    Epoch 900: Loss=0.6860, Train Accuracy=0.5513 Test Accuracy=0.5200


### Step 5: Visualization and Interpretation

Visualization tools like confusion matrices and ROC curves provide insights into the model's performance. The confusion matrix helps in understanding the classification accuracy, while the ROC curve illustrates the trade-off between sensitivity and specificity.

Confusion Matrix : A Confusion Matrix is a fundamental tool used in classification problems to evaluate the performance of a model. It provides a matrix showing the number of correct and incorrect predictions made by the model, categorized by the actual and predicted classes.
Where
-  True Positive (TP): Correctly predicted positive class (class 1).
- True Negative (TN): Correctly predicted negative class (class 0).
- False Positive (FP): Incorrectly predicted as positive (class 1), but the actual class is negative (class 0). This is also called a Type I error.
- False Negative (FN): Incorrectly predicted as negative (class 0), but the actual class is positive (class 1). This is also called a Type II error.

ROC Curve (Receiver Operating Characteristic Curve):
The ROC Curve is a graphical representation used to evaluate the performance of a binary classification model across all classification thresholds. It plots two metrics:
- True Positive Rate (TPR) or Recall (Sensitivity)-It is the proportion of actual positive instances (class 1) that were correctly classified as positive by the model.
- False Positive Rate (FPR)-It is the proportion of actual negative instances (class 0) that were incorrectly classified as positive by the model.
  
AUC:
AUC stands for Area Under the Curve and is a performance metric used to evaluate the quality of a binary classification model. Specifically, it refers to the area under the ROC curve (Receiver Operating Characteristic curve), which plots the True Positive Rate (TPR) versus the False Positive Rate (FPR) for different threshold values.

Classification Report:
A Classification Report is a summary of various classification metrics, which are useful for evaluating the performance of a classifier on the given dataset.

#### Exercise 5:

Write code to visualize the model's performance using confusion matrices and ROC curves. Generate classification reports to evaluate precision, recall, and F1-score. Retrain the model with L2 regularization and evaluate the performance.



```python
## Write your code here

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import itertools

# Visualize the confusion matrix
#Change the variable names as used in your code
y_pred_test_labels = (y_pred_test.to('cpu') > 0.5).float()
cm = confusion_matrix(y_test.to('cpu'), y_pred_test_labels)

plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = range(2)
plt.xticks(tick_marks, ['Loss', 'Win'], rotation=45)
plt.yticks(tick_marks, ['Loss', 'Win'])

thresh = cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Print classification report
print("Classification Report:\n", classification_report(y_test.to('cpu'), y_pred_test_labels, target_names=['Loss', 'Win']))

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test.to('cpu'), y_pred_test.to('cpu'))
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
```


    
![png](Final_Project_League_of_Legends_Match_Predictor_v2_colab_files/Final_Project_League_of_Legends_Match_Predictor_v2_colab_15_0.png)
    


    Classification Report:
                   precision    recall  f1-score   support
    
            Loss       0.49      0.39      0.44        95
             Win       0.54      0.64      0.58       105
    
        accuracy                           0.52       200
       macro avg       0.51      0.51      0.51       200
    weighted avg       0.52      0.52      0.51       200
    



    
![png](Final_Project_League_of_Legends_Match_Predictor_v2_colab_files/Final_Project_League_of_Legends_Match_Predictor_v2_colab_15_2.png)
    


Double-click <b>here</b> for the Hint.
<!--

#Change the name of variables as per your code
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import itertools

# Visualize the confusion matrix
#Change the variable names as used in your code
y_pred_test_labels = (y_pred_test > 0.5).float()
cm = confusion_matrix(y_test, y_pred_test_labels)

plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = range(2)
plt.xticks(tick_marks, ['Loss', 'Win'], rotation=45)
plt.yticks(tick_marks, ['Loss', 'Win'])

thresh = cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred_test_labels, target_names=['Loss', 'Win']))

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
-->


### Step 6: Model Saving and Loading

#### Task 6: Save and load the trained model.

This task demonstrates the techniques to persist a trained model using `torch.save` and reload it using `torch.load`. Evaluating the loaded model ensures that it retains its performance, making it practical for deployment in real-world applications.

1. Saving the Model:
- Save the model's learned weights and biases using torch.save().( e.g. , torch.save(model.state_dict(), 'your_model_name.pth'))
- Saving only the state dictionary (model parameters) is preferred because it’s more flexible and efficient than saving the entire model object.

2. Loading the Model:
- Create a new model instance (e.g., `model = LogisticRegressionModel()`) and load the saved parameters. ( e.g. , `model.load_state_dict(torch.load('your_model_name.pth'))`)`.

3. Evaluating the Loaded Model:
   - After loading, set the model to evaluation mode by calling `model.eval()
   - After loading the model, evaluate it again on the test dataset to make sure it performs similarly to when it was first trained..Now evaluate it on the test data.
   - Use `torch.no_grad()` to ensure that no gradients are computed.

#### Exercise 6:

Write code to save the trained model and reload it. Ensure the loaded model performs consistently by evaluating it on the test dataset.



```python
## Write your code here
# Save the model
torch.save(model.state_dict(), 'model.pth')
```


```python
# Load the model
model = LogisticRegressionModel(X_train.shape[1]).to(device)
model.load_state_dict(torch.load('model.pth'))

# Ensure the loaded model is in evaluation mode
model.eval()

# Evaluate the loaded model
with torch.inference_mode():
  y_pred_test = model(X_test)
  y_pred_test_labels = (y_pred_test >= 0.5).float()
  test_accuracy = (y_pred_test_labels.eq(y_test).sum().item()) / y_test.size(0)
  print(f"Test Accuracy: {test_accuracy:.4f}")
```

    Test Accuracy: 0.5200


    <ipython-input-15-dec1d9dba775>:3: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      model.load_state_dict(torch.load('model.pth'))


### Step 7: Hyperparameter Tuning

#### Task 7: Perform hyperparameter tuning to find the best learning rate.

By testing different learning rates, you will identify the optimal rate that provides the best test accuracy. This fine-tuning is crucial for enhancing model performance .
1. Define Learning Rates:
   - Choose these learning rates to test ,[0.01, 0.05, 0.1]

2. Reinitialize the Model for Each Learning Rate:
  - For each learning rate, you’ll need to reinitialize the model and optimizer e.g.(`torch.optim.SGD(model.parameters(), lr=lr)`).
   - Each new learning rate requires reinitializing the model since the optimizer and its parameters are linked to the learning rate.

3. Train the Model for Each Learning Rate:
  - Train the model for a fixed number of epochs (e.g., 50 or 100 epochs) for each learning rate, and compute the accuracy on the test set.
  - Track the test accuracy for each learning rate and identify which one yields the best performance.

4. Evaluate and Compare:
  - After training with each learning rate, compare the test accuracy for each configuration.
   - Report the learning rate that gives the highest test accuracy

#### Exercise 7:

Perform hyperparameter tuning to find the best learning rate. Retrain the model for each learning rate and evaluate its performance to identify the optimal rate.



```python
## Write your code here
lr_rates = [0.01, 0.05, 0.1]
epochs = 1000

for idx, lr_rate in enumerate(lr_rates):

  print(f"\n\nTraining: {idx}, Learning rate: {lr_rate}")

  model = LogisticRegressionModel(X_train.shape[1]).to(device)
  criterion = nn.BCELoss()
  optimizer = optim.SGD(model.parameters(), lr=lr_rate, weight_decay=0.01)

  for epoch in range(epochs):
      model.train()
      optimizer.zero_grad()
      predictions = model(X_train)
      loss = criterion(predictions, y_train)
      loss.backward()
      optimizer.step()

      # Evaluate accuracy every 100 epochs
      if epoch % 100 == 0:
          with torch.inference_mode():

              # Convert predictions to 0 or 1 for accuracy calculation
              y_pred_train = (predictions >= 0.5).float()
              train_accuracy = (y_pred_train.eq(y_train).sum().item()) / y_train.size(0)

              model.eval()
              y_pred_test = (model(X_test) >= 0.5).float()
              test_accuracy = (y_pred_test.eq(y_test).sum().item()) / y_test.size(0)
              print(f"Epoch {epoch}: Loss={loss.item():.4f}, Train Accuracy={train_accuracy:.4f} Test Accuracy={test_accuracy:.4f}")
```

    
    
    Training: 0, Learning rate: 0.01
    Epoch 0: Loss=0.7392, Train Accuracy=0.5075 Test Accuracy=0.5350
    Epoch 100: Loss=0.7191, Train Accuracy=0.5062 Test Accuracy=0.5200
    Epoch 200: Loss=0.7063, Train Accuracy=0.5062 Test Accuracy=0.5250
    Epoch 300: Loss=0.6983, Train Accuracy=0.5212 Test Accuracy=0.5250
    Epoch 400: Loss=0.6933, Train Accuracy=0.5337 Test Accuracy=0.5450
    Epoch 500: Loss=0.6903, Train Accuracy=0.5337 Test Accuracy=0.5350
    Epoch 600: Loss=0.6884, Train Accuracy=0.5350 Test Accuracy=0.5550
    Epoch 700: Loss=0.6872, Train Accuracy=0.5437 Test Accuracy=0.5300
    Epoch 800: Loss=0.6865, Train Accuracy=0.5525 Test Accuracy=0.5300
    Epoch 900: Loss=0.6861, Train Accuracy=0.5475 Test Accuracy=0.5200
    
    
    Training: 1, Learning rate: 0.05
    Epoch 0: Loss=0.7233, Train Accuracy=0.4763 Test Accuracy=0.5100
    Epoch 100: Loss=0.6881, Train Accuracy=0.5288 Test Accuracy=0.4750
    Epoch 200: Loss=0.6855, Train Accuracy=0.5487 Test Accuracy=0.4900
    Epoch 300: Loss=0.6853, Train Accuracy=0.5463 Test Accuracy=0.5100
    Epoch 400: Loss=0.6853, Train Accuracy=0.5450 Test Accuracy=0.5050
    Epoch 500: Loss=0.6853, Train Accuracy=0.5437 Test Accuracy=0.5050
    Epoch 600: Loss=0.6853, Train Accuracy=0.5437 Test Accuracy=0.5050
    Epoch 700: Loss=0.6853, Train Accuracy=0.5437 Test Accuracy=0.5050
    Epoch 800: Loss=0.6853, Train Accuracy=0.5437 Test Accuracy=0.5050
    Epoch 900: Loss=0.6853, Train Accuracy=0.5437 Test Accuracy=0.5050
    
    
    Training: 2, Learning rate: 0.1
    Epoch 0: Loss=0.7172, Train Accuracy=0.5075 Test Accuracy=0.4850
    Epoch 100: Loss=0.6854, Train Accuracy=0.5363 Test Accuracy=0.4900
    Epoch 200: Loss=0.6853, Train Accuracy=0.5425 Test Accuracy=0.5050
    Epoch 300: Loss=0.6853, Train Accuracy=0.5437 Test Accuracy=0.5100
    Epoch 400: Loss=0.6853, Train Accuracy=0.5437 Test Accuracy=0.5050
    Epoch 500: Loss=0.6853, Train Accuracy=0.5437 Test Accuracy=0.5050
    Epoch 600: Loss=0.6853, Train Accuracy=0.5437 Test Accuracy=0.5050
    Epoch 700: Loss=0.6853, Train Accuracy=0.5437 Test Accuracy=0.5050
    Epoch 800: Loss=0.6853, Train Accuracy=0.5437 Test Accuracy=0.5050
    Epoch 900: Loss=0.6853, Train Accuracy=0.5437 Test Accuracy=0.5050


### Step 8: Feature Importance

#### Task 8: Evaluate feature importance to understand the impact of each feature on the prediction.

The code to evaluate feature importance to understand the impact of each feature on the prediction.

 1.Extracting Model Weights:
  - The weights of the logistic regression model represent the importance of each feature in making predictions. These weights are stored in the model's linear layer (`model.linear.weight`).
 - You can extract the weights using `model.linear.weight.data.numpy()` and flatten the resulting tensor to get a 1D array of feature importances.

2.Creating a DataFrame:
 - Create a pandas DataFrame with two columns: one for the feature names and the other for their corresponding importance values (i.e., the learned weights).
 - Ensure the features are aligned with their names in your dataset (e.g., `X_train.columns).

3. Sorting and Plotting Feature Importance:
  - Sort the features based on the absolute value of their importance (weights) to identify the most impactful features.
  - Use a bar plot (via `matplotlib`) to visualize the sorted feature importances, with the feature names on the y-axis and importance values on the x-axis.

4. Interpreting the Results:
  - Larger absolute weights indicate more influential features. Positive weights suggest a positive correlation with the outcome (likely to predict the positive class), while negative weights suggest the opposite.

#### Exercise 8:

Evaluate feature importance by extracting the weights of the linear layer and creating a DataFrame to display the importance of each feature. Visualize the feature importance using a bar plot.



```python
## Write your code here

import pandas as pd
import matplotlib.pyplot as plt

# Extract the weights of the linear layer
## Write your code here
weights = model.to('cpu').linear.weight.data.numpy().flatten()
features = X.columns
print(f"Weights: {weights}, Features: {features}")

# Create a DataFrame for feature importance
## Write your code here
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': weights})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

plt.figure(figsize=(10,5))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

```

    Weights: [ 0.12611817 -0.00445964 -0.01299391  0.16853996 -0.00707707  0.10214223
     -0.0326108  -0.01763816], Features: Index(['kills', 'deaths', 'assists', 'gold_earned', 'cs', 'wards_placed',
           'wards_killed', 'damage_dealt'],
          dtype='object')
            Feature  Importance
    3   gold_earned    0.168540
    0         kills    0.126118
    5  wards_placed    0.102142
    1        deaths   -0.004460
    4            cs   -0.007077
    2       assists   -0.012994
    7  damage_dealt   -0.017638
    6  wards_killed   -0.032611



    
![png](Final_Project_League_of_Legends_Match_Predictor_v2_colab_files/Final_Project_League_of_Legends_Match_Predictor_v2_colab_23_1.png)
    


Double-click <b>here</b> for the Hint
<!--
#Use the following code to extract the weight and create dataframe
#Change the name of variables per your code

Extract the weights of the linear layer:
weights = model.linear.weight.data.numpy().flatten()
features = X.columns
Create a DataFrame for feature importance:
feature_importance = pd.DataFrame({'Feature': features, 'Importance': weights})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print(feature_importance)
Plot feature importance plt.figure(figsize=(10, 6))
plt.bar(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.show()
-->


#### Conclusion:

Congratulations on completing the project! In this final project, you built a logistic regression model to predict the outcomes of League of Legends matches based on various in-game statistics. This comprehensive project involved several key steps, including data loading and preprocessing, model implementation, training, optimization, evaluation, visualization, model saving and loading, hyperparameter tuning, and feature importance analysis. This project provided hands-on experience with the complete workflow of developing a machine learning model for binary classification tasks using PyTorch.

© Copyright IBM Corporation. All rights reserved.



```python

```
