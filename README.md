December 2021

Project Astro


---

## Overview:

This repository is the groundwork of an up and coming mobile app called Astro!
For this repo it is just focusing on the machine learning aspect of the app where it performs Human-Activity-Recognition on a relatively basic level using sensor data from smartphones
This is done by people performing their daily life activities while carrying a waist-mounted smartphone that has accelerometer and gyroscope sensors

NOTE FOR PARKER-- SEE IF YOU NEED THE LSTM MODEL AND THEN UPDATE THE PART RIGHT BELOW THIS AS YOU SEE FIT, THEN START WORKING ON THE COMMANDS FOR HOW TO INSTALL EACH OF THE DEPENDENCIES

<br><br>
The repository has 3 ipython notebook
<br>
1 [PRE_PROCESS_DATAipynb](https://github.com/lmu-mandy/projects-astro/blob/main/PRE_PROCESS_DATA.ipynb) : Data pre-processing and Analysis
<br>
2 [ML_MODELS.ipynb](https://github.com/lmu-mandy/projects-astro/blob/main/ML_MODELS.ipynb) : All of the Machine Learning models
<br>
<br><br>
All the code is written in python 3.7 and can be ran with JupyterLab or VSCode with your virtual enviroment set up<br><br>
To download code onto your system, pull up your terminal and type: `git clone https://github.com/lmu-mandy/projects-astro.git`
** After you have cloned the repo, read below to see how to install all of the dependencies to make the code work on your system **

- [Brew](https://brew.sh/)
- [Anaconda](https://medium.com/ayuth/install-anaconda-on-macos-with-homebrew-c94437d63a37)
- NOTE: if you have an M1 Mac, you may need to wait longer to use this repo or use good ol Google 

** Then run the followiung comands in your terminal **

- Python 3.7 `conda install python=3.7`
- numpy : `conda install numpy`
- pandas : `conda install -c anaconda pandas`
- tensorflow : `conda install -c conda-forge tensorflow`
- keras : `conda install -c conda-forge keras`
- matplotlib : `conda install -c conda-forge matplotlib`
- seaborn : `conda install -c anaconda seaborn`
- sklearn : `pip install -U scikit-learn`
- itertools : `conda install -c anaconda more-itertools`
- datetime : `conda install -c trentonoliphant datetime`

** NOTE: if the code is still not running, make sure you have your virtual environment set up correctly: https://docs.python.org/3/library/venv.html **

## Introduction:

Every modern Smart Phone has a number of [sensors](https://www.gsmarena.com/glossary.php3?term=sensors). we are interested in two of the sensors Accelerometer and Gyroscope.
<br>
The data is recorded with the help of sensors
<br>
This is a 6 class classification problem as we have 6 activities to detect.<br>

This project has two parts, the first part trains, tunes and compares Logistic Regression, Linear support vector classifier, RBF(Radial Basis Function) SVM classifier, Decision Tree, and Random Forest Classifier <br>
The second part uses the raw time series windowed data to train (Long Short term Memory)LSTM models. The LSTM models are semi tuned manually to fast forward the tuning task.
This code was actually adapted from https://github.com/srvds/Human-Activity-Recognition and I do have to give them full credit for coming up with the right window to train the models. In the future, I plan to further fine tune this model to work with wearable sensors instead

---

## Dataset:

The Dataset comes from University of California Irvine where it can be downloaded from:
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones#

As of now, this repo only trains and tests 30 people from the dataset
As we go into 2022, I hope to include a much larger train and test sample so that way we get a feel of what it would be like in the real world in terms of accuracy

<br>
**The Activities that this repository can currently predict are:**
* Walking 
* Sitting 
* Standing 
* Walking Upstairs 
* Walking Downstairs 
* Laying Down

[**Accelerometers**](https://en.wikipedia.org/wiki/Accelerometer) detects the magnitude and direction of the acceleration as a vector quantity and can be used to help sense orientation due to direction of weight changes
<br><br>
[**GyroScope**](https://en.wikipedia.org/wiki/Gyroscope) maintains orientation along an axis so that when the device is tilted or rotated, it doesnt affect the orientation

<br><br>
Accelerometer measures the directional movement of a device but will not be able to resolve its lateral orientation or tilt during that movement accurately unless a gyro is there to fill in that info.
<br>
With an accelerometer you can either get a really "noisy" info output that is responsive, or you can get a "clean" output that's sluggish. But when you combine the 3-axis accelerometer with a 3-axis gyro, you get an output that is both clean and responsive in the same time.
<br><br>

#### Understanding the dataset

- Both sensors generate data in 3 Dimensional space over time. Hence the data captured are '3-axial linear acceleration'(_tAcc-XYZ_) from accelerometer and '3-axial angular velocity' (_tGyro-XYZ_) from Gyroscope with several variations.
- prefix 't' in those metrics denotes time.
- suffix 'XYZ' represents 3-axial signals in X , Y, and Z directions.
- The available data is pre-processed by applying noise filters and then sampled in fixed-width windows(sliding windows) of 2.56 seconds each with 50% overlap. ie., each window has 128 readings.

#### Featurization

For each window a feature vector was obtained by calculating variables from the time and frequency domain. each datapoint represents a window with different readings.<br>
Readings are divided into a window of 2.56 seconds with 50% overlapping.

- Accelerometer readings are divided into gravity acceleration and body acceleration readings,
  which has x,y and z components each.

- Gyroscope readings are the measure of angular velocities which has x,y and z components.

- Jerk signals are calculated for BodyAcceleration readings.

- Fourier Transforms are made on the above time readings to obtain frequency readings.

- Now, on all the base signal readings., mean, max, mad, sma, arcoefficient, engerybands,entropy etc., are calculated for each window.

- We get a feature vector of 561 features and these features are given in the dataset.

- Each window of readings is a datapoint of 561 features,and we have 10299 readings.

- These are the signals that we got so far.(prefix t means time domain data, prefix f means frequency domain data)

#### Train and test data were saperated

- The readings from **_70%_** of the volunteers(21 people) were taken as **_trianing data_** and remaining **_30%_** volunteers recordings(9 people) were taken for **_test data_**

* All the data is present in 'UCI_HAR_DATA/' folder in present working directory.
  - Feature names are present in 'UCI_HAR_DATA/features.txt'
  - **_Train Data_** (7352 readings)
    - 'UCI_HAR_DATA/train/X_train.txt'
    - 'UCI_HAR_DATA/train/subject_train.txt'
    - 'UCI_HAR_DATA/train/y_train.txt'
  - **_Test Data_** (2947 readinds)
    - 'UCI_HAR_DATA/test/X_test.txt'
    - 'UCI_HAR_DATA/test/subject_test.txt'
    - 'UCI_HAR_DATA/test/y_test.txt'

---


#### Check for Imbalanced class<br>

if some class have too little or too large numbers of values compared to rest of the classes than the dataset is imbalanced.<br>
**Plot-1**
<br>
<img src="p1.png" height=500 width=700>
<br><br>
In this plot on the X-axis we have subjects(volunteers) 1 to 30. Each color represents an activity<br>
On the y-axis we have amount of data for each activity by provided by each subject.<br>
**Plot-2**
<br>
<img src="p2.png">
<br><br>
From plot1 and plot2 it is clear that dataset is almost balanced.<br>

#### Variable analysis

**Plot-3**
<br>

```python
sns.set_palette("Set1", desat=0.80)
facetgrid = sns.FacetGrid(train, hue='ActivityName', size=6,aspect=1)
facetgrid.map(sns.distplot,'tBodyAccMagmean', hist=False).add_legend()
plt.annotate("Stationary Activities", xy=(-0.956,14), xytext=(-0.9, 23), size=20,\
            va='center', ha='left',\
            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))

plt.annotate("Moving Activities", xy=(0,3), xytext=(0.2, 9), size=20,\
            va='center', ha='left',\
            arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))
plt.show()
```

<br>
<img src="p3.png">
The above plot is of tBodyAccMagmean which is mean values of magnitude of acceleration in time space. <br>

**Plot-4**
<br>Box plot, mean of magnitude of an acceleration
<br>

```python
plt.figure(figsize=(7,7))
sns.boxplot(x='ActivityName', y='tBodyAccMagmean',data=train, showfliers=False, saturation=1)
plt.ylabel('Acceleration Magnitude mean')
plt.axhline(y=-0.7, xmin=0.1, xmax=0.9,dashes=(5,5), c='g')
plt.axhline(y=-0.05, xmin=0.4, dashes=(5,5), c='m')
plt.xticks(rotation=90)
plt.show()
```

<img src="p4.png">
<br>
From the two plots above we can see that stationary activities can be linearly separated from activities with motion.
<br>


---

## Models

#### Machine Learning Algorithms

scikit-learn is used for all the 6 alogorithms listed below.<br>
Hyperparameters of all models were tuned by grid search CV<br>
Models fitted:<br>

- Logistic Regression
- Linear Support Vector Classifier(SVC)
- Radial Basis Function (RBF) kernel SVM classifier
- Decision Tree
- Random Forest

#### Models Comparisions

| model               | Accuracy | Error  |
| ------------------- | -------- | ------ |
| Logistic Regression | 96.27%   | 3.733% |
| Linear SVC          | 96.61%   | 3.393% |
| rbf SVM classifier  | 96.27%   | 3.733% |
| Decision Tree       | 86.43%   | 13.57% |
| Random Forest       | 91.31%   | 8.687% |

**Logistic Regression**

**Plot-6**

Normalized confusion matrix for Linear Regression Model

<img src="p8.png">

Diagonal Value of 1 means 100% accuracy for that class, and 0 means 0% accuracy.<br>
considering the diagonal elements we have value 1 for rows corresponding to 'Laying' and 'Walking'.<br>
while 'sitting' has value of only 0.87. In the row 2nd row and 3rd column we have value 0.12 which basically means about 12% readings of the class sitting is misclassified as standing.



#### LSTM Model

keras with tensorflow backend is used.<br>
LSTM models need large amount of data to train properly, we also need to be cautious not to overfit.<br>

> The raw series data is used to train the LSTM models, and not the heavily featured data.
> We don't want to reduce the data available to train the model hence the test dataset is used as validation data.<br>
> dropout Layers used to keep overfitting in check.

Initialization of some of the parameters

```python
timesteps = len(X_train[0])
input_dim = len(X_train[0][0])
n_classes = _count_classes(Y_train)

print(timesteps)
print(input_dim)
print(len(X_train))
```

```
128
9
7352
```

**LSTM model 1**

This is a single LSTM(128) model

```python

# Initiliazing the sequential model
model = Sequential()
# Configuring the parameters
model.add(LSTM(n_hidden, input_shape=(timesteps, input_dim)))
# Adding Batchnormalization
model.add(BatchNormalization())
# Adding a dropout layer
model.add(Dropout(pv))
# Adding a dense output layer with sigmoid activation
model.add(Dense(n_classes, activation='sigmoid'))
model.summary()
```

```python
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm_30 (LSTM)               (None, 128)               70656
_________________________________________________________________
batch_normalization_10 (Batc (None, 128)               512
_________________________________________________________________
dropout_29 (Dropout)         (None, 128)               0
_________________________________________________________________
dense_25 (Dense)             (None, 6)                 774
=================================================================
Total params: 71,942
Trainable params: 71,686
Non-trainable params: 256
_________________________________________________________________
```

LSTM models require large amount of compute power.
The following parameters are selected after some experimental runs to get a good accuracy.
This is thanks to all of the people who were working on Hunan Activity Recognition 

```python
epochs = 30
batch_size = 32
n_hidden = 128
pv = 0.25 # keep probability of dropout layer
```

With this simple LSTM(128) architecture we got 93.75% accuracy and a loss of 0.22
<br>
Confusion Matrix

| Pred /True         | LAYING | SITTING | STANDING | WALKING | WALKING_DOWNSTAIRS | WALKING_UPSTAIRS |
| ------------------ | ------ | ------- | -------- | ------- | ------------------ | ---------------- |
| LAYING             | 537    | 0       | 0        | 0       | 0                  | 0                |
| SITTING            | 5      | 390     | 93       | 0       | 0                  | 3                |
| STANDING           | 0      | 96      | 436      | 0       | 0                  | 0                |
| WALKING            | 0      | 1       | 0        | 473     | 10                 | 12               |
| WALKING_DOWNSTAIRS | 0      | 0       | 0        | 0       | 420                | 0                |
| WALKING_UPSTAIRS   | 0      | 0       | 0        | 0       | 1                  | 470              |

**LSTM model 2**

This model has 2 LSTM layers
LSTM(128) and LSTM(64) stacked.

```python

# Initiliazing the sequential model
model1 = Sequential()
# Configuring the parameters
model1.add(LSTM(n_hidden1, return_sequences=True, input_shape=(timesteps, input_dim)))
# dropout layer
model1.add(Dropout(pv1))

model1.add(LSTM(n_hidden2))
#  dropout layer
model1.add(Dropout(pv2))
# output layer with sigmoid activation
model1.add(Dense(n_classes, activation='sigmoid'))
model1.summary()
```

```python


## References:

https://en.wikipedia.org/wiki/Gyroscope <br>
https://scikit-learn.org/stable/supervised_learning.html#supervised-learning <br>
http://colah.github.io/posts/2015-08-Understanding-LSTMs/ <br>
https://keras.io/getting-started/sequential-model-guide/ <br>
https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/ <br>
https://appliedaicourse.com <br>
https://github.com/srvds/Human-Activity-Recognition
https://www.sciencedirect.com/science/article/pii/S2667096821000392

