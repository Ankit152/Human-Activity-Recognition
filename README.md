# Human Activity Recognition 🎭

`A Machine Learning approach to predict the activities of person.`

The Human Activity Recognition database was built from the recordings of 30 study participants performing activities of daily living (ADL) while carrying a waist-mounted smartphone with embedded inertial sensors. The objective is to classify activities into one of the six activities performed.

## Description of experiment 📋

The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities `WALKING`, `WALKINGUPSTAIRS`, `WALKINGDOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING` wearing a smartphone `Samsung Galaxy S II` on the waist. Using its embedded accelerometer and gyroscope, 3-axial linear acceleration and 3-axial angular velocity was captured at a constant rate of 50Hz. The experiments have been video-recorded to label the data manually. The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data.

The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. From each window, a vector of features was obtained by calculating variables from the time and frequency domain.

## Attribute information ℹ️

*For each record in the dataset the following is provided:*

* Triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration.
* Triaxial Angular velocity from the gyroscope.
* A 561-feature vector with time and frequency domain variables.
* Its activity label.
* An identifier of the subject who carried out the experiment.

## Exploratory Data Analysis 🗠

### Countplot of Activities 📊

`Below is the countplot of the activities performed by all the subjects.`
<p align=center>
  <img src="https://github.com/Ankit152/Human-Activity-Recognition/blob/main/img/countplot.jpg">
</p>

*From the above plot we can conclude that the datapoints are somewhat balanced.*

### Countplot of Subjects 📊
`Below is the countplot of the activities performed by all the subjects grouped by subjects.`
<p align=center>
  <img src="https://github.com/Ankit152/Human-Activity-Recognition/blob/main/img/subject.jpg">
</p>

*From the above plot we can conclude that a particular subject performs any of the activities more as compared to the other activities.*
