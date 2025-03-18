# Sensor Data Fusion of PPG and Acceleration using Artificial Neural Networks for Heart Rate Estimation

The heart rate (HR) is an essential parameter for a person’s health and fitness. For years,
this parameter was measured by electrocardiogram (ECG) as a standard. This required the
use of expensive and cumbersome equipment. The last few years, wearable devices such as
smartwatches or fitness trackers were developed and gained popularity. They offer the possibility
to measure blood volume fluctuations and derive heart rate in a portable and continuous
way by means of photoplethysmography (PPG). 

A PPG signal is measured by a certain hardware configuration, consisting of a LED and
photo detector. Light is shot into the tissue of the subject, where it is reflected or absorbed.
The photo detector measures the amount of reflected light over the time. This function is
inverted and then known as a PPG signal. The signal consists of a static (DC) and pulsatile
(AC) part. The pulsatile or dynamic part represents changes of the blood volume that occurs
due to the cardiac cycle of systolic and diastolic phases, by which the light gets more or less
absorbed by the blood.

Although PPG is a promising technology, challenges have emerged to determine accurate
heart rates due to motion artifacts in certain situations. To overcome these challenges, this
work employs neural networks to determine more accurate heart rates based on the use of
PPG and Acceleration (ACC) data.

The goal of this work is to implement an existing Neural Network (NN) model and evaluate
it, using specific datasets. The model should be able to provide accurate heart rate (HR)
estimations both at rest and during movements. The accuracy of the HR estimation could
improve the effectiveness of wearable devices for measuring heart health.
In this work, the proposed model achieved a resultant Mean Absolute Error (MAE) and
Average of Relative Absolute Error (ARE) of 1.099 bpm and 0.92% for the training data and
3.173 bpm and 2.50% for the testing data using BAMI1 and BAMI2. Additional results using
the BAMI along with the ISPC dataset provided a resultant Mean Absolute Error (MAE) and
Average of Relative Absolute Error (ARE) of 0.945 bpm and 0.75% for the training data and
3.192 bpm and 2.55% for the testing data.
