# EASYML
*Auto Machine Learning System*

## 1. Introduce EasyML

EasyML is a web-based Auto Machine Learning platform designed to facilitate model development without the need for coding expertise. By simply uploading your data and configuring your experiment, including selecting the target, features, and scoring metrics, EasyML will generate over 10 models along with their corresponding evaluation scores.

*For a more detailed understanding of the EasyML system, we encourage you to watch the video below.*

[![img.png](docs/img/img.png)](https://www.youtube.com/watch?v=jRtNJl3y2as&t)

Feel free to reach out to me on LinkedIn (http://www.linkedin.com/in/lucnguyenvn) if you would like to discuss further, it would be a pleasure (honestly).

## 2. How to use EasyML

### 2.1 Upload Data and analysis your data
Once the model has been built, you can proceed to upload your data by clicking on the File Management button. Currently, the system only supports CSV files, but I am actively working on an upgrade to allow customers to upload other file formats such as Excel and text files. This enhancement will be available in the near future to provide a more versatile data upload capability for our users.

![img_1.png](docs/img/img_1.png)

To EDA your data, click Open EDA. I used PandasProfiling to help me automatic this report.

![img_2.png](docs/img/img_2.png)

### 2.2 Create Experiment

Click Experiment, select your data using to build model, choose target, features, train and test split ratio and submit.
![img_3.png](docs/img/img_3.png)

### 3. Experiment Detail and Prediction
After Experiment success, you can check the performance at this page.
Although, you can make a prediction with best model score.
![img_4.png](docs/img/img_4.png)

### 4. Model Evaluations
By click on models in Leaderboard, you can check more detail about model performance.

![img_5.png](docs/img/img_5.png)
