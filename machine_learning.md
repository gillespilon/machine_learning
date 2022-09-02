# Machine Learning Notes

# Introduction to Machine Learning with scikit-learn

These are notes from Kevin Markham's course.

- From basic to advanced techniques
- No assumptions about prior knowledge of machine learning
- Minimal experience with Python

## Lesson 01. What is machine learning and how does it work?

Objectives

_ What is machine learning?
- What are the two main categories of machine learning?
- What are some examples of machine learning?
- How does machine learning work?

What is machine learning?

- Semiautomated extraction of knowledge from data
- Starts with a question that might be answered using data
- Computer provides the insight
- Requires many smart decisions by a human

There are two main categories of machine learning.

- Supervised learning

    - Making predictions using data
    - There is an outcome we are trying to predict
    - For example, the survival for each passenger on the Titanic

- Unsupervised learning

    - Extracting structure from data
    - There is no right answer

How does machine learning work?

1. Train a machine learning model using labeled data

- Labeled data has been labeled with the outcome
- Machine learning model learns the relationship between the attributes of the data and their outcome

2. Make predictions on new data for which the label is unknown

- Want the model to accurately predict the future rather than the past

The model is learning from past examples (inputs and outputs) and applied what it has learned to future inputs in order to predict future outputs.

- How do I choose which attributes of my data to include in the model?
- How do i choose which model to use?
- How do I optimize this model for best performance?
- How do I ensure that I am building a model that will generalize to unseen data?
- Can I estimate how well my model is likely to perform on unseen data?

Resources

- James, Gareth et al. (2021). An Introduction to Statistical Learning with Applications in R. 2nd ed. New York, NY: Springer. isbn: 978-1-4614-7138-7. url: https://www.statlearning.com/. Read section 2.1.
- [Caltech video](https://www.youtube.com/watch?v=mbyG85GZ0PI&t=2162s). Watch this 15 minute segment.

Quiz

1. If a bank was using machine learning to locate fraudulent transactions, is that an example of supervised or unsupervised learning? Supervised
2. If a website was using machine learning to group its visitors based on browsing history, is that an example of supervised or unsupervised learning? Unsupervised
3. If you are working on a supervised learning problem, are required to have labeled data? Yes

## Lesson 02. Setting up Python for Machine Learning: scikit-learn and Jupyter Notebook

Objectives

- What are the benefits and drawbacks of scikit-learn?
- How do I install scikit-learn?
- How do I use Jupyter Notebook?
- What are some good resources for for learning Python?

What are the benefits and drawbacks of scikit-learn?

- Consistent interface to machine learning models
- Provides many tuning parameters with sensible defaults
- Exceptional documentation
- Rich set of functionality for companion tasks: model selection, model evaluation, and data preparation
- Active community for development and support
- Harder than R to get started with machine learning but much easier to go deeper
- Less emphasis that R on model interpretability

    - scikit-leaer emphasizes machine learning -> predictive accuracy
    - R emphasizes statistical learning -> model interpretability and uncertainty

Further reading

- [Ben Lorica. Six reasons why I recommend scikit-learn](https://www.oreilly.com/content/six-reasons-why-i-recommend-scikit-learn/)
- [scikit-learn authors. API design for machine learning software](https://arxiv.org/pdf/1309.0238v1.pdf)
- [Data School. Should you teach Python or R for data science?](https://www.dataschool.io/python-or-r-for-data-science/)

How do I install scikit-learn?

- Option 1. [scikit-learn installation page](https://scikit-learn.org/stable/install.html)
- Option 2. A Python distribution that includes scikit-learn

How do I use Jupyter Notebook?

- Launch Jupyter Notebook from the command line using `jupyter notebook`
- Kevin provides a brief overview of Jupyter Notebook functionality
- Kevin also provides comments on [nbviewer](https://nbviewer.jupyter.org/) to view previously-published notebooks online as static documents
- I prefer to code Python as .py scripts in order to run them in a production environment

What are some good resources for for learning Python?

- [Codecademy's Python course. browser-based, tons of exercises](https://www.codecademy.com/learn/learn-python)
- [DataQuest. browser-based, teaches Python in the context of data science](https://www.dataquest.io/)
- [Google's Python class. slightly more advanced, includes videos and downloadable exercises (https://developers.google.com/edu/python/with solutions)]()
- [Python for Everybody. beginner-oriented book, includes slides and videos](https://www.py4e.com/)

I have not done any of these. The fourth link sounds the most useful to a begineer.

Quiz

1. In general, does the field of machine learning place a greater emphasis on predictive accuracy or model interpretability and uncertainty? Predictive accuracy
2. Does the Jupyter Notebook support programming languages other than Python? Yes
3. Which mode allows you to navigate a Jupyter Notebook and create cells using the keyboard? Command mode
4. Which keyboard key will always switch you to command mode? Esc

## Lesson 03. Getting started in scikit-learn with the famous iris dataset

Objectives

- What is the famous iris dataset and how does it relate to Machine Learning?
- How do we load the iris dataset into scikit-learn?
- How do we describe a dataset using Machine Learning terminology?
- What are scikit-learn's four key requirements for working with data?

What is the famous iris dataset and how does it relate to Machine Learning?

- 50 samples of 3 different species, 150 samples total
- Measurements: sepal length, sepal width, petal length, petal width
- Measured by Edgar Anderson in 1936
- Sir Ronald Fisher wrote a paper in 1936 about the iris dataset, specifically about how linear discriminant analysis could be used to accurately distinguish the three species using only the sepal and petal measurements
- This is a supervised machine learning problem because we are trying to learn the relationship between the measurements to predict the species.

How do we load the iris dataset into scikit-learn?

- See `iris.py`
- Each row represents one flower
- Each column represent one of the four measurements

How do we describe a dataset using Machine Learning terminology?

- Each row is an observation (sample, example, instance, record)
- Each column is a feature (predictor, attribute, independent variable, input, regressor, covariate)
- The target represents what we will predict: 0 for setosa, 1 for versicolor, 2 for virginica (target, response, outcome, label, dependant variable)
- There are two types of supervised machine learning

    - Classification has a categorical response
    - Regression has an ordered and continuous response

- Personal side note. Cassie Korzygov uses the terms classification and prediction
- The iris flower is a classification problem

What are scikit-learn's four key requirements for working with data?

- Features and response are separate objects
- Features and response should be numeric
- The response object ("y") for a classification problem can be an array of string values, and is no longer limited to numeric values.
- Features and response should be NumPy arrays
- Features and response should have specific shapes (ndarray)
- Convention is for feature data to be stored in an object named X (uppercase because it is a matrix)
- Convention is for response data to be stored in an object names y (lowercase because it is a vector)

Further reading

- [Wikipedia iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set)
- [UCI Machine Learning Repository. Iris dataset](http://archive.ics.uci.edu/ml/datasets/Iris)
- [scikit-learn documentation. Dataset loading utilities](https://scikit-learn.org/stable/datasets.html)
- [Jake VanderPlas. Fast Numerical Computing with NumPy (slides)](https://speakerdeck.com/jakevdp/losing-your-loops-fast-numerical-computing-with-numpy-pycon-2015)
- [Jake VanderPlas. Fast Numerical Computing with NumPy (video)](https://www.youtube.com/watch?v=EEUXKG97YRw)
- [Scott Shell. An Introduction to NumPy (PDF)](https://sites.engineering.ucsb.edu/~shell/che210d/numpy.pdf)

Quiz

1. Let's say a bank was using machine learning to locate fraudulent transactions. What term would be used to describe a single transaction record? Observation or sample
2. In that same example, what term would be used to describe the field that shows the transaction time? Feature or predictor
3. In that same example, what term would be used to describe the field that indicates fraud or not fraud? Response or target
4. If a car sharing service was using machine learning to predict arrival time, is that an example of classification or regression? Regression
5. In scikit-learn, what is the convention to represent a feature matrix? Uppercase X
6. If your feature matrix has a shape of (500, 10), what shape should the response vector be? (500,)

## Lesson 04. Training a Machine Learning Model with scikit-learn

Objectives

- What is the K-nearest neighbors classification model?
- What are the four steps for model training and prediction in scikit-learn?
- How can I apply this pattern to other machine learning models?

What is the K-nearest neighbors classification model?

- Pick a value for K
- Search for the K observations in the training data that are nearest to the measurement of the unknown iris
- Use the most popular response value from the K-nearest neighbors as the predicted response value for the unknown iris

What are the four steps for model training and prediction in scikit-learn?

- See `iris.py` for the K-nearest neighbors code

1. Import the class you plan to use
2. Instantiate the estimator

    - Estimator is scikit-learn's term for model
    - Instantiate means make an instance of
    - Name of the object does not matter
    - Can specify tuning parameters (hyperparameters) during this step
    - All parameters not specified are set to their defaults

3. Fit the model with data (model training)

    - Model is learning the relationship between X and y
    - Occurs in-place

4. Predict the response for a new observation

    - New observations are called out-of-sample data
    - Uses the information it learned during the model training process
    - Returns a NumPy array (ndarray)
    - Can predict for multiple observations at once

How can I apply this pattern to other machine learning models?

- scikit-learn's models have a uniform interface, thus use the same four-step pattern on a different model with relative ease
- See `iris.py` for the logistic regression code

We don't know the true response values. We are often not able to truly measure how well our models perform on out-of-sample data.

Further reading

- [UCI Machine Learning Repository. Iris dataset](http://archive.ics.uci.edu/ml/datasets/Iris)
- [Nearest Neighbors (user guide)](https://scikit-learn.org/stable/modules/neighbors.html)
- [KNeighborsClassifier (class documentation)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [Logistic Regression (user guide)](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [LogisticRegression (class documentation)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Videos from An Introduction to Statistical Learning](https://www.dataschool.io/15-hours-of-expert-machine-learning-videos/)

    - Classification Problems and K-Nearest Neighbors (Chapter 2)
    - Introduction to Classification (Chapter 4)
    - Logistic Regression and Maximum Likelihood (Chapter 4)

Quiz

1. When creating an instance of a model, does it matter what you name the object? No
2. During which step do you specify the hyperparameters for a model? When creating an instance of that class
3. Do you have to specify all hyperparameters for a model or only the ones you want to change? Only the ones you want to change from the default values
4. When fitting a model with data, do you need to assign the results to an object? No
5. Can you make predictions for multiple out-of-sample observations at the same time? Yes

# Master Machine Learning with scikit-learn

These are notes from Kevin Markham's course.
