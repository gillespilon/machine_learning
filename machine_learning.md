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

# Master Machine Learning with scikit-learn

These are notes from Kevin Markham's course.
