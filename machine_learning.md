# Machine Learning Notes

# Introduction to Machine Learning with scikit-learn

These are notes from Kevin Markham's course.

## Lesson 01. What is machine learning and how does it work?

- From basic to advanced techniques
- No assumptions about prior knowledge of machine learning
- Minimal experience with Python

What is machine learning?

- Semi-automated extraction of knowledge from data
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

- James, Gareth et al. (2021). An Introduction to Statistical Learning with Applications in R. 2nd ed. New York, NY: Springer. isbn: 978-1-4614-7138-7. url: https://www.statlearning.com/. Read sectiion 2.1.
- [Caltech video](https://www.youtube.com/watch?v=mbyG85GZ0PI&t=2162s). Watch this 15 minute segment.

Quiz

1. If a bank was using machine learning to locate fraudulent transactions, is that an example of supervised or unsupervised learning? Supervised
2. If a website was using machine learning to group its visitors based on browsing history, is that an example of supervised or unsupervised learning? Unsupervised
3. If you are working on a supervised learning problem, are required to have labeled data? Yes

# Master Machine Learning with scikit-learn

These are notes from Kevin Markham's course.
