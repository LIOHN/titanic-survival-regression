# titanic-survival-regression

## What's inside?
- a simple, modular, reproducible and portable **scikit-learn** Logistic Regression (ML) model (pipeline) that predicts whether a passenger would survive given ticket and passenger data. 
- data analysis and feature engineering steps
- testing with **pytest**, and **tox** to simplify the process
- packaging code (*python setup.py sdist bdist_wheel*)
- model serving code via **REST API (Flask app)**
- code for deployment to Heroku without containers, and deployment to Heroku with app containerised with Docker. 
- a **CICD (CircleCI)** pipeline with the above defined as jobs 

The focus of this project was on implementing newly acquired knowledge on CICD (CircleCI) and containerisation with Docker, from Udemy and LinkedIn Learning courses I have been learning from since graduating.

From deploying machine learning model course assignment:
"Perhaps one of the most infamous shipwrecks in history, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 people on board. Interestingly, by analysing the probability of survival based on few attributes like gender, age, and social status, we can make very accurate predictions on which passengers would survive. Some groups of people were more likely to survive than others, such as women, children, and the upper-class. Therefore, we can learn about the society priorities and privileges at the time."
