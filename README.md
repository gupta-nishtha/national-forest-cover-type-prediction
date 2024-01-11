# Forest Cover Type Prediction in Roosevelt National Forest of Northern Colorado

This is a [Kaggle Competition](https://www.kaggle.com/c/forest-cover-type-prediction/overview)

This repository contains some notebooks to analyze the problem and create a machine learning model to predict the cover type in Roosevelt National Forest.

## Problem Description

In this competition you are asked to predict the forest cover type (the predominant kind of tree cover) from strictly cartographic variables (as opposed to remotely sensed data). The actual forest cover type for a given 30 x 30 meter cell was determined from US Forest Service (USFS) Region 2 Resource Information System data. Independent variables were then derived from data obtained from the US Geological Survey and USFS. The data is in raw form (not scaled) and contains binary columns of data for qualitative independent variables such as wilderness areas and soil type.

## Approach

In it, there are several data points, such as the distance to fire points, slope, elevation, and even soil types. The challenge lies in using these different variables to predict the potential cover type. To tackle this challenge, I took an approach that breaks the problem statement into three phases:

Phase 1:
In this phase of the project, you will explore:
- The properties of individual predictors
- The relationships amongst collections of predictors
- The connection between predictors and the target

Phase 2:
In this phase, two separate models (e.g., logistic regression and classification tree, or linear regression and regression tree), evaluate their individual performance, and discuss the pros and cons of each relative to the other. The big ideas here are to see how these methods work "in the wild" that is, on real data sets and to start thinking about why we might choose one model over another. 

1. Model Construction and Interpretation 
- Construct two models of different types
- Control for overfitting in each model
- Clearly explain each model, i.e., the most important coefficients from logistic/linear or the most important splits for trees

2. Model Evaluation and Comparison
- Discuss error rate and any other relevant measurements of performance for each model 
- Discuss the pros and cons of each model with respect to the other
- Ultimately choose the "better'' model, where "better'' has been informed and defined by your work on previous points

Phase 3:
Now that some basic data cleaning and make two individual models for your data set, it’s time to double down and see what you can come up with when you really bring to bear all the tools we’ve learned in the course. This can be broken up into three parts:

1. Data Cleaning
Now it’s time to actually clean all the columns in your data set.

- Document the strategies you used to clean each column or group of columns
- Discuss the strengths and weaknesses of these strategies
- Compare model performance before and after cleaning
- As a good rule of thumb, you should not be deleting rows with missing values; in the real world, we need to make predictions even under incomplete information.

2. Model Building
- Build one more model (in addition to the two you already have); this can either be a model we covered in class or a new model that you research as a team.
- Analyze performance of this new model

3. Ensemble Learning
No single model will be best-in-class, and therefore we need to practice implementing ensemble methods to improve individual model performance. 

4. Construct a stacking model using the three base models you constructed
- Compare performance of stacking to the best performance of the constituent models
- Compare and contrast performance using stacking, paying special attention to the added value (if any) that stacking provides

## Conclusion
To summarize, I implemented data cleaning strategies such as checking for null values and removing unnecessary variables/columns. Despite implementing these strategies, the model performance did not change significantly. I also chose Naive Bayes as our third machine learning model. This model only accepts categorical input, so I binned some columns, such as elevation, to run into the model. Despite lowering the error rate from a benchmark rate of 64% to 36%, the Naive Bayes model was the worst performer in comparison to our other models, KNN and Classification Tree. To blend all models together to hopefully achieve optimal results, I stacked the models. I implemented a random forests model as our manager model and the rest as helper models. Running it performed better than our Naive Bayes’ and KNN models at 24% and just 1% higher than the classification tree model.
