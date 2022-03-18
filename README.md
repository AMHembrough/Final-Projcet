# Presentation

The presentation outlines the project, including the following:

## Selected topic 

Increased urbanization results in increased energy use.  Crude oil has historically been used for energy production, though renewable energy alternatives also exist.  International, federal and state-wide mandates and regulations, such as the Renewable Fuel Standards (RFS) and the Low Carbon Fuel Standards (LCFS) require renewable fuel production volume obligations to be met annually.  This type of govt regulation is one of many factors that could impact crude oil pricing, especially in the next 5-10 years as renewable diesel production, electric vehicles, and other alternative energy sources become more popular here in the United States and abroad.  Expensive and/or volatile crude oil pricing (due to alternative fuel mandates, political turmoil like that in Russia & Ukraine today, etc.) could push consumers to utilize alternative fuels more quickly.  Alternative fuels, in turn, could have a less harmful, negative anthropological effects on our environment.   

The outcome of this project is to forecast crude oil pricing using a time series forecasting machine learning model called ARIMA. 

## Reason why we selected this topic 

We are interested in learning about the relationship between crude oil, alternative fuels, and factors that impact their demand and pricing.  This is ever important, especially now, given Ukraine and Russia turmoil which has resulted in significantly higher and more volatile crude oil pricing.  Will we see a shift towards renewable fuel sources more quickly?  Crude oil price forecasting may shed some light!

## Description of source of data 

Crude oil price which was obtained through Kaggle. 

The csv file contains crude oil daily pricing in USD from 2/3/2011 through 12/31/2019.  There are 2198 rows of data, including the header row.  

## Questions they hope to answer with the data 

We want to learn how to conduct time series analysis with “out-of-sample” forecasting via machine learning, specifically ARIMA.  

Time permitting, I am also interested in learning VAR over the next two weeks which would allow me to forecast several related variables in time.  

## Description of the data exploration phase of the project 

Given that I am interested in learning ARIMA, I will be exploring a single variable.  

## Description of the analysis phase of the project 

We used two metrics to assess the quality of the ARIMA model, specifically MSE and SMAPE.  

Mean squared error measures the difference between the predicted values and the actual values.  The lesser the MSE, the closer the fit.  In our model, the MSE is 6.58e-06.  

SMAPE was also calculated to determine model accuracy.  The Symmetric Mean Absolute Percentage Error, or SMAPE, is a measurement based on percentage errors.  Like MSE, the lower the value of SMAPE, the higher the model accuracy.  Because SMAPE is percentage based, it’s scale-dependent and can be used compare across datasets or models.  In our model, the SMAPE is 2.77%. 

![Model Accuracy]( _______________________________)

# GitHub

## All code necessary to perform exploratory analysis 

We have utilized 3 jupyter notebooks during the exploratory portion of the class.  

The first notebook used python and pandas to data exploration on our initial urbanization dataset.  This was included in our first deliverable, and showcases how to import data into a dataframe, clean the data by dropping unnecessary columns, join dataframes, transpose dataframes, edit the index, edit the header row, do basic plotting to understand relationships between variables, and so much more.  

A second jupyter notebook was used when building the machine learning portion of the project, specifically regression-enhanced random forest.  Because we chose to not use this model, this jupyter notebook was not added to the main branch of the github repository.  

A third jupyter notebook was used to create our ARIMA model which is the model we will be submitting.  This code for this model has been uploaded to the main branch of the github repository.  

## Some code necessary to complete the machine learning portion of the project 

Code can be found in Jupyter notebook.  

##  Description of the communication protocols 

We did not meet outside of class during the weeks of 7-Mar.  There are plans for the group to meet this Saturday, 19-March.  

There was some communication on via Slack, both within the channel and DM’s within group.  

## Outline of the project 
-	Find dataset (source : Kaggle)
-	Clean dataset using python, pandas
-	Create database using postgres
-	Build model for “in-sample” forecasting
-	Evaluate model accuracy
-	Run model for “out-of sample” forecasting
-	Build dashboard using Tableau

# Machine Learning Model

All code in the main branch is production ready.

## Description of preliminary data preprocessing 

We practiced a lot of data preprocessing in the first two weeks of the final project.  While we will not use the dataframes we created during this process, it was a great way to refine and showcase some of the data cleaning processes we learned during our 24-week bootcamp.  Specifically, we did the following data processing in our initial jupyter notebook (which was submitted with the first deliverable).
-	Import data into a dataframe
-	Cleaned the dataframe by dropping unnecessary columns
-	Join dataframes together
-	Transpose dataframes
-	Edit the index
-	Edit the header row
-	Plot data to understand relationship between variables
-	….and so much more.  

The crude oil price dataset was relatively clean and required little preprocessing.  I experimented with ARIMA using other datasets that did not provide daily values, but instead offered weekly or monthly values.  To make these datasets align well in a pandas dataframe, I did an “interpolation” using pandas which allowed me to forward and backward fill the missing data points.   Below is a screenshot of that code. 

![Extrapolation]( _______________________________)

## Description of preliminary feature engineering and preliminary feature selection, including their decision-making process 

ARIMA utilizes a single variable for time series forecasting.  The variable that we used was crude oil price. 

## Description of how data was split into training and testing sets 

We split our data into training and testing sets by 80 and 20 percent, respectively.  This results in 1743 training samples and 436 testing samples.  We experimented with other splits, notably 90-10 but found the model to be more accurate when using a n 80-20 split.   

![Training & Testing Sets]( _______________________________)

## Explanation of model choice, including limitations and benefits

Originally, we had selected several databases from World Bank and EIA which we believed would be useful in projecting China and India energy demand over time.  Originally, we thought to explore the data using linear regression or multivariate regression, but decided to dive deeper.  

We considered the machine learning techniques we had learned though the course, specifically supervised machine learning, unsupervised machine learning and neural networks.  Given that we had known independent variables that we wanted to leverage to project our dependent variable (= energy use “out-of-sample”), we narrowed our options to supervised machine learning.

We next identified that we were looking for a regression model versus a classification model.  We choose random forest as our machine learning model.  Random forest modeling uses an ensemble of decision trees, each of which is created from a unique sample of rows that makes its own predictions.  The predictions from each decision tree are then averaged, and the random forest prediction is then the average of the individual predictions made by the decision trees.  Individually, the decision trees are weak learners, but collectively they make a robust model known as random forest! 

Through literature review and additional review of the course materials, we learned of the “extrapolation problem” which exists with random forests when doing regression analysis.  The “extrapolation problem” is when the model fails to predict data outside of the scope of the model, meaning predicted values cannot be outside of the training set values of the target.  For this reason, we identified the next model for our needs which was regression-enhanced random forest, or RERF.  

The RERF model we created to project energy demands using independent variables related to urbanization (i.e. urban population, GDP, etc) had a relatively decent performance based on statistical values we calculated, notable R2 and root mean square error (RMSE).  Despite identifying a model with good performance, we realized that RERF wasn’t the best model for time series forecasting, especially given the datasets we initially selected which had way too few rows of data.  While the model was great for “in-sample” forecasting, it failed to accomplish our goal of “out-of-sample” forecasting. 

Through discussions with several TAs and our course instructor, we learned of a more appropriate model for time series forecasting called ARIMA.  ARIMA is an acronym that stands for AutoRegressive Integrated Moving Average.  This type of modeling utilizes a relationship between a single observation in time and X number of lagged observations. This is an ideal model for “out-of-sample” forecasting. 

# Database

Team members submit the code for their machine learning model, as well as the following:

## Database stores static data for use during the project 

Postgres SQL, or Postgres, is the relational database system that we used for our final project.  In conjunction with Postgres, we also used pgAdmin to write and execute queries, and to view our results.  

## Database interfaces with the project in some format (e.g., scraping updates the database, or database connects to the model)

## Includes at least two tables (or collections, if using MongoDB) 

We imported two datasets into Postgres SQL using pdAdmin, one which stores daily pricing for crude oil, and a second which stores daily pricing for renewable fuel D-codes D3, D4, D5, and D6.  

![Postgres Crude Table]( _______________________________)

![Postgres RIN Price Table]( _______________________________)

## Includes at least one join using the database language (not including any joins in Pandas) 

We joined our crude oil pricing data table with our RIN D-type pricing table.  Our output was a table showing crude pricing, alongside prices for D3, D4, D5 and D6 RIN prices.  

![Postgres Join]( _______________________________)

## Includes at least one connection string (using SQLAlchemy or PyMongo) Note: If you use a SQL database, you must provide your ERD with relationships.

![Entity Relationship Diagram]( _______________________________)

# Dashboard

A blueprint for the dashboard is created and includes all of the following:

## Storyboard on Google Slide(s) 

These slides will need updated since we have changed the scope of our project to accommodate a model that is better suited for time series forecasting, specifically ARIMA modeling.  

https://docs.google.com/presentation/d/1hcUUxudJCIKIiH2meyf5kHqM0MWkAN7eNCEKxKkjidY/edit?usp=sharing 

## Description of the tool(s) that will be used to create final dashboard 

We plan to utilize Tableau to create our final Dashboard. 

## Description of interactive element(s)

We plan to make visuals that you can interact with inside Tableau (ie tooltips, etc).  This will be a nice feature to offer to our end users.  