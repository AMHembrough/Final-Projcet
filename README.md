# Presentation

### Selected topic 

Increased urbanization results in increased energy use.  Crude oil has historically been used for energy production, though renewable energy alternatives also exist.  International, federal, and state-wide mandates and regulations, such as the Renewable Fuel Standards (RFS) and the Low Carbon Fuel Standards (LCFS) require renewable fuel production volume obligations to be met annually.  This type of govt regulation is one of many factors that could impact crude oil pricing, especially in the next 5-10 years as renewable diesel production, electric vehicles, and other alternative energy sources become more popular here in the United States and abroad.  Expensive and/or volatile crude oil pricing (due to alternative fuel mandates, political turmoil like that in Russia & Ukraine today, etc.) could push consumers to utilize alternative fuels more quickly.  Alternative fuels, in turn, could have a less harmful, negative anthropological effects on our environment.   

The outcome of this project is to forecast crude oil pricing using a time series forecasting machine learning model called ARIMA. 

### Reason why we selected this topic 

We are interested in learning about the relationship between crude oil, alternative fuels, and factors that impact their demand and pricing.  This is ever important, especially now, given Ukraine and Russia turmoil which has resulted in significantly higher and more volatile crude oil pricing.  We anticipate a shift towards renewable fuel sources more quickly, which could have fewer negative anthropological effects on the environment than the consumption of crude oil.

### Description of source data 

Crude oil price which was obtained through Kaggle. The csv file contains crude oil daily pricing in USD from 2/3/2011 through 12/31/2019.  There are 2198 rows of data, including the header row.  

We also explored other variables with our ARIMA model such as sustainability index and pricing of various RIN d-codes, specifically D3, D4, D5 and D6.  The sustainability data was derived from Kaggle, while the RIN pricing data were derived from the Environmental Protection Agency, or EPA.  

### Questions we hope to answer with the data 

We want to learn how to conduct time series analysis with “out-of-sample” forecasting via supervised machine learning, specifically ARIMA.  We will project crude oil pricing.

Time permitting, we are also interested in completing a VAR analysis.  

### Description of the data exploration phase of the project 

Originally, we had selected several databases from World Bank and EIA which we believed would be useful in projecting China and India energy demand over time.  Through the data exploration phase of the project, we learned that our datasets contained too few rows.  

Within the last week, we began looking for a new dataset that would satisfying our database row needs.  While we could not find the same metrics as our original dataset, we did find a related dataset containing daily crude oil pricing.  

### Description of the analysis phase of the project 

The analysis phase of the project helped us understand the strengths and weakness of several machine learning models.  During this process, we explored a number of supervised machine learning models including regression analysis, random forests, regression-enhanced random forests (RERF), and autoregressive integrated moving average (ARIMA).  We hope to alsoe experiment with vector autoregression (VAR).  

The model we are presenting as our final project is the ARIMA model.  We used two metrics to assess the quality of the ARIMA model, specifically MSE and SMAPE.  
-	Mean squared error measures the difference between the predicted values and the actual values. The lesser the MSE, the closer the fit. In our model, the MSE is 1.77.
-	SMAPE was also calculated to determine model accuracy. The Symmetric Mean Absolute Percentage Error, or SMAPE, is a measurement based on percentage errors. Like MSE, the lower the value of SMAPE, the higher the model accuracy. Because SMAPE is percentage based, it’s scale-dependent and can be used compare across datasets or models. In our model, the SMAPE is 1.50%. It will be fun to compare SMAPE between this ARIMA model, and a later VAR model.

![Model Accuracy](https://github.com/AMHembrough/Final-Projcet/blob/main/Resources/Accuracy.PNG)

# Machine Learning Model

### Description of preliminary data preprocessing 

We practiced a lot of data preprocessing in the first two weeks of the final project using the EPA and World bank urbanization data.  While our end product will not utilize the dataframes we collected during this stage of data exploration, it was a great way to refine and showcase some of the data cleaning processes we learned during our 24-week bootcamp.  

Specifically, we did the following data processing in our initial Jupyter notebook (which was submitted with the first deliverable).
-	Import data into a dataframe
-	Clean the dataframe by dropping unnecessary columns
-	Join dataframes together
-	Transpose dataframes
-	Edit the index
-	Edit the header row
-	Plot data to understand relationship between variables
-	….and so much more

The crude oil price dataset was relatively clean and required little preprocessing.  However, in addition to using the ARIMA model to project crude oil pricing, we also ran an ARIMA model on additional data such as RIN pricing and sustainability index.  These data, unlike the crude oil pricing data, were provided weekly or monthly.  To make these datasets align well in a single pandas dataframe, we did an “interpolation” using pandas which allowed us to forward and backward fill the missing data points.   Below is a screenshot of that code. 

![Extrapolation](https://github.com/AMHembrough/Final-Projcet/blob/main/Resources/Extrapolation.PNG)

### Description of preliminary feature engineering and preliminary feature selection, including our decision-making process 

The ARIMA model uses the dependency between an observation and the residual error from the moving average.  
The variable we used was crude oil pricing.  

### Description of how data was split into training and testing sets 

We split our data into training and testing sets by 90 and 10 percent, respectively.  This results in 1961 training samples and 218 testing samples.  We experimented with other splits, notably 80-20 but used 90-10 in the end.  

![Training & Testing Sets](https://github.com/AMHembrough/Final-Projcet/blob/main/Resources/Training.PNG)

### Explanation of model choice, including limitations and benefits

Originally, we had selected several databases from World Bank and EIA which we believed would be useful in projecting China and India energy demand over time.  Originally, we thought to explore the data using linear regression or multivariate regression, but decided to dive deeper.  

We considered the machine learning techniques we had learned though the course, specifically supervised machine learning, unsupervised machine learning and neural networks.  Given that we had known independent variables that we wanted to leverage to project our dependent variable (= energy use “out-of-sample”), we narrowed our options to supervised machine learning.

We next identified that we were looking for a regression model versus a classification model.  We choose random forest as our machine learning model.  Random forest modeling uses an ensemble of decision trees, each of which is created from a unique sample of rows that makes its own predictions.  The predictions from each decision tree are then averaged, and the random forest prediction is then the average of the individual predictions made by the decision trees.  Individually, the decision trees are weak learners, but collectively they make a robust model known as random forest! 

Through literature review and additional review of the course materials, we learned of the “extrapolation problem” which exists with random forests when doing regression analysis.  The “extrapolation problem” is when the model fails to predict data outside of the scope of the model, meaning predicted values cannot be outside of the training set values of the target.  For this reason, we then tried a regression-enhanced random forest, or RERF.  

The RERF model we created to project energy demands using independent variables related to urbanization (i.e. urban population, GDP, etc) had a relatively decent performance based on statistical values we calculated, notable R2 and root mean square error (RMSE).  Despite identifying a model with good performance, we realized that RERF wasn’t the best model for time series forecasting, especially given the datasets we initially selected which had way too few rows of data.  While the model was great for “in-sample” forecasting, it failed to accomplish our goal of “out-of-sample” forecasting. 

Through discussions with several TAs and our course instructor, we learned of a more appropriate model for time series forecasting called ARIMA.  This is an ideal model for “out-of-sample” forecasting.  ARIMA is an acronym that stands for Autoregressive Integrated Moving Average.  Autoregression means that the model uses a dependent relationship between an observation and X number of lagged observations.  The I in ARIMA stands for integrated, meaning that the model utilizes difference between a single observation and the previous observation.  The RA in ARIMA stands for moving average, meaning that the model uses the dependency between an observation and the residual error from the moving average. 

There are three parameters of ARIMA, specifically (p,d,q).  P refers to the lag order, or the number of lag observations included in the model. D refers to the degree of differencing, or the number of times the raw observations are differenced.   Q refers to the order of the moving average, or the size of the moving average window.  Each of these parameters are substituted with integers to indicate the type of ARMMA model being used.  If a zero is substituted for a parameter, that element will not be used in the ARIMA model.  

When we fit the ARIMA model to our crude oil dataset, we learned that our data was best fit to ARIMA (0,1,0) which had the smallest AIC value compared to other fits.  
![ARIMA fit]( https://github.com/AMHembrough/Final-Projcet/blob/main/Resources/ARIMA%20fit.PNG)

One limitation of ARIMA modeling, which we’ve mentioned already, is that it only utilizes a single variable.  For this reason, it’s worth noting that the VAR model doesn’t have this limitation, which is why we are eager to try this model out, too! 

# GitHub

### All code necessary to perform exploratory analysis 

We have utilized 3 Jupyter notebooks during the exploratory portion of the class.  
The first notebook utilized python and pandas for data exploration of our initial urbanization dataset.  This was included in our first deliverable, and showcases how to import data into a dataframe, clean the data by dropping unnecessary columns, join dataframes, transpose dataframes, edit the index, edit the header row, do basic plotting to understand relationships between variables, and more.  

A second Jupyter notebook was used when building the machine learning portion of the project, specifically regression-enhanced random forest.  Because we chose to not use this model, this Jupyter notebook was not added to the main branch of the GitHub repository.  

A third Jupyter notebook was used to create our ARIMA model which is the model we will be submitting for our final deliverable.  This code for this model has been uploaded to the main branch of the GitHub repository.  

### Some code necessary to complete the machine learning portion of the project 

Please refer to our Jupyter notebook which has been uploaded to Github.

### Description of the communication protocols 

We did not meet outside of class during the weeks of 7-Mar.  There are plans for the group to meet this Saturday, 19-March.  

There was some communication on via Slack, both within the channel and DM’s within group.  

### Outline of the project 
-	Find dataset (source : Kaggle)
-	Clean dataset using python, pandas
-	Create database using postgres
-	Build model for “in-sample” forecasting
-	Evaluate model accuracy
-	Run model for “out-of sample” forecasting
-	Build dashboard using Tableau

# Database

### Database stores static data for use during the project 

Postgres SQL, or Postgres, is the relational database system that we used for our final project.  In conjunction with Postgres, we also used pgAdmin to write and execute queries, and to view our results.  

### Includes at least two tables (or collections, if using MongoDB) 

We created two tables in Postgres SQL using pdAdmin.   The first table stores daily pricing for crude oil, and the seconds table stores daily pricing for renewable fuel D-codes D3, D4, D5, and D6.  

![Postgres Crude Table](https://github.com/AMHembrough/Final-Projcet/blob/main/Resources/Postgres%20table.PNG)

![Postgres RIN Price Table](https://github.com/AMHembrough/Final-Projcet/blob/main/Resources/Postgres%20table2.PNG)

### Includes at least one join using the database language (not including any joins in Pandas) 

We joined our crude oil pricing data table with our RIN D-type pricing table.  Our output was a table showing crude pricing, alongside prices for D3, D4, D5 and D6 RIN prices.  

![Postgres Join](https://github.com/AMHembrough/Final-Projcet/blob/main/Resources/Postgres%20join.PNG)

### Includes at least one connection string (using SQLAlchemy or PyMongo) Note: If you use a SQL database, you must provide your ERD with relationships.

Please see the entity relationship diagram for our SQL database below.  

![Entity Relationship Diagram](https://github.com/AMHembrough/Final-Projcet/blob/main/Resources/Entity%20relationship%20diagram.PNG)

# Dashboard

### Storyboard on Google Slide(s) 

Here is a link to our storyboard on Google Slides: 

https://docs.google.com/presentation/d/1hcUUxudJCIKIiH2meyf5kHqM0MWkAN7eNCEKxKkjidY/edit?usp=sharing 

### Description of the tool(s) that will be used to create final dashboard 

We are utilizing Tableau to create visuals for our final project.  Here are examples of some of the visuals we plan to deliver.  

![Tableau RIN](https://github.com/AMHembrough/Final-Projcet/blob/main/Resources/Tableau%20RIN.PNG)

![Tableau Crude](https://github.com/AMHembrough/Final-Projcet/blob/main/Resources/Tableau%20crude.PNG)

### Description of interactive element(s)

We plan to make visuals that you can interact with inside Tableau (ie tooltips, etc).  This will be a nice feature to offer to our end users.  
