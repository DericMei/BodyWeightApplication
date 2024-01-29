# **BodyWeightMagicApplication**
The **BEST** project I have ever done so far! By utilizing machine learning tools, exploratory analysis, and dash, I built a web based application deployed by Heroku and running a PostgreSQL dabase in the background, to automate my entire body weight analysis for me! Check it out: 

[Eric Mei's Body Weight Magic Tool](https://bodyweightapp-f5c3d823ac56.herokuapp.com/). 

(Notice this web based application is designed only for desktop browsers, layout will not look well in smartphones)

## Introduction/Background
I was fat for as long as I could remember before 2014, my body weight was constantly over 230lbs and reached a peak over 250lbs in the year of 2014. Thats when I finally decided to lose weight.

I did lots of researches online and tried out so many different stuff myself and finally successfully lost 100 lbs in a year and began my fitness journey ever since.   
I got my personal trainer certificate from American Council on Exercise (ACE) and I am a certifited nutritional specialist from ACE as well. I learned how to be fit and stay fit. Thats when I started to monitor my body weight constantly. I lost a year or two of body weight data initially because I was using another app.   

After doing the last project of building a LSTM model using my body weight data, which is just for exploratory purposes, now I believe I can use my newly learned skills to build a web application of my own to do the job where I would do manually automatically for me! 

If you are interested, check out the LSTM project here: [LSTM Project on Body Weight](https://zijiemei.com/2023/12/29/my-body-weight-analysis/).

The reason I got the idea of this project is that I have just started my yearly routine cutting phase. This is because every year I go on a cycle of gaining a few pounds during the winter and holiday seasons and start a new cutting phase the next year around the end of winter or start of spring to be back in six-pack shape. During this cutting phase, I count calories and monitor my body weight very closely to understand my caloric deficit state. And because there are no apps out there that can do what I needed to do, which is a weekly average body weight analysis. I have to manually calculate my weekly average every single week to understand my current cutting situations. The reason weekly average is so important in weight management is daily body weight fluctuates too much for analysis, there are numerous reasons that can cause dramatic changes in body weight on a daily basis, such as meal timing, sleep, stress level, time of record, etc. These can cause unecessary attention to unimportant trends. Weekly averages on the other hand, make so much more sense. It is so much easier to see the overall trends with weekly averages and monitor how someone is doing when it comes to body weight management. Thats where I got the idea of automating the entire process and using even more machine learning tools to analyze underlying trends by building a web application.

(1kg of pure body fat is roughly 7700 Kcal (Calories), so on a weekly average base, during the cutting phase, I can easily know what is my daily caloric deficit, which I can further determine if I need to adjust my daily diet.)

Below is a picture of my personal body transformation of before and after. Left one is me back in 2013, right one is me in 2022, and I look pretty much the same now only a bit more muscular :). Check it out!!
<img src="https://raw.githubusercontent.com/DericMei/First-Trial/main/weight.JPG" width="600" height="400"/>

## Objectives
The main objective for this project is to build a comprehensive body weight analysis tool for myself to use which can automate my daily manual routine. In order to do that, I need the application to have the following functions:

- Record my body weight daily and upload it to a cloud based SQL database where it stores all my body weight data.
- Because I need to record the weight on the page, I cant risk people messing up my database, so I have to add a password function so that only by entering the correct password, one can record body weight and perform other operations such as train the model.
- Connect to the cloud based SQL database automatically and retrieve my bodyweight data to do analysis.
- Do all sorts of up to date analysis and calculations for me automatically.
- Present the results of the analysis in a visually pleasing way so that I can easily know my current state.
- As a data scientist, I want to impliment machine learning tools to do some seasonality analysis and discover underlying trends for me, as well as predictions just for fun. (I care about seasonality analysis, predictions on the other hand is not that useful for me in body weight analysis, they are just for fun)
- I need a way to retrain my model directly on the web application so I do not need to go back go source codes to retrain models when new body weight data accumulates.

## Application Walkthrough and Demo
**There are 4 main components of the application:**
1. **Basic Information**
    - This part contains basic informations such as weight tracking informations, weekly summaries, date, and so on.
1. **Password Input and Weight Recording**
    - For the public, you can only see the password entry box.
    - When I enter the correct password, I can further record my daily weight or update it.
    - It also unlocks the model training button down in the machine learning session.
1. **Weight Data Plots**
    - This part is initially set to collapse, you need to click on Collapse to see it.
    - It contains daily vs weekly average body weight across different time spans.
    - It also contains all of my weight data displayed on different metrics, such as daily, weekly, monthly and seasonaly averages.
1. **Machine Learning Insights**
    - This part contains all the informations about the facebook prophet model.
    - It has seasonality trend plots for different time scales.
    - It also predicts my body weight for today, just for fun, it is not accurate at all and I don't really need it.
    - It has a training button which can only be seen on the correct password entry which can retrain the prophet model with newest data available.
    - It also has a section with some personal links.

**Here is a GIF Demo of the webpage:**

<img src="https://raw.githubusercontent.com/DericMei/BodyWeightApplication/main/demo.gif">

## Project Summary
Eric Mei's "Body Weight Magic Tool" is an innovative web-based application developed for automated body weight analysis. It's a comprehensive tool, deployed on Heroku, with a PostgreSQL database backend, designed for desktop browsers. This project is a testament to my journey in personal fitness and skills in data science and web development. It serves as a personalized, automated body weight analysis and tracking tool, showcasing practical application of machine learning and data visualization in personal health management.

## Methodology (Project building pipeline)
To do the project, except for the main .py file where the web application is built on, I need to have a separate jupyter notebook file on the side to test out codes as well as prepare my sql database and so on. The overall flow of the project goes like this:
1. **Data Cleaning**
    - Exported my body weight data from my phone
    - Then used excel at first then moved on to pandas on the jupyter notebook file.
2. **Create a Cloud Based SQL Database** 
    - This step is **critical** for the success of the project since it enables me to store all cleaned body weight data, and the web application can communicate with it to retrieve data as well as updating new data.
    - I tried to use SQLite first since that was what I am most familiar with, unfortunately, it was a dead end since I cant put the database on cloud for Heroku deployment.
    - So I figured out to use Heroku add-on and created a PostgreSQL cloud based database, which I am paying for every month. 
3. **Building Dash Application**
    - Created a virtual environment to isolate the project and only install nessasary packages.
    - Application layout design
        - Thanks to Mimo, where I learned HTML language, I was able to design the layout of the application easily.
        - I also learned a little about CSS to adjust style on my web page.
    - Callback functions
        - For Dash, I need to create so many callbacks functions to get the desired results.
    - Feature design
        - Weight tracking information
        - Weekly summary (I need to know the status I am in right now for the week)
        - Password function
        - Weight record function
        - Weight Data Plots (daily, weekly average, all data)
        - Machine Learning insights (Seasonality trends, model training status)
            - I also need a function to retrain the model with the push of a button here so that I can train the model with new data without needing to adjust source codes.
4. **Modeling and Adjusting Database Along the side on Jupyter Notebook**
    - Tried linear regression model first
        - I did feature engineer first to create some new time related features such as seasons, day_of_week, day_of_month, etc.
        - Fitted the linear regression model.
        - Results are not significant which is why I did not put it on my web application.
    - Facebook Prophet Model
        - I did some research to figure out what is a good time series model that can generate good seasonality trends and insights, then figured out the Facebook Prophet model is a good choice.
        - Did Grid Search Cross Validation trying to fit the best prophet model.
        - It is too computationally intensive, to run grid search cross validation since it can easily take more than a couple hours to run, and I don't need it to predict, I only need seasonality trends to understand underlying trends.
        - So I went ahead and train the model after basic tuning, learned the process and moved codes to my application.
    - Saving model to cloud database
        - Since I need the model to run on my web based application, I need to save it to a cloud storage place for my application to run. However, I only have a cloud based PostgreSQL database.
        - Although it is not ideal, since the trained model is pretty small in size, I was able to serialize it and save it into a column of a newly created table in my cloud based PostgreSQL dabase.
        - Now with this being done, I can load the model from my main .py file from the database.
    - Database fixes and adjustments
        - While building the application, I need to keep testing how it works, and some of the things I did always messes up the database. So I have to keep fixing the part I messed up and checking to make sure the database is running at its best.
5. **Deployment**
    - Create the requirements.txt, runtime.txt, Procfile, files for heorku deployment.
    - Pushed all finalized changes to GitHub.
    - Connected Heroku application to GitHub.
    - Set up environmental variables such as password with Heroku.
    - Adjust timezone that the app is running in to my timezone on Heroku, since this application is very time sensitive. Queries all rely on relitive time to run.
    - Finally, deployed the web based application using Heroku!

## Key Insights/Take-Aways
For this specific project, I had to say I have all the domain knowledge I need! The data are all my personal weight records, I know exactly what happens at what time points and how to explain different seasonality trends! Here are some key take-aways for the project:
- For the weekly summary part, I can easily see my weekly average comparing to last week's average. I don't even need to think because on the right of it, it shows me my caloric status for the week and my estimate daily caloric value.

<img src="https://raw.githubusercontent.com/DericMei/First-Trial/main/weekly%20summary.png" width="600" height="140"/>

- For weight data plots, I can easily compare my daily weight and weekly average on the same time span to see differences. Notice that weekly averages are so important when it comes to weight management since on a daily scale, weight fluctuates too much to actually see any clear trends, especially in a short period of time! You can easily see what I am talking about by taking a look at the picture below. I also have the option to see my overall data on different average scales to know my current status.

<img src="https://raw.githubusercontent.com/DericMei/First-Trial/main/weight%20data%20plots.png" width="600" height="250"/>

- For the facebook prophet model, I can easily see my weekly and yearly seasonality trends. If you take a look at the plot below, on a weekly basis over the past years, I normally will go on some wild cheat meals during the weekends and goes back to clean diet on week days, that is why you can see that my weight normally are the highest for saturdays and mondays, and lowest on wednsdays. For yearly trends, my body weight normally will peak at the begining of Febuwary because it is the holiday seasons ending with the Chinese new year!

<img src="https://raw.githubusercontent.com/DericMei/First-Trial/main/weekly%20season.png" width="350" height="230"/><img src="https://raw.githubusercontent.com/DericMei/First-Trial/main/yearly%20season.png" width="350" height="230"/>

- I also managed to create a button to train the model with the press of it with the newest data!

## Learnings/Obstacles
There are so many stuff I did not know, I struggled so much going through with this project since I had to learn so many new stuff along the way to make everything work. Here are some major learnings:
1. **Cloud based SQL database**
    - I did not have prior knowledge on how to setup a cloud based SQL database, I only knew how to setup local MySQL and SQLite databases.
    - I was able to learn how to set up a PostgreSQL cloud database with Heroku add-on.
1. **Environmental Variables**
    - I had no idea what environmental variables were before, I initially just hardcoded my database and password credentials in my source codes like an idiot.
    - After receiving the automated Heroku credential leak message, I quickly learned that I need to setup environmental variables to store sensitive informations.
    - I learned how to set up both local environmental variables for development and also with Heroku on the cloud.
1. **Dash**
    - I had basic knowledge of dash before, but were not fluent with it.
    - For this project, I had to learn so much more about callback functions, how to hide and show parts of the page based on inputs, how to collapse or have dropdown options, etc.
    - Now I am pretty confident with Dash already that I am going to build more Dash apps in the future.
1. **HTML**
    - Thanks to Mimo, I learned some basic HTML coding but those are far from enough to design a web application
    - I had to learn HTML languages along the way to make my application layout look better.
1. **CSS**
    - I had no prior knowledge on CSS language.
    - I had to learn CSS language along the way as well to understand how to make my webpage look better.
1. **Facebook Prophet Model**
    - I never tried Facebook Prophet Model before, I was only pretty familiar with LSTM models for time series analysis.
    - I had to learn how the model works, all the different hyper-parameters and how to interpret the results.
    - I am so glad that I learned this model, it is so powerful on seasonality trend analysis!
1. **Heroku Deployment**
    - I spent at least 5 more hours on deployment alone since there are so many different bugs going on for deploying an app! 
    - I had my LSTM model plugged in for the first edition, but PyTorch is such a big package that it went over slug size limitation of Heroku by too much so that I had to get rid of it.
    - I also had difficulty how to write codes so that Heroku can connect to the database as well as my environmental variables.  

Although it took me well over 50 hours to go through the entire project, I am so glad I did not give up in the middle. I learned so much new skills by doing this project and now I have a fully functional application for my own use!

## Next Steps and Plans
This project took me well over 50 hours to complete but unlocked so many new possibilities for me! I now have the ability to design my own websites and connect cloud based databases to them! With all the newly learned skills, my recent future plan is to build a similar website to do all my weight training analysis for me, and possibly combine both of them into one web application for the ease of use! I also plan to update this application if I learned some new techniques that can make it either function better or look better! 

I also plan to learn accessibility of HTML so that I can fit the application better on phone screens. Currently, this application is only for web browsers on computers, layouts does not fit well on phone screens.

If anyone is interested in how to lose weight or how to track weight, feel free to contact me!

## Tools Used
VS codes, Jupyter Notebook, Python, Dash, Heroku, SQL, HTML, CSS, Excel

## Data Science Techniques Used
PostgreSQL, Sqlite3, Plotly, Prophet Model, Time Series Analysis, Virtual Environment, Environmental Variables Handling, Scikit-Learn, Grid Search Cross Validation, Seasonality Trend Analysis, Serializing models to save in database, Pandas, Numpy, EDA, Logestic Regression, LSTM, Seaborn, Matplotlib, Supervised Machine Learning, etc.

## Main Files
For future refrences, I am not going to list all files in the repository since some of them are not that relevent. I will only point out the important ones:
- weight_tracker.py (this is the dash web application python file)
- Project Code.ipynb (this is the jupyter notebook file I have alone with the python file to support the main design of this application, including database creation, data cleaning, EDA and so much more)
- custome_styles.css (this is the css file that supports the web application for styles design)
- demo.gif (this is the demo you saw on top of this page)

## Usage
Feel free to check out the web based application and look around, it will always have my newest body weight data available because I will use it on a daily basis.
## Data Source
All data used in this project are my own real body weight records over the past 5ish years, and it will keep updating with new records daily.