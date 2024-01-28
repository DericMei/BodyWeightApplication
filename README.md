# **BodyWeightMagicApplication**
The **BEST** project I have ever done! By utilizing machine learning tools, exploratory analysis, and dash, I built a web based application deployed by Heroku and running a PostgreSQL dabase in the background, to automate my entire body weight analysis for me! Check it out: [Eric Mei's Body Weight Magic Tool](https://bodyweightapp-f5c3d823ac56.herokuapp.com/).
## Introduction/Background
I was fat for as long as I could remember before 2014, my body weight was constantly over 230lbs and reached a peak over 250lbs in the year of 2014. Thats when I finally decided to lose weight.

I did lots of researches online and tried out so many different stuff myself and finally successfully lost 100 lbs in a year and began my fitness journey ever since.   
I got my personal trainer certificate from American Council on Exercise (ACE) and I am a certifited nutritional specialist from ACE as well. I learned how to be fit and stay fit. Thats when I started to monitor my body weight constantly. I lost a year or two of body weight data initially because I was using another app.   

After doing the last project of building a LSTM model using my body weight data, which is just for exploratory purposes, now I believe I can use my newly learned skills to build a web application of my own to do the job where I would do manually automatically for me!

The reason I got the idea of this project is that I have just started my yearly routine cutting phase. This is because every year I go on a cycle of gaining a few pounds during the winter and holiday seasons and start a new cutting phase the next year around the end of winter or start of spring to be back in six-pack shape. During this cutting phase, I count calories and monitor my body weight very closely to understand my caloric deficit state. And because there are no apps out there that can do what I needed to do, which is a weekly average body weight analysis, I have to manually calculate my weekly average every single week to understand my current cutting situations. Thats where I got the idea of automating the entire process and using even more machine learning tools to analyze underlying trends by building a web application.

(1kg of pure body fat is roughly 7700 Kcal, so on a weekly average base, during the cutting phase, I can easily know what is my daily caloric deficit, which I can further determine if I need to adjust my daily diet.)

Below is a picture of my personal body transformation of before and after. Check it out!!
<img src="https://raw.githubusercontent.com/DericMei/First-Trial/main/weight.JPG" width="600" height="400"/>

## Objectives
The main objective for this project is to build a comprehensive body weight analysis tool for myself to use which can automate my daily manual routine. In order to do that, I need the application to have the following functions:

- Record my body weight daily and upload it to a cloud based SQL database where it stores all my body weight data.
- Because I need to record the weight on the page, I cant risk people messing up my database, so I have to add a password function so that only by entering the correct password, one can record body weight and perform other operations such as train the model.
- Connect to the cloud based SQL database automatically and retrieve my bodyweight data to do analysis.
- Do all sorts of analysis and calculations for me automatically.
- Present the results of the analysis in a visually pleasing way so that I can easily know my current state.
- As a data scientist, I want to impliment machine learning tools to do some seasonality analysis and discover underlying trends for me, as well as predictions just for fun.
- I need a way to retrain my model directly on the web application so I do not need to go back go source codes to retrain models when new body weight data accumulates.

## Project Summary
To do the project, I need to have a jupyter notebook file on the side to test out codes as well as prepare my sql database.

## Methodology (Project building pipeline)

## Key Insights/Take-Aways

## Learnings/Obstacles

## Next Steps and Plans
This project took me well over 50 hours to complete but unlocked so many new possibilities for me! I now have the ability to design my own websites and connect cloud based databases to them! With all the newly learned skills, my recent future plan is to build a similar website to do all my weight training analysis for me, and possibly combine both of them into one web application for the ease of use! I also plan to update this application if I learned some new techniques that can make it either function better or look better! 

If anyone is interested in how to lose weight or how to track weight, feel free to contact me!

## Tools Used
VS codes, Jupyter Notebook, Python, Dash, Heroku, SQL, HTML, CSS, Excel

## Data Science Techniques Used
PostgreSQL, Sqlite3, Plotly, Prophet Model, Time Series Analysis, Environmental Variables Handling, Scikit-Learn, Grid Search Cross Validation, Pandas, Numpy, EDA, Logestic Regression, LSTM, Seaborn, Matplotlib, Supervised Machine Learning, etc.
## Main Files
For future refrences, I am not going to list all files in the repository since some of them are not that relevent. I will only point out the important ones:
- weight_tracker.py (this is the dash web application python file)
- Project Code.ipynb (this is the jupyter notebook file I have alone with the python file to support the main design of this application, including database creation, data cleaning, EDA and so much more)
- custome_styles.css (this is the css file that supports the web application for styles design)

## Usage
Feel free to check out the web based application and look around, it will always have my newest body weight data available because I will use it on a daily basis.
## Data Source
All data used in this project are my own real body weight records over the past 5ish years, and it will keep updating with new records daily.