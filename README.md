# Political-Analyser
Political Sentiment Analysis using NLP and Machine Learning in Python.
Political Analyser is a Natural Language Processing Sentiment Analysis System that works on data obtained from twitter, to predict election result outcomes.
This political analyser is currently configured to analyse political opinion and predict election results for three parties only, i.e., The Aam Aadmi Party, Bhartiya Janta Party, and The Indian National Congress with focus on the political opinions and elections in Delhi.

Hardware Requirements: 
RAM: 4 GB or above, Processor: Intel i3 or above, or any other processor with similar configuration, Graphics Card: 1 GB or above, Hard Disk Space: 200 MB or above.

Software and Tools Required:
Operating System: Windows 7 or above.

Python Versions: Python 2.7 (with IDLE), Python 3.5 (with IDLE).
Download the required python versions from: https://www.python.org/downloads/
Install by running the installer.
Usually, these python versions come with their repective IDLEs already present in the package.

**Note:
IDLE provides both script editing and script execution facilities, hence no explicit editors and frameworks are required.
To run a script in IDLE, right click on the script and select the option ‘Edit with IDLE’ for respective python version.
In the IDLE Editor window, either press F5 key or go to the ‘Run’ menu in the menu bar and select the option ‘Run module’, to run the script in Shell Window.
**

Database: MongoDB
Download MongoDB from: https://www.mongodb.com/download-center#community
1.	Select your PC and OS configuration and download the ‘Community Server’ edition.
2.	Open the installer to install MongoDB.
3.	Create a directory ‘data’ in the System drive (C:). Inside the ‘data’ directory create another directory ‘db’.
4.	Create a new user environment variable ‘PATH’ with address value: C:\Program Files\MongoDB\Server\3.4\bin
5.	Now open the command window and type: mongod 
to start the database server.
The server listens at port 27017. To close the database server, close the command window.
Python Packages required:
To install a python package:
1.	Go to the folder ‘Scripts’ (for e.g. -> C:\Python35\Scripts) in the respective python version folder for which you want to install the package.
2.	Open the command window inside the ‘Scripts’ folder by pressing ‘Ctrl’ + ‘Shift’ + ‘right click’, and select ‘open command window here’ option.
3.	Inside the command prompt type the command: pip install package_name.
4.	The package will get automatically downloaded and installed.
5.	To check the package version installed, type in the command window: pip show package_name.

The required packages needed to be installed are:
pymongo
tweepy
time
preprocessor
textblob
tweet-preprocessor
subprocess
matplotlib
pandas
unidecode
nltk
scikit-learn
numpy
scipy
itertools

The python packages that cannot be installed by pip correctly, have been provided as .whl files for installation. These are: numpy-1.12.1+mkl-cp27-cp27m-win_amd64.whl, scipy-0.19.0-cp27-cp27m-win_amd64.whl
To install these packages, copy and paste them in the ‘Scripts’ folder of your respective python version (Python 2.7, in this case). Then open the command window there, and type: pip install package_name (with extension; .whl).
For e.g. -> pip install numpy-1.12.1+mkl-cp27-cp27m-win_amd64.whl



Running the System:
Now to actually run the system and get the Sentiment Analysis Prediction results, follow these steps:

1. Create a developer id on twitter.

2. Create an app on the developer id, as follows:

Step 2.1 : Sign in to apps.twitter.com with your twitter id to create an app:

Step 2.2: After login, click on create an app to start creating a new app.
 
Step 2.3: At the create app menu add your app name, its description, the website for which it is used. This fully-qualified URL is used in the source attribution for tweets created by your application and will be shown in user-facing authorization screens.
(If you don't have a URL yet, just put a placeholder here but remember to change it later.)
To restrict the app from using callbacks, leave the Callback URL field empty.
 
Step 2.4: After the app is created, go to the Keys and Access Tokens menu, there the consumer key and consumer secret are already present. To generate the Access tokens, click on Access Tokens button. Your Access Token and Access Token Secret is generated:

All these four keys are important at the time of actual tweet extraction through OAuth and must not be shared with anyone.



**Note: Run all the below mentioned scripts in python 3.5 IDLE, unless stated otherwise.

3. To get the analysis results using Textblob classifier, run the script ‘start.py’.
The timeline graphs for all the three parties will get plotted.
To get a well plotted timeline graph run the script once every day (like throughout a week).
For testing purposes some timeline values have already been stored in the aap.txt, bjp.txt and inc.txt.
So the script could be run once only, and it will give well plotted timeline graphs for all the three parties respectively.

4. To get the Sentiment Analysis metrics (using Textblob classifier) for each party, run the script ‘metrics_for_textblob.py’.
When prompted for file name enter the filename of the party you want metrics for, with the .csv extension. The three choices are: aap.csv, bjp.csv, and inc.csv.
The tweets in these three csv files have been pre-labelled 1(positive) or 0(negative), by our side and thus can be used for training and testing purposes.
Otherwise, if you want to create your own .csv files for each party and label the tweets in them by your own side, run the three scripts: aap_call.csv, bjp_call.csv, and inc_call.csv. These will create three .csv files for the three parties respectively, for which you would have to label the tweets 1(+ve) or 0(-ve) under a ‘sentiment’ column next to the ‘text’ column, from your side.


**Note: Run all the below mentioned scripts in python 2.7 IDLE, unless stated otherwise.

5. To get the classification results for Naïve- Bayes classifier and Vader classifier on a very large labelled dataset ‘Senti_t1.csv’ (13000 tweets about the USA 2016 presidential GOP debate), run the script ‘nb_vader_gop.py’.
This script takes some time to process and display the results (Sentiment Analysis metrics) in the Shell Window.

6. To get the results for Vader classifier on the aap.csv, bjp.csv and inc.csv run the three scripts (in order): vader_aap.py, vader_bjp.py, and vader_inc.py.
These scripts will plot the prediction bar-graph on the unlabeled tweets data, and plot the confusion matrices (both normalized and non-normalized) and the classification report, for each labelled dataset respectively.
The classification reports are saved as .png files with names ‘test_plot_classif_report_aap.png’, ‘test_plot_classif_report_bjp.png’, and ‘test_plot_classif_report_inc.png’, respectively, for each party.

