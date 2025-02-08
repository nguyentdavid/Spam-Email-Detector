# Spam-Email-Detector
Cybersecurity Programming Project

Program Description: This program successfully predicts/classifies emails based off of their content.
The program/model is initially trained using a CSV folder which has labels classifying different
emails as spam or ham. The user has an option to pick between 3 different training sets to see different
results when predicting/classifying a test set (spam.csv). After predicting/classifying the test set,
the spam emails and ham emails are sorted and filtered accordingly so that all predicted spam emails are
saved in spam_emails.csv while all ham emails are saved in non_spam_emails.csv. The program will also
produce a bar graph to visualize the accuracy comparisons between spam and ham email detection results.

SEE SCREENSHOTS.PDF FOR PROGRAM RESULTS

DESCRIPTION FOR EACH SECTION:
Each section of the code has been marked and divided for readability. There are 4 total portions of the code (excluding functions)
where the program will have a different objective. 
TRAINING THE MODEL: The program takes in an input from the user for the 
CSV file that will be used to train the model. This is to see how different training datasets will impact the model differently. 
The program will load the training dataset by reading the CSV file using ISO-8859-1 encoding (pandas default UTF-8 wasn’t working) 
and converts the data into all lowercase for data manipulation. This will make it easier for the model to read/understand as everything 
is now similar style (lowercase). It will then create columns (training dataset must have preset column names of ‘label’ and ‘email’) 
to separate the CSV data in order to train the model. The program then uses scikit-learn’s 
TfidfVectorizer (Term Frequency Inverse Document Frequency) to extract the and analyze/train the model based off the ‘email’ column 
and the ‘label’ column. The ‘email’ column uses the vectorizer in order to retrieve the frequency of each word/phrase and 
returns in a matrix form for analysis. A classifier/predictor will then be initialized and trained on the data from above.

PREDICTING/CLASSIFYING EMAILS: The program reads a preset CSV file, spam.csv, (borrowed from KunjMaheshwari, Github) using ISO-8859-1 encoding 
to stay consistent with training the model. This file only contains spam/ham email contents without any column names or labels. 
The program does the exact same as above, except it converts every row in the CSV file into a string and saving it as its own index/entry. 
Once this is done, each string will be converted into lowercase and saved as ‘processed_text’ for easier data manipulation/analysis. 
The ‘processed_text’ will be put into the vectorizer to convert the data into vectors in order for the classifier (created at the end of TRAINING MODEL section) 
to predict whether or not the contents are Spam or Ham (and each string will be labeled accordingly). All ‘Spam’ emails that were predicted 
will be saved in a new CSV file named ‘spam_emails.csv’ while all ‘Ham’ or non-spam emails will be saved in ‘non_spam_emails.csv’. 

COMPARING TO ANSWER KEY: The program once again reads a different CSV file (spam_answers.csv) using encoding ISO-8859-1 as same from above. 
It will only have 2 columns, 1 for the label and 1 for the email content. The program will then count how many spam emails there are as well 
as how many ham emails there are in order to calculate the accuracy of the model later on.

SPAM EMAIL PREDICTOR/CLASSIFIER ANALYSIS: The program calculates the model’s prediction/classifier accuracy by using the ‘calculate_accuracy’ function 
created. This compares the number of spam/ham emails that the model predicted/classified to the ACTUAL number of spam/ham emails in the answer key. 
The program will then print out a few of the emails the model predicted/classified in both spam/ham CSV files. It will also print out the 
accuracy of the model’s ability to correctly predict/classify spam and ham emails separately. 
The program then uses matplotlib and the ‘plot_accuracy’ to graph a bar graph to visualize the 2 accuracy scores compared to each other. 


Setup Directions:
Open whatever IDE you will be using and ensure that Python is supported (I built this program in PyCharm).
Python version 3.9 is going to be the minimum requirement for the system to run the 
program (newer versions will work and my system runs on Python 3.9.12). 
Install all packages and libraries stated in the materials (pandas, scikit-learn, matplotlib). 
Download the CSV files (spam.csv, train_spam.csv, train_spam2.csv, train_spam3.csv, spam_answers.csv) 
and save them in the same location as your program. 
Run the main program in your local IDE OR access your computer’s terminal and navigate to 
the location where the program and all the CSV files are saved. 
You can then in your terminal use ‘python3 SpamDetector.py’ to run the program. 
Follow the instructions and enjoy!

Usage:
spam.csv (test set) should be only 1 column with all emails' content. 
Each row should have content for 1 email (spam or ham) only.
Example: 
YOU HAVE WON 2000 CASH CLAIM NOW!!!
URGENT!!! YOU HAVE BEEN HACKED!!! CALL NOW TO FIX!!!

train_spam.csv, train_spam2.csv, train_spam3.csv should be in the format of 2 columns.
First column starts with 'label' and every row after that will hold the label accoringly to the email.
Second column starts with 'email' and every row after that will hold the email content accordingly to the label.
Example:
label	email
spam	WINNER!!! claim your rewards call 123958312
ham		Sorry John, I'll call you later.

Borrowed Components: 
Training sets were borrowed from 
https://github.com/mshenoda/roberta-spam/blob/main/data/spam_message_test.csv
And
https://github.com/KunjMaheshwari/Email-Spam-Detector/blob/master/spam.csv
KunjMaheshwari’s spam.csv was used to select equal amounts of spam/ham emails to 
build train_spam.csv and train_spam2.csv and the model attempted to predict/classify on spam.csv
mshenoda’s spam_message_test.csv was used to select equal amounts of spam/ham emails to create train_spam3.csv

Important Components:
Python environment (PyCharm, Jupyter notebook, VSCode, etc.)
Download necessary libraries/packages: pandas, scikit-learn
* Import necessary libraries/packages and functions: 
* Import TfidfVectorizer from scikit-learn’s feature_extraction.text
* https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer.fit_transform
* Import MultinomialNB from scikit-learn’s naive_bayes 
* https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#:~:text=The%20multinomial%20Naive%20Bayes%20classifier,word%20counts%20for%20text%20classification).
* Import string to use string operations for processing text
* Import matplotlib.pyplot to create graphs to visualize results
* CSV file that holds training data for the program (train_spam.csv OR train_spam2.csv)
* CSV file that holds the testing data for the program (spam.csv)
* CSV file that holds the answers for the testing data (spam_answers.csv)

FUTURE COMPONENTS/ADDITIONS:
Better/more analysis (accuracy, precision, etc.)
Implement within an email application (outlook, gmail, yahoo, etc.)
Add natural language processing for special characters and more
