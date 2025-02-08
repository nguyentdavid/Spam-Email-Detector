import pandas as pd  # For manipulating and analyzing data
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#:~:text=The%20multinomial%20Naive%20Bayes%20classifier,word%20counts%20for%20text%20label).
from sklearn.feature_extraction.text import TfidfVectorizer  # Text vectorization tool (NLP)
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer.fit_transform
from sklearn.naive_bayes import MultinomialNB  # Importing Multinomial Naive Bayes classifier to classify text (NLP)
import string  # To use string operations for text processing
import matplotlib.pyplot as plt  # To plot graphs


# Function to preprocess text
def convert_to_lower(text):
    # Converting all text (no special characters) to lowercase to analyze easier
    text = ''.join([char.lower() for char in text if char not in string.punctuation])
    return text


# Function to calculate accuracy
def calculate_accuracy(detected_emails, test_data_answers, label):
    # Determine correct predictions and calculate accuracy
    # Filtering test data's answers by their label (spam/ham), and adding them to 'correct'
    # if the contents of csv file match with any of the contents in test data's answers
    correct = test_data_answers[(test_data_answers['label'] == label) &
                                (test_data_answers['email'].isin(detected_emails['email']))]
    # Number of correct / total num of emails = accuracy
    accuracy = len(correct) / len(detected_emails)
    accuracy *= 100  # Converting to percentage
    return accuracy


# Function to plot accuracy results (help visualize)
def plot_accuracy(spam_accuracy, non_spam_accuracy):
    # Labels for bar chart
    labels = ['Spam', 'Ham (Non-spam)']
    accuracies = [spam_accuracy, non_spam_accuracy]  # Categories for each bar
    plt.figure()  # Creating the graph window
    plt.bar(labels, accuracies, color=['red', 'blue'])  # Creating bar chart
    plt.xlabel('Email Type')  # x-axis
    plt.ylabel('Accuracy %')  # y-axis
    plt.title('Comparing Email label Accuracies')  # Title
    plt.ylim(0, 100)  # y-axis range (0-100) to show percentages from 0 to 100
    plt.show()  # Displaying the chart/plot


# Main function to identify/classify spam/nonspam emails in a data set
def main():
    while True:
        try:
            training_dataset = int(input('Please select a CSV file to train the model:\n'
                                         '1: train_spam.csv\n2: train_spam2.csv\n3: train_spam3.csv\n'
                                         '0: Exit Program\n\nEnter a number 0-3: '))
            if training_dataset == 1:
                training_dataset = 'train_spam.csv'
            elif training_dataset == 2:
                training_dataset = 'train_spam2.csv'
            elif training_dataset == 3:
                training_dataset = 'train_spam3.csv'
            elif training_dataset == 0:
                print("Exiting program. Have a nice day!")
                break
            ### TRAINING THE MODEL PORTION ###
            # Loading training data with no header
            train_data = pd.read_csv(training_dataset, encoding='ISO-8859-1', header=None)
            # Ensure training data is read correctly
            print(f'Training model using {training_dataset}: ')
            print(train_data.head())  # Display first 5 rows of training data
            print(train_data.tail())  # Display last 5 rows of training data
            print('-----------------------------------------------')
            train_data.columns = ['label', 'email']  # Setting column names (files have column names)

            # Converting all emails to lowercase to easily manipulate
            train_data['processed_text'] = train_data['email'].apply(convert_to_lower)
            print("After cleaning up email content: ")
            print(train_data.head())
            print(train_data.tail())

            # Initializing and fit the TF-IDF (Term Frequency Inverse Document Frequency) vectorizer for NLP
            # ALlows program/model to convert frequency of words into vectors using statistical formulas
            # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
            vectorizer = TfidfVectorizer()
            # 'Learn vocab and IDF, return document-term matrix'
            # Extracting data on vocab and frequency and returning in matrix form, saving into x_train variable for analyzing
            x_train = vectorizer.fit_transform(train_data['processed_text'])
            # Extracting labels (spam/ham) into y_train variable
            y_train = train_data['label']  # Extract labels
            # Failed attempt to print the matrix with weights associated to words in the training dataset
            # Borrowed idea from https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a
            '''df_tfidfvect = pd.DataFrame(data=x_train.toarray(), index=[{training_dataset}], columns=y_train)
            print(df_tfidfvect)'''

            # Initializing classifier and training using x_train, y_train (data, labels)
            classifier = MultinomialNB()
            classifier.fit(x_train, y_train)
            ### END OF TRAINING THE MODEL PORTION ###

            ### BEGIN PREDICTING/CLASSIFYING EMAILS ###
            # Loading and preprocessing testing data
            # test_data = reading spam.csv using Latin-1 encoding
            test_data = pd.read_csv('spam.csv', encoding='ISO-8859-1')
            # Going row by row and converting each entire row into a string and saving into 'email' as a separate entry
            test_data['email'] = test_data.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
            # Converting each string/entry in 'email' into lowercase and saving into processed_text when done
            test_data['processed_text'] = test_data['email'].apply(convert_to_lower)

            # Predicting/classifying using testing data
            # converting processed text into vectors to be predicted/classified
            x_test = vectorizer.transform(test_data['processed_text'])
            # Predicting/classifying the processed_text after being trained above
            y_pred = classifier.predict(x_test)
            test_data['predicted_label'] = y_pred  # Storing predictions/labels into DataFrame

            # Saving the results to CSV files
            # filtering the test data's where predicted_label is 'spam'
            spam_emails = test_data[test_data['predicted_label'] == 'spam'][['email']]
            # Saving filtered test data's into spam_emails.csv (index=False required to prevent row indices when saving)
            spam_emails.to_csv('spam_emails.csv', index=False)
            # filtering the test data's where predicted_label is 'ham'
            non_spam_emails = test_data[test_data['predicted_label'] == 'ham'][['email']]
            # Saving filtered test data's into non_spam_emails.csv (index=False required to prevent row indices when saving)
            non_spam_emails.to_csv('non_spam_emails.csv', index=False)
            ### END OF PREDICTING/CLASSIFYING EMAILS ###

            ### BEGIN COMPARING TO ANSWER KEY ###
            # Loading test data's answer key (spam.answers.csv) data for comparison
            # Default encoding for pandas is UTF-8 but wasn't able to get it to work so switched to ISO-8859-1
            test_data_answers = pd.read_csv('spam_answers.csv', encoding='ISO-8859-1')
            test_data_answers.columns = ['label', 'email']  # Columns for the test data's answer key

            # Counting and printing number of spam/ham emails are in test data's answer key (spam.answers.csv)
            spam_count = test_data_answers[test_data_answers['label'] == 'spam'].count()['label']
            ham_count = test_data_answers[test_data_answers['label'] == 'ham'].count()['label']
            print(f'Total (real) number of spam emails in spam.csv: {spam_count}')
            print(f'Total (real) number of non-spam (ham) emails in spam.csv: {ham_count}')
            print('-----------------------------------------------')
            ### END COMPARING TO ANSWER KEY ###

            ### BEGIN ANALYSIS OF SPAM EMAIL PREDICTOR/CLASSIFIER ###
            # Calculating and printing accuracy of spam emails and ham emails
            # Will also print a few emails for both spam/ham
            spam_accuracy = calculate_accuracy(spam_emails, test_data_answers, 'spam')
            non_spam_accuracy = calculate_accuracy(non_spam_emails, test_data_answers, 'ham')
            print('Saving Spam emails in spam_emails.csv:')
            print(f'Number of spam emails detected: {len(spam_emails)}')
            print('Sample spam emails detected:')
            print((pd.read_csv('spam_emails.csv', encoding='ISO-8859-1', header=None)).head())
            print('-----------------------------------------------')
            print('Saving Non-Spam emails in non_spam_emails.csv:')
            print(f'Number of non-spam (ham) emails detected: {len(non_spam_emails)}')
            print('Sample non-spam emails detected:')
            print((pd.read_csv('non_spam_emails.csv', encoding='ISO-8859-1', header=None)).head())
            print('-----------------------------------------------')
            # Accuracy rounded to thousandth place
            print(f'Accuracy of spam detection: {spam_accuracy:.3f}%')
            print(f'Accuracy of non-spam detection: {non_spam_accuracy:.3f}%')

            # Plotting bar graph to visualize accuracy score comparisons
            plot_accuracy(spam_accuracy, non_spam_accuracy)

            while True:
                repeat = input("\nTry again?\nEnter 'y' for Yes\nEnter 'n' for No\nPlease enter y or n: ")
                if repeat.lower() == 'y':
                    print('-----------------------------------------------')
                    break
                elif repeat.lower() == 'n':
                    print("Exiting program. Have a nice day!")
                    return
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")

            ### END ANALYSIS OF SPAM EMAIL PREDICTOR/CLASSIFIER ###

        except ValueError:
            print("Invalid input. Please try again with a number 1-3.")
            print('-----------------------------------------------')


# Run main function/program
if __name__ == '__main__':
    main()
