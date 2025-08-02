import pandas as pd
import GWCutilities as util

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

print("\n-----\n")

#Create a variable to read the dataset
df = pd.read_csv("heartDisease_2020_sampling.csv")

print(
    "We will be performing data analysis on this Indicators of Heart Disease Dataset. Here is a sample of it: \n"
)

#Print the dataset's first five rows
firstfive = df.head()
print(firstfive)

input("\n Press Enter to continue.\n")



#Data Cleaning
#Label encode the dataset

print(
    "\nHere is a preview of the dataset after label encoding. This will be the dataset used for data analysis: \n"
)


df = util.labelEncoder(df, ["HeartDisease" , "Sex", "GenHealth", "Smoking", "AlcoholDrinking", "AgeCategory", "PhysicalActivity"])
print(df.head())


input("\nPress Enter to continue.\n")

#One hot encode the dataset
print(
    "\nHere is a preview of the dataset after one hot encoding. This will be the dataset used for data analysis: \n"
)

df = util.oneHotEncoder(df, ["Race"])
print(df.head())


input("\nPress Enter to continue.\n")



#Creates and trains Decision Tree Model
from sklearn.model_selection import train_test_split

X = df.drop("HeartDisease", axis = 1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

#print(X_train.head())


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth = 8, class_weight = "balanced", random_state = 1)
clf = clf.fit(X_train, y_train)





#Test the model with the testing data set and prints accuracy score
test_predict = clf.predict(X_test)

from sklearn.metrics import accuracy_score
test_acc = accuracy_score(y_test, test_predict)

print("The test accuracy is: "+str(test_acc))

#Prints the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, test_predict, labels = [1,0])
print("the confusion matrix of the tree is : \n"+str(cm))


#Test the model with the training data set and prints accuracy score
train_predict = clf.predict(X_train)

from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_train, train_predict)

print("The train accuracy is: "+str(train_acc))





input("\nPress Enter to continue.\n")



#Prints another application of Decision Trees and considerations
print("Decision Trees could be used to recommend books to read. It could take inputs such as age of the reader, genre they would like etc.")
print("\nThe model must not be biased towards a particular genre or book due to unbalanced data. \n For accuracy to be the best, the training and testing accuracy must be the highest values while being similar.")





#Prints a text representation of the Decision Tree
print("\nBelow is a text representation of how the Decision Tree makes choices:\n")
input("\nPress Enter to continue.\n")

util.printTree(clf, X.columns)