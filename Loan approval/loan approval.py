#oi vivliothkes pou xrhshmopoihsa 
import pandas as pd 
from sklearn import tree,metrics
from sklearn.model_selection import train_test_split
import pydotplus
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

#eisagwgh csv me relative path
df = pd.read_csv ('loan_approval.csv')

#gemizoume ta kena stoixeia me mean kai mode antistoixa
df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].mean())
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

#metatrepw ta stoixeia se arithimita dinontas tous tis times 0 h 1 
from sklearn.preprocessing import LabelEncoder
cols = ['Gender',"Married","Education",'Self_Employed',"Property_Area","Loan_Status"]
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])

dic = {'0': 0, '1': 1, '2': 2, '3+': 3}
df ['Dependents'] = df['Dependents'].map(dic)

#orizoume ta features kai ta target wste na mporoume na kanoume thn provlepsh
features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
X = df[features]
y = df['Loan_Status']

#apo ta dedomena dialegoume to 10% gia test kai to 90% gia to training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1) 

#xrhshmopoioume decision tree kai vriskoume to accuracy
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#kanoume export olo to decision tree
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)
print("\nAccuracy:",metrics.accuracy_score(y_test, y_pred),"\n")

#exagoume olo to decision tree 
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')


img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()

#o xrhsths eisagei ta dedomena xrhshmopoiontas trained dataset gia na kanei to prediction
gender=input('Give_gender ')
married=input('marriage_status ')
depender=input('depender ')
education=input('education_level ')
self=input('Self_Employed ')
applicant=input('ApplicantIncome ')
coapplicant=input('CoapplicantIncome ')
loan=input('Amount_of_the_loan ')
loanAmount=input('Loan_Amount_Term ')
credit=input('Credit_History ')
property=input('Property_Area ')
if(dtree.predict([[gender, married, depender, education, self, applicant, coapplicant, loan, loanAmount, credit, property]])==1):
    print('\nLoan accepted')
else:
    print('\nLoan declined')

