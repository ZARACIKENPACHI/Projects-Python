#oi vivliothikes pou xrisimopoiw
import pandas as pd
from sklearn import tree,metrics
from sklearn.model_selection import train_test_split
import pydotplus
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

#eisagwgh csv arxeiou me relative path
df = pd.read_csv ('Quality_of_red_wine.csv')

#tha vroume posa monadika values yparxoun gia to quality  
print(df['quality'].unique())

#orizoume ta features kai ta target wste na mporoume na kanoume thn provlepsh
features = ['fixed acidity', 'volatile acidity','citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
X = df[features]

#xwrizw ta krasia analoga thn poiothta toys, an einai to y mikrotero to 7 dinw thn timh 0 pou shmainei oti einai kakhkhs poiothtas alliws dinw thn timh 1
y = df['quality'].apply(lambda y_value : 1 if y_value>=7 else 0)

#apo ta dedomena dialegoume to 20% gia test kai to 80% gia to training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#xrhshmopoioume decision tree kai vriskoume to accuracy
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#eisagoume to decision tree upologizoume to accuracy(xrisimopoiw entropy anti gia gini index giati tha mas dwsei kalitero accuracy)
dtree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dtree = dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)
print("\nAccuracy:",metrics.accuracy_score(y_test, y_pred),"\n")

#kanoume export olo to decision tree
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')


img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()

#o xrhsths eisagei ta dedomena xrhshmopoiontas trained dataset gia na kanei to prediction
fixed=input('fixed acidity ')
volatile=input('volatile acidity ')
citric=input('citric acid ')
residual=input('residual sugar ')
chlorides=input('chlorides ')
frees=input('free sulfur dioxide ')
totals=input('total sulfur dioxide ')
density=input('density ')
pH=input('pH ')
sulphates=input('sulphates ')
alcohol=input('alcohol ')
if(dtree.predict([[fixed, volatile, citric, residual, citric, frees, totals, density, pH, sulphates, alcohol]])==0):
    print('\nBad wine')
else:
    print('\nExcellent  wine')