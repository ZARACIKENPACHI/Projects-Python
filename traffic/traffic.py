#oi vivliothikes pou xrisimopoiw
from sqlite3 import Time
import pandas as pd
from sklearn import tree,metrics
from sklearn.model_selection import train_test_split
import pydotplus
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

#eisagwgh csv arxeiou me relative path
df = pd.read_csv ('metakiniseis2.csv')

#perigrafw ta dedomena wste na vrw to 50% kai na diapistosw oti panw apo auto exoume kinhsh
print(df.describe())

#orizoume ta features kai ta target wste na mporoume na kanoume thn provlepsh
features =['appprocesstime','countedcars']
X=df[features]
y=df['average_speed'].apply(lambda y_value : 1 if y_value>=37.712428 else 0)

print(y)

#apo ta dedomena dialegoume to 20% gia test kai to 80% gia to training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#eisagoume to decision tree upologizoume to accuracy(xrisimopoiw entropy anti gia gini index giati tha mas dwsei kalitero accuracy)
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train.values, y_train)
y_pred = dtree.predict(X_test.values)
print("\nAccuracy:",metrics.accuracy_score(y_test, y_pred),"\n")

#kanoume export olo to decision tree
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')


img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()

#o xrhsths eisagei ta dedomena xrhshmopoiontas trained dataset gia na kanei to prediction
Time=input('appprocesstime ')
count=input('countedcars ')
if(dtree.predict([[Time, count ]])==0):
    print('\nNot much traffic')
else:
    print('\nA lot of traffic')