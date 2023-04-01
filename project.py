import pandas as pd
import plotly.express as px

df = pd.read_csv("data1.csv")

score = df["TOEFL Score"].tolist()
result = df["GRE Score"].tolist()

fig = px.scatter(x = score, y = result)
fig.show()

import plotly.graph_objects as pg

chance = df["Chance of admit"].tolist()
colors = []
for data in chance:
    if data == 1:
        colors.append("green")
    else :
        colors.append("red")

fig  = pg.Figure(data=pg.Scatter(x = score, y = result, mode="markers",marker=dict(color = colors)))        
fig.show()

sr = df[['TOEFL Score', 'GRE Score']]

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

sr_train, sr_test, chance_train, chance_test = train_test_split(sr, chance, test_size=0.25, random_state=0)
print(sr_train)

cls = LogisticRegression(random_state=0)
cls.fit(sr_train, chance_train)

LogisticRegression(C = 1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, l1_ratio=None, max_iter=100, multi_class="auto", n_jobs=None, penalty='12', random_state=0, solver='lbfgs', tol=0.0001, verbose=0, warm_start=False )



chance_predict = cls.predict(sr_test)

from sklearn.metrics import accuracy_score
print('accuracy:' , accuracy_score(chance_test, chance_predict))

userScore =  int(input('Enter score:'))
userResult = int(input("Enter GRE score:" ))

user_test = sc_x.transform([[userScore, userResult]])

userPredict = cls.predict(user_test)

if userPredict[0] == 1:
    print("The User will get the job!")
else:
    print("The User may not get the job")    

print('Enter Score:')









