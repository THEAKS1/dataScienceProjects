import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

data = pd.read_csv("train.csv")
data = data.dropna()

data["Date"] = pd.to_datetime(data["Date"])

data["Day"] = [x.day for x in data["Date"]]
data["Month"] = [x.month for x in data["Date"]]
data["Year"] = [x.year for x in data["Date"]]

data["postcode"] = [re.search("^[A-Z]*", i)[0] for i in data["postcode"]]


# Converting Local_Authority_(Highway) to numerical data

def special_cat_to_num(df, n):
    temp = df.unique()
    temp.sort()
    
    thresholds = []
    for i in range(0,len(temp),n):
        thresholds.append(temp[i])
    
    x = []
    flag = False
    for i in df:
        for j in range(1,len(thresholds)):
            if i < thresholds[j]:
                x.append(j-1)
                flag = True
                break
        if not flag:
            x.append(j)
        flag = False
            
    return x

data["Local_Authority_(Highway)"] = special_cat_to_num(data["Local_Authority_(Highway)"], 15)
data["postcode"] = special_cat_to_num(data["postcode"], 10)
data["Hour"] = pd.to_datetime(data["Time"], format = "%H:%M").dt.hour
data["Minute"] = pd.to_datetime(data["Time"], format = "%H:%M").dt.minute

data = data.drop(["Accident_ID","Date","Time"], axis = 1)


categorical_data = ["Day_of_Week","Local_Authority_(Highway)", "1st_Road_Class", "Road_Type", "Speed_limit", "2nd_Road_Class", "Pedestrian_Crossing-Human_Control", "Pedestrian_Crossing-Physical_Facilities",
                    "Light_Conditions", "Weather_Conditions","Road_Surface_Conditions", "Special_Conditions_at_Site", "Carriageway_Hazards", "Urban_or_Rural_Area", "Did_Police_Officer_Attend_Scene_of_Accident",
                    "state", "postcode", "country"]
for i in categorical_data:
    encoder = LabelEncoder()
    data[i] = encoder.fit_transform(data[i])
    
col_to_move = data.pop("Number_of_Casualties")
data.insert(0, "Number_of_Casualties", col_to_move)

X = data.iloc[:,1:]
y = data.iloc[:,0].values

for i in categorical_data:
    one_hot_encoder = OneHotEncoder()
    temp = one_hot_encoder.fit_transform(X[[i]]).toarray()
    temp = pd.DataFrame(temp, columns=[i+str(x) for x in range(len(temp[0]))])
    X = X.join(temp)