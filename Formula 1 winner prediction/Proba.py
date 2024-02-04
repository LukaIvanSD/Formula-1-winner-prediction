import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import utils_nans1 as line


result_df = pd.read_csv('results.csv')
stats_df = pd.read_csv('status.csv')
drivers_df = pd.read_csv('drivers.csv')
races_df = pd.read_csv('races.csv')
constructor_df = pd.read_csv('constructors.csv')
driver_standings_df = pd.read_csv('driver_standings.csv')
preseason_df=pd.read_csv('F1Testing.csv')
qualifying_df=pd.read_csv('qualifying.csv')
pitstops_df=pd.read_csv('pit_stops.csv')
weather_df=pd.read_csv('weather.csv')
safety_cars_df=pd.read_csv('safety_cars.csv')
broj_ponavljanja = weather_df['circuit_id'].value_counts()

#df=pd.DataFrame()
#df.to_csv("Fast.csv",index=False)
'''
import fastf1
season_df=fastf1.get_session(2020,7,'FP1')
season_df.load()

pd_df=pd.DataFrame(season_df.session_info)
pd_df.to_csv("Fast.csv",index=False)
pd_df=pd.DataFrame(season_df.total_laps)
pd_df.to_csv("Fast1.csv",index=False)
'''
'''
season_df=fastf1.get_session(2020,15,'R')
season_df.load(weather=True)
print(season_df.weather_data)
pd_df=pd.DataFrame(season_df.weather_data)
pd_df.to_csv("Fast1.csv",index=False)
# Ispis rezultata
'''
#print(broj_ponavljanja)
race_df = races_df[["raceId", "year", "round", "circuitId"]].copy()
#EDA AND PRETPOCESSING
import pandas as pd
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

#sortiram po godini kako bi izbacio sve odatke pre 2000 jer su formule drugacije bile
race_df = race_df.sort_values(by=['year', 'round'])
race_df = race_df[race_df["year"] >= 2000]

#izdvajam najbitnije feature iz rezultata
res_df = result_df[['raceId', 'driverId', 'constructorId', 'grid', 'positionOrder']].copy()

#spajanje rezultata sa trkom
df = pd.merge(race_df, res_df, on='raceId')
#print(df.head())


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#TOP 3 za vozace prosla godina
df['Top 3 Finish'] = df['positionOrder'].le(3).astype(int)
#print(df.head())

# Calculating the total number of races and top 3 finishes for each driver in each year
driver_yearly_stats = df.groupby(['year', 'driverId']).agg(
    Total_Races=('raceId', 'nunique'),
    Top_3_Finishes=('Top 3 Finish', 'sum')
).reset_index()

#print("Driver annual stats")
#print(driver_yearly_stats)

# Calculating the percentage of top 3 finishes for each driver in each year
driver_yearly_stats['Driver Top 3 Finish Percentage (This Year)'] = (driver_yearly_stats['Top_3_Finishes'] / driver_yearly_stats['Total_Races']) * 100

# Shifting the driver percentages to the next year for last year's data
driver_last_year_stats = driver_yearly_stats.copy()
driver_last_year_stats['year'] += 1
driver_last_year_stats = driver_last_year_stats.rename(columns={'Driver Top 3 Finish Percentage (This Year)': 'Driver Top 3 Finish Percentage (Last Year)'})

df = pd.merge(df, driver_last_year_stats[['year', 'driverId', 'Driver Top 3 Finish Percentage (Last Year)']], on=['year', 'driverId'], how='left')

# Checking the merged data
#print("New dataframe")
#print(df[df["year"]>=2000])




#TOP 3 za konstruktore prosla godina

# Calculating mean of top 3 finishes percentages for the two drivers in each constructor last year
constructor_last_year_stats = df.groupby(['year', 'constructorId', 'round']).agg(
    Sum_Top_3_Finishes_Last_Year=('Driver Top 3 Finish Percentage (Last Year)', 'sum')
).reset_index()

# Calculating the percentage of top 3 finishes for each constructor last year
constructor_last_year_stats['Constructor Top 3 Finish Percentage (Last Year)'] = constructor_last_year_stats["Sum_Top_3_Finishes_Last_Year"]/2

df = pd.merge(df, constructor_last_year_stats[['year', 'constructorId', 'round', 'Constructor Top 3 Finish Percentage (Last Year)']], on=['year', 'constructorId', 'round'], how='left')

# Checking the merged data
#print("New dataframe")
#print(df[df["year"]>=2000])




#DRIVER TOP 3 THIS YEAR

# Creating a function to calculate the top 3 finish percentage before the current round for drivers
def calculate_driver_top_3_percentage_before_round(row, df):
    # Filter for races in the same year, for the same driver, but in earlier rounds
    previous_races = df[(df['year'] == row['year']) & (df['driverId'] == row['driverId']) & (df['round'] < row['round'])]
    if len(previous_races) == 0:
      return pd.NA

    total_races = previous_races['raceId'].nunique()
    top_3_finishes = previous_races['Top 3 Finish'].sum()

    # Calculate the percentage
    return (top_3_finishes / total_races) * 100 if total_races > 0 else pd.NA

# Apply the function to each row in the DataFrame
df['Driver Top 3 Finish Percentage (This Year till last race)'] = df.apply(lambda row: calculate_driver_top_3_percentage_before_round(row, df), axis=1)



#Za konstruktore top 3 do sadasnje trke

# Calculating mean of top 3 finishes percentages for the two drivers in each constructor this year
constructor_this_year_stats = df.groupby(['year', 'constructorId', 'round']).agg(
    Sum_Top_3_Finishes_This_Year=('Driver Top 3 Finish Percentage (This Year till last race)', 'sum')
).reset_index()

#print("Constructor annual stats")
#print(constructor_this_year_stats)

# Calculating the percentage of top 3 finishes for each constructor this year
constructor_this_year_stats['Constructor Top 3 Finish Percentage (This Year till last race)'] = constructor_this_year_stats["Sum_Top_3_Finishes_This_Year"]/2

df = pd.merge(df, constructor_this_year_stats[['year', 'constructorId', 'round', 'Constructor Top 3 Finish Percentage (This Year till last race)']], on=['year', 'constructorId', 'round'], how='left')

# Checking the merged data
#print("New dataframe")
#print(df[df["year"]>=2000])



#PROSECNA POZICIJA PROSLE GODINE za vozace

# Calculating the total number of races and top 3 finishes for each driver in each year
driver_yearly_stats = df.groupby(['year', 'driverId']).agg(
    Total_Races=('raceId', 'nunique'),
    Avg_position=('positionOrder', 'mean')
).reset_index()


# Calculating the percentage of top 3 finishes for each driver in each year
driver_yearly_stats['Driver Avg position (This Year)'] = driver_yearly_stats['Avg_position']

# Shifting the driver percentages to the next year for last year's data
driver_last_year_stats = driver_yearly_stats.copy()
driver_last_year_stats['year'] += 1
driver_last_year_stats = driver_last_year_stats.rename(columns={'Driver Avg position (This Year)': 'Driver Avg position (Last Year)'})

df = pd.merge(df, driver_last_year_stats[['year', 'driverId', 'Driver Avg position (Last Year)']], on=['year', 'driverId'], how='left')

# Checking the merged data
#print("New dataframeGAAAS",df.columns)
#print(df[df["year"]>=2000])




#za konstruktore prosecna pozicija prosle godine


# Calculating mean of top 3 finishes percentages for the two drivers in each constructor last year
constructor_last_year_stats = df.groupby(['year', 'constructorId', 'round']).agg(
    sum_position_last_year=('Driver Avg position (Last Year)', 'sum')
).reset_index()

#print("Constructor annual stats")
#print(constructor_last_year_stats)

# Calculating the percentage of top 3 finishes for each constructor last year
constructor_last_year_stats['Constructor Avg position (Last Year)'] = constructor_last_year_stats["sum_position_last_year"]/2

df = pd.merge(df, constructor_last_year_stats[['year', 'constructorId', 'round', 'Constructor Avg position (Last Year)']], on=['year', 'constructorId', 'round'], how='left')

# Checking the merged data
#print("New dataframe")
#print(df[df["year"]>=1983])


#--------------------------------------------------------------

#za vozace prosecna pozicija ove godine 


def calculate_driver_avg_position_before_round(row, df):
    # Filter for races in the same year, for the same driver, but in earlier rounds
    previous_races = df[(df['year'] == row['year']) & (df['driverId'] == row['driverId']) & (df['round'] < row['round'])]
    if len(previous_races) == 0:
      return pd.NA
    # Calculate the total races and sum of positions
    total_races = previous_races['raceId'].nunique()
    positionSum = previous_races['positionOrder'].sum()

    # Calculate average position
    return (positionSum / total_races) if total_races > 0 else pd.NA

# Apply the function to each row in the DataFrame
df['Driver Average Position (This Year till last race)'] = df.apply(lambda row: calculate_driver_avg_position_before_round(row, df), axis=1)

#---------------------------------------------------------------------------

#za konstruktore prosecna pozicija ove godine


# Calculating mean of top 3 finishes percentages for the two drivers in each constructor this year
constructor_this_year_stats = df.groupby(['year', 'constructorId', 'round']).agg(
    sum_Position_Constructor = ('Driver Average Position (This Year till last race)', 'sum')
).reset_index()

#print("Constructor annual stats")
#print(constructor_this_year_stats)

# Calculating the percentage of top 3 finishes for each constructor this year
constructor_this_year_stats['Constructor Average Position (This Year till last race)'] = constructor_this_year_stats["sum_Position_Constructor"]/2

df = pd.merge(df, constructor_this_year_stats[['year', 'constructorId', 'round', 'Constructor Average Position (This Year till last race)']], on=['year', 'constructorId', 'round'], how='left')

# Checking the merged data
#print("New dataframe")
#print(df[df["year"]>=2000])


#print(df[(df["year"] == 2023)& (df["round"] > 3) ].head(30))

#--------------------------------------------------------------------------------
# Histogram for Top 3 Finishes
df['Top 3 Finish'].value_counts().plot(kind='bar')
plt.title('Frequency of Top 3 Finishes')
plt.xlabel('Top 3 Finish')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
print(plt.show())
#-----------------------------------------------------------------
df2=pd.DataFrame(df)
df2.to_csv('output3.csv', index=False)
#We drop information related to the result of this race (e.g. finishing position) from the dataframe to prevent data leakage
df_final = df.drop(labels=["raceId"], axis=1)
print("Number of rows in total:", df_final.shape[0])

# Count rows where 'year' is not 1982 before dropping NaN values
initial_count = len(df_final[df_final['year'] != 2000])


#NEDOSTAJUCE VREDNOSTI
df_final['Driver Top 3 Finish Percentage (Last Year)'] = df_final['Driver Top 3 Finish Percentage (Last Year)'].fillna(0.0)
df_final['Constructor Top 3 Finish Percentage (Last Year)'] = df_final['Constructor Top 3 Finish Percentage (Last Year)'].fillna(0.0)
df_final['Driver Avg position (Last Year)'] = df_final['Driver Avg position (Last Year)'].fillna(15.0)
df_final['Constructor Avg position (Last Year)'] = df_final['Constructor Avg position (Last Year)'].fillna(8.0)
# Count rows where 'year' is not 2000 after dropping NaN values
final_count = len(df_final[df_final['year'] != 2000])
df_final = df_final.dropna()

# Calculate the number of rows dropped
rows_dropped = initial_count - final_count

print("Number of rows dropped where year is not 2000:", rows_dropped)



df_final_keepPositionOrder = df_final.copy()
df_final = df_final.drop(["positionOrder"], axis = 1)
print("GAAAAS")
print(df_final.head())

#---------------------------------------------------
#EXPLORATORY DATA ANALYSIS
df_final["Driver Top 3 Finish Percentage (This Year till last race)"] = df_final["Driver Top 3 Finish Percentage (This Year till last race)"].astype(float)
df_final["Constructor Top 3 Finish Percentage (This Year till last race)"] = df_final["Constructor Top 3 Finish Percentage (This Year till last race)"].astype(float)
df_final["Driver Average Position (This Year till last race)"] = df_final["Driver Average Position (This Year till last race)"].astype(float)
df_final["Constructor Average Position (This Year till last race)"] = df_final["Constructor Average Position (This Year till last race)"].astype(float)
df_final_keepPositionOrder["Driver Top 3 Finish Percentage (This Year till last race)"] = df_final_keepPositionOrder["Driver Top 3 Finish Percentage (This Year till last race)"].astype(float)
df_final_keepPositionOrder["Constructor Top 3 Finish Percentage (This Year till last race)"] = df_final_keepPositionOrder["Constructor Top 3 Finish Percentage (This Year till last race)"].astype(float)
df_final_keepPositionOrder["Driver Average Position (This Year till last race)"] = df_final_keepPositionOrder["Driver Average Position (This Year till last race)"].astype(float)
df_final_keepPositionOrder["Constructor Average Position (This Year till last race)"] = df_final_keepPositionOrder["Constructor Average Position (This Year till last race)"].astype(float)

# Average finish position per year
avg_finish_per_year = df.groupby('year')['positionOrder'].mean()
avg_finish_per_year.plot(kind='line')
plt.title('Average Finish Position per Year')
plt.xlabel('Year')
plt.ylabel('Average Finish Position')
print(plt.show())
# heatmap
plt.figure(figsize=(10,7))
sns.heatmap(df_final_keepPositionOrder.corr(), annot=True, mask = False, annot_kws={"size": 7})
print(plt.show())
#UTICAJ GRID NA positionOrder
#------------------------------------------------------------------------------------
print("------------Uticaj GRID na positionOrder-------------------")
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
data = df_final_keepPositionOrder[['grid', 'positionOrder']]

X_train, X_test, y_train, y_test = train_test_split(data[['grid']], data['positionOrder'], test_size=0.2, random_state=42)

regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

coef_grid = regressor.coef_[0]
print(f'Koeficijent za atribut "grid": {coef_grid:.2f}')
print("------------------------------------------------------------------------------")
'''
#UTICAJ rezultata treninga na GRID
#------------------------------------------------------------------------------------
print("------------ rezultata treninga na GRID-------------------")
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
qualifying_df.dropna()
qualifying_df['q1'] = pd.to_timedelta(qualifying_df['q1'])

# Konvertujte vrednosti u sekunde kao float
qualifying_df['q1_seconds'] = qualifying_df['q1'].dt.total_seconds()

# Sada možete koristiti 'q1_seconds' kolonu kao numerički tip
X = qualifying_df[['q1_seconds', 'raceId', 'driverId']]


X_train, X_test, y_train, y_test = train_test_split(X, qualifying_df['position'], test_size=0.2, random_state=42)

regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

coef_grid = regressor.coef_[0]
print(f'Koeficijent za atribut "grid": {coef_grid:.2f}')
print("------------------------------------------------------------------------------")
'''
#-------------------------------------------------------------------------------------
#unos godine
year = input("Unesite Godinu: ")
year=int(year)
round=input("Unesite rundu: ")
round=int(round)
#DODAVANJE STATUSID KOLONE
result_df=result_df.merge(races_df[['raceId', 'year','round']], on='raceId', how='left')
result_df = result_df.sort_values(by=['year', 'round', 'position'], ascending=[True, True, False])
result_df.to_csv("statucic.csv",index=False)
df_final_keepPositionOrder = df_final_keepPositionOrder.merge(result_df[['year','round','driverId', 'statusId']], on=['year', 'round', 'driverId'], how='left')
vrednosti_i_brojevi = df_final_keepPositionOrder['statusId'].value_counts(normalize=True)
for vrednost, broj in vrednosti_i_brojevi.items():
    if broj>1: 
        print(f'Vrednost: {vrednost}, Broj pojavljivanja: {broj}')
finished=np.array([1])  
finishedlater=np.array([11,12,13])         
df_final_keepPositionOrder['statusId'] = df_final_keepPositionOrder['statusId'].apply(
    lambda x: 1 if x in finished else 2 if x in finishedlater else 3
)
#PRAVLJENJE X I Y PODATAKA



filtered_df = df_final_keepPositionOrder[df_final_keepPositionOrder['year'] < year]
y =  filtered_df["positionOrder"]
x =filtered_df.drop(columns=["positionOrder","Top 3 Finish"])
x.to_csv("x.csv",index=False)
y.to_csv("y.csv",index=False)
points = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1])
weights = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2,2,1.0,1.0,2,2])
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# Podela podataka na trening i test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#---------------------------------------------------------------------------------------------------
print("---------------------------REGRESIJA----------------------------------------------------------")
#Linearna regresija model
from sklearn.linear_model import Lasso
regressor = LinearRegression()
# 4. Treniranje modela
X = sm.add_constant(X_train)  # dodajte konstantu za računanje intercepta
model = sm.OLS(y_train, X_train).fit()

# Ispis rezultata regresije
print(model.summary())
regressor.fit(X_train, y_train)
# 5. Evaluacija modela
y_pred = regressor.predict(X_test)
print("R^2 adjusted:")
print(line.get_rsquared_adj(regressor,X_test,y_test))
mse = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mse}')
#regresija testiranje
data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==round)]
new_X = data_2023.drop(columns=['positionOrder','Top 3 Finish'])
#Prikazivanje koeficijenata atributa zajedno sa nazivima
print("Koeficijenti atributa:")
predicted_classes_2023 = regressor.predict(new_X)
atributi_nazivi = X_train.columns
for naziv, coef in zip(atributi_nazivi, regressor.coef_):
    print(f"Atribut '{naziv}': {coef}")
#SORTIRANJE FINALNIH POZICIJA
sorted_indices = np.argsort(predicted_classes_2023)
assigned_values = np.arange(1, len(sorted_indices) + 1)
mapping_dict = dict(zip(sorted_indices, assigned_values))
predicted_classes_assigned = np.vectorize(mapping_dict.get)(np.arange(len(predicted_classes_2023)))
# Prikaz rezultata
print("Originalni niz:")
print(predicted_classes_2023)
print("\nNiz sa dodeljenim vrednostima:")
print(predicted_classes_assigned)

accuracy = accuracy_score(df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==round)]['positionOrder'],predicted_classes_assigned)
print(f'Accuracy: {accuracy}')
mae = mean_absolute_error(df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==round)]['positionOrder'],predicted_classes_assigned)
print(f"Mean Absolute Error: {mae}")
min_index = np.argmin(predicted_classes_assigned)
print(drivers_df.where(drivers_df['driverId']==new_X.iloc[min_index]['driverId'])['surname'].dropna())
print(f'Predicted Classes for 2023: {predicted_classes_2023}')
print("----------------PREDIKCIJA Godine------------------")
poslednji_red = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year)].iloc[-1]
round_vrednost = poslednji_red['round']
count = int(round_vrednost)+1
result_df = pd.DataFrame({
    'surname': drivers_df['surname'],
    'points': 0 ,
    'driverId':drivers_df['driverId'],
    'real_points':0,
    'constructorId':0,
})
for i in range(len(new_X)):
    result_df.loc[result_df['driverId'] == new_X.iloc[i]['driverId'], 'constructorId'] = new_X.iloc[i]['constructorId']

right=0
right_podium_total=0
right_podium=0
for i in range(2,count):
    counter=0
    print("#########################################################")
    data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round'] == i)]
    new_X = data_2023.drop(columns=['positionOrder', 'Top 3 Finish'])
    predicted_classes_2023 = regressor.predict(new_X)
    sorted_indices = np.argsort(predicted_classes_2023)
    assigned_values = np.arange(1, len(sorted_indices) + 1)
    mapping_dict = dict(zip(sorted_indices, assigned_values))
    predicted_classes_assigned = np.vectorize(mapping_dict.get)(np.arange(len(predicted_classes_2023)))
    min_index = np.argmin(predicted_classes_assigned)
    winner_position = new_X.iloc[min_index]['driverId'], new_X.iloc[min_index]['grid'].item()
    print("Pozicija pobednika je: {}".format(winner_position))
    winner_surname = drivers_df.loc[drivers_df['driverId'] == new_X.iloc[min_index]['driverId'], 'surname'].item()
    print("Prediktovani POBEDNIK {} runde je: {}".format(i, winner_surname))
    winner_position = new_X.iloc[0]['driverId'], new_X.iloc[0]['grid'].item()
    print("Pozicija stvarnog pobednika je: {}".format(winner_position))
    winner_surname1 = drivers_df.loc[drivers_df['driverId'] == new_X.iloc[0]['driverId'], 'surname'].item()
    print("Stvarni pobednik {} runde je: {}".format(i, winner_surname1))    
    accuracy = accuracy_score(df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==i)]['positionOrder'],predicted_classes_assigned)
    print(f'ACCURACY: {accuracy:.2f}')
    mae = mean_absolute_error(df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==i)]['positionOrder'],predicted_classes_assigned)
    print(f"Mean Absolute Error: {mae:.2f}")
    rezkopija = predicted_classes_assigned.copy()
    for i in range(0,10):
        min_index = np.argmin(rezkopija)
        if (min_index in {0, 1, 2}) and i == 0:
            right_podium = right_podium + 1
        if (min_index in {0, 1, 2}) and i<3:
            counter=counter+1
        if counter==3 and i<3:
            right_podium_total=right_podium_total+1        
        result_df.loc[result_df['driverId'] == new_X.iloc[min_index]['driverId'], 'points'] += points[i]
        result_df.loc[result_df['driverId'] == new_X.iloc[i]['driverId'], 'real_points'] += points[i]
        rezkopija[min_index]=rezkopija[min_index]+20
    if winner_surname==winner_surname1:
        right=right+1
count2=count-2       
print("Procenat pogodjenih pobednika je: {:.2%}".format(right/count2))  
print("Procenat pogodjenih kompletnih podiuma je: {:.2%}".format(right_podium_total/count2))
print("Procenat pogodjenih da je prediktovani pobednik zavrsio na podiumu je: {:.2%}".format(right_podium/count2))   
result_df = result_df[(result_df['real_points'] != 0) | (result_df['points'] != 0)]   
result_df=result_df.sort_values(by='points', ascending=False)   
result_df.to_csv("Rezultati.csv",index=False)
saberi_po_constructorId = result_df.groupby('constructorId')['points'].sum().reset_index()
saberi_po_constructorId = saberi_po_constructorId.sort_values(by='points', ascending=False).reset_index(drop=True)
merged_df = saberi_po_constructorId.merge(constructor_df, on='constructorId', how='left')
print("PREDIKCIJA PORETKA ZA KONSTRUKTORE:")
print(merged_df[['constructorId', 'constructorRef', 'points']])
print("STVARNI POREDAK KONSTRUKTORA")
saberi_po_constructorId = result_df.groupby('constructorId')['real_points'].sum().reset_index()
saberi_po_constructorId = saberi_po_constructorId.sort_values(by='real_points', ascending=False).reset_index(drop=True)
merged_df = saberi_po_constructorId.merge(constructor_df, on='constructorId', how='left')
print(merged_df[['constructorId', 'constructorRef', 'real_points']])



#---------------------------------------------------------------------------------------------------------
print("--------------------------------RANDOM FOREST CLASSIFIER-------------------------------------------")
from sklearn.metrics import precision_score
classifier = RandomForestClassifier()
y=y.apply(lambda x: 1 if x == 1 else 2 if x==2 else 3 if x==3 else 4)
X_train1, X_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.2, random_state=42)
classifier.fit(X_train1, y_train1)

# Evaluacija modela
print("R^2 adjusted:")
print(line.get_rsquared_adj(classifier,X_test1,y_test1))
y_pred = classifier.predict(X_test1)
accuracy = accuracy_score(y_test1, y_pred)
print(f'Accuracy: {accuracy}')
df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
correct_predictions=0
for index, row in df.iterrows():
    if row['y_test'] == 1 and row['y_test'] == row['y_pred']:
        correct_predictions += 1
total_y_test_1 = len(df[df['y_test'] == 1])
accuracy_for_y_test_1 = correct_predictions / total_y_test_1 if total_y_test_1 > 0 else 0
print(f'Preciznost za y_test == 1: {__builtins__.round(accuracy_for_y_test_1*100,1)}%')
#Predikcija za klassifier

data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year  ) & (df_final_keepPositionOrder['round']==round )]
new_X = data_2023.drop(columns=['positionOrder','Top 3 Finish'])
# Predviđanje klasa za 2023. godinu
predicted_classes_2023 = classifier.predict_proba(new_X)

# Prikazivanje važnosti atributa
importances = classifier.feature_importances_
feature_names = X_train1.columns
for feature, importance in zip(feature_names, importances):
    print(f"{feature}: {importance}")

indeks_najveceg_reda = np.argmax(predicted_classes_2023[:, 0])
print("Prediktovani pobednik je:")
print(drivers_df.where(drivers_df['driverId']==new_X.iloc[indeks_najveceg_reda]['driverId'])['surname'].dropna())
print("Stvaran pobednik je:")
print(drivers_df.where(drivers_df['driverId']==new_X.iloc[0]['driverId'])['surname'].dropna())
print(f'Predicted Classes for 2023: {predicted_classes_2023}')
print("--------------------PREDIKCIJA Godine-----------------------")
right=0
right_podium_total=0
right_podium=0
for i in range(2, count):
    print("#########################################################")
    data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round'] == i)]
    new_X = data_2023.drop(columns=['positionOrder', 'Top 3 Finish'])
    predicted_classes_2023 = classifier.predict_proba(new_X)
    indeks_najveceg_reda = np.argmax(predicted_classes_2023[:, 0])
    indeks_najveceg_reda2 = np.argmax(predicted_classes_2023[:, 1])
    indeks_najveceg_reda3 = np.argmax(predicted_classes_2023[:, 2])
    winner_surname = drivers_df.loc[drivers_df['driverId'] == new_X.iloc[indeks_najveceg_reda]['driverId'], 'surname'].item()
    print("Prediktovani POBEDNIK {} runde je: {}".format(i, winner_surname))
    winner_surname1 = drivers_df.loc[drivers_df['driverId'] == new_X.iloc[0]['driverId'], 'surname'].item()
    print("Stvaran POBEDNIK {} runde je: {}".format(i, winner_surname1))
    if winner_surname==winner_surname1:
        right=right+1
    if (indeks_najveceg_reda in {0, 1, 2}) and (indeks_najveceg_reda2 in {0, 1, 2}) and (indeks_najveceg_reda3 in {0, 1, 2}):
        right_podium_total = right_podium_total + 1
    if indeks_najveceg_reda in {0, 1, 2}:
        right_podium = right_podium + 1
count2=count-2       
print("Procenat pogodjenih pobednika je: {:.2%}".format(right/count2))
print("Procenat pogodjenih kompletnih podiuma je: {:.2%}".format(right_podium_total/count2))
print("Procenat pogodjenih da je prediktovani pobednik zavrsio na podiumu je: {:.2%}".format(right_podium/count2))
#----------------------------------------------------------------------------------------------------------
print("-----------------------------------------RANDOM FOREST REGRESSOR-----------------------------------")
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor()
regressor.fit(X_train, y_train)
# 5. Evaluacija modela
print("R^2 adjusted:")
print(line.get_rsquared_adj(regressor,X_test,y_test))
y_pred = regressor.predict(X_test)
mse = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mse}')
#regresija testiranje
data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==round)]
new_X = data_2023.drop(columns=['positionOrder','Top 3 Finish'])
# Predviđanje klasa za 2023. godinu
predicted_classes_2023 = regressor.predict(new_X)
#Vaznost atributa
print("Feature importances:")
atributi_nazivi = X_train.columns
for naziv, importance in zip(atributi_nazivi, regressor.feature_importances_):
    print(f"Feature '{naziv}': {importance}")

#SORTIRANJE FINALNIH POZICIJA
sorted_indices = np.argsort(predicted_classes_2023)
assigned_values = np.arange(1, len(sorted_indices) + 1)
mapping_dict = dict(zip(sorted_indices, assigned_values))
predicted_classes_assigned = np.vectorize(mapping_dict.get)(np.arange(len(predicted_classes_2023)))
# Prikaz rezultata
print("Originalni niz:")
print(predicted_classes_2023)
print("\nNiz sa dodeljenim vrednostima:")
print(predicted_classes_assigned)

accuracy = accuracy_score(df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==round)]['positionOrder'],predicted_classes_assigned)
print(f'Accuracy: {accuracy}')
mae = mean_absolute_error(df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==round)]['positionOrder'],predicted_classes_assigned)
print(f"Mean Absolute Error: {mae}")
min_index = np.argmin(predicted_classes_assigned)
print(drivers_df.where(drivers_df['driverId']==new_X.iloc[min_index]['driverId'])['surname'].dropna())
print(f'Predicted Classes for 2023: {predicted_classes_2023}')

print("PREDIKCIJA GODINE")
poslednji_red = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year)].iloc[-1]
round_vrednost = poslednji_red['round']
count = int(round_vrednost)+1
resultforest_df = pd.DataFrame({
    'surname': drivers_df['surname'],
    'points': 0 ,
    'driverId':drivers_df['driverId'],
    'real_points':0,
    'constructorId':0,
})
for i in range(len(new_X)):
    resultforest_df.loc[resultforest_df['driverId'] == new_X.iloc[i]['driverId'], 'constructorId'] = new_X.iloc[i]['constructorId']
right=0
right_podium_total=0
right_podium=0
for i in range(2,count):
    counter=0
    print("#########################################################")
    data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round'] == i)]
    new_X = data_2023.drop(columns=['positionOrder', 'Top 3 Finish'])
    predicted_classes_2023 = regressor.predict(new_X)
    sorted_indices = np.argsort(predicted_classes_2023)
    assigned_values = np.arange(1, len(sorted_indices) + 1)
    mapping_dict = dict(zip(sorted_indices, assigned_values))
    predicted_classes_assigned = np.vectorize(mapping_dict.get)(np.arange(len(predicted_classes_2023)))
    min_index = np.argmin(predicted_classes_assigned)
    winner_position = new_X.iloc[min_index]['driverId'], new_X.iloc[min_index]['grid'].item()
    print("Pozicija pobednika je: {}".format(winner_position))
    winner_surname = drivers_df.loc[drivers_df['driverId'] == new_X.iloc[min_index]['driverId'], 'surname'].item()
    print("Prediktovani POBEDNIK {} runde je: {}".format(i, winner_surname))
    winner_position = new_X.iloc[0]['driverId'], new_X.iloc[0]['grid'].item()
    print("Pozicija stvarnog pobednika je: {}".format(winner_position))
    winner_surname1 = drivers_df.loc[drivers_df['driverId'] == new_X.iloc[0]['driverId'], 'surname'].item()
    print("Stvarni pobednik {} runde je: {}".format(i, winner_surname1))    
    accuracy = accuracy_score(df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==i)]['positionOrder'],predicted_classes_assigned)
    print(f'ACCURACY: {accuracy:.2f}')
    mae = mean_absolute_error(df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==i)]['positionOrder'],predicted_classes_assigned)
    print(f"Mean Absolute Error: {mae:.2f}")
    rezkopija = predicted_classes_assigned.copy()
    for i in range(0,10):
        min_index = np.argmin(rezkopija)
        if (min_index in {0, 1, 2}) and i == 0:
            right_podium = right_podium + 1
        if (min_index in {0, 1, 2}) and i<3:
            counter=counter+1
        if counter==3 and i<3:
            right_podium_total=right_podium_total+1        
        resultforest_df.loc[resultforest_df['driverId'] == new_X.iloc[min_index]['driverId'], 'points'] += points[i]
        resultforest_df.loc[resultforest_df['driverId'] == new_X.iloc[i]['driverId'], 'real_points'] += points[i]
        rezkopija[min_index]=rezkopija[min_index]+20
    if winner_surname==winner_surname1:
        right=right+1
count2=count-2       
print("Procenat pogodjenih pobednika je: {:.2%}".format(right/count2))  
print("Procenat pogodjenih kompletnih podiuma je: {:.2%}".format(right_podium_total/count2))
print("Procenat pogodjenih da je prediktovani pobednik zavrsio na podiumu je: {:.2%}".format(right_podium/count2))   
resultforest_df = resultforest_df[(resultforest_df['real_points'] != 0) | (resultforest_df['points'] != 0)]   
resultforest_df=resultforest_df.sort_values(by='points', ascending=False)   
resultforest_df.to_csv("RezultatiForest.csv",index=False)
saberi_po_constructorId = resultforest_df.groupby('constructorId')['points'].sum().reset_index()
saberi_po_constructorId = saberi_po_constructorId.sort_values(by='points', ascending=False).reset_index(drop=True)
merged_df = saberi_po_constructorId.merge(constructor_df, on='constructorId', how='left')
print("PREDIKCIJA PORETKA ZA KONSTRUKTORE:")
print(merged_df[['constructorId', 'constructorRef', 'points']])
print("STVARNI POREDAK KONSTRUKTORA")
saberi_po_constructorId = result_df.groupby('constructorId')['real_points'].sum().reset_index()
saberi_po_constructorId = saberi_po_constructorId.sort_values(by='real_points', ascending=False).reset_index(drop=True)
merged_df = saberi_po_constructorId.merge(constructor_df, on='constructorId', how='left')
print(merged_df[['constructorId', 'constructorRef', 'real_points']])
#-------------------------------------------------------------------------------------------------------------
print("----------------------------- LogisticRegression-------------------------------------------")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
y1 = filtered_df["positionOrder"].apply(lambda x: 1 if x == 1 else 0)
x =filtered_df.drop(columns=["positionOrder","Top 3 Finish"])


# Podela podataka na trening i test set
X_train4, X_test4, y_train4, y_test4 = train_test_split(x, y1, test_size=0.2, random_state=42)
model = LogisticRegression()
# Treniranje modela
model.fit(X_train4, y_train4)
y_pred = model.predict(X_test4)
labels_proba = model.predict_proba(X_test4)[:, 1]
y_pred_proba=model.predict_proba(X_test4)
# Evaluacija performansi modela
print(y_pred_proba)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

correct_predictions=0
for i in range(len(y_pred)):
    if y_pred[i] == 1 and np.argmax(y_pred_proba[i])==1:
        correct_predictions += 1
total_y_test_1 = len(y_test[y_test == 1])
accuracy_for_y_test_1 = correct_predictions / total_y_test_1 if total_y_test_1 > 0 else 0
print(f'Preciznost za y_test == 1: {__builtins__.round(accuracy_for_y_test_1 * 100, 1)}%')

accuracy = accuracy_score(y_test4, y_pred)
report = classification_report(y_test4, y_pred)
matrix = confusion_matrix(y_test4, y_pred)

fpr, tpr, thresholds = roc_curve(y_test4, labels_proba)
roc_auc = auc(fpr, tpr)
print(f'Accuracy: {accuracy}')
print('Classification Report:\n', report)
print('Confusion Matrix:\n', matrix)
print(f'AUC-ROC: {roc_auc}')

# Prikazivanje AUC-ROC krive
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


#regresija testiranje
data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==round)]
new_X = data_2023.drop(columns=['positionOrder','Top 3 Finish'])
# Predviđanje klasa za 2023. godinu
predicted_classes_2023 = model.predict_proba(new_X)
indeks_najveceg_reda = np.argmax(predicted_classes_2023[:, 1])
print("Prediktovani pobednik je:")
print(drivers_df.where(drivers_df['driverId']==new_X.iloc[indeks_najveceg_reda]['driverId'])['surname'].dropna())
print("Stvaran pobednik je:")
print(drivers_df.where(drivers_df['driverId']==new_X.iloc[0]['driverId'])['surname'].dropna())
print("PREDIKCIJA Godine")
print(f'Predicted Classes for 2023: {predicted_classes_2023}')
right=0
for i in range(2, count):
    print("#########################################################")
    data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round'] == i)]
    new_X = data_2023.drop(columns=['positionOrder', 'Top 3 Finish'])
    predicted_classes_2023 = model.predict_proba(new_X)
    indeks_najveceg_reda = np.argmax(predicted_classes_2023[:, 1])
    winner_surname = drivers_df.loc[drivers_df['driverId'] == new_X.iloc[indeks_najveceg_reda]['driverId'], 'surname'].item()
    print("Prediktovani POBEDNIK {} runde je: {}".format(i, winner_surname))
    winner_surname1 = drivers_df.loc[drivers_df['driverId'] == new_X.iloc[0]['driverId'], 'surname'].item()
    print("Stvaran POBEDNIK {} runde je: {}".format(i, winner_surname1))
    if winner_surname==winner_surname1:
        right=right+1  
count2=count-2       
print("Procenat pogodjenih pobednika je: {:.2%}".format(right/count2))
#-------------------------------------------------------------------------------------------------------------------
print("---------------------------LASSO-----------------------------------------------------")
from sklearn.model_selection import cross_val_score
alphas = [0.01, 0.05, 0.1, 0.15, 1.0, 5.0, 10.0]

lasso_regressor = Lasso(alpha=0.5)
lasso_regressor.fit(X_train, y_train)
print("R^2 adjusted:")
print(line.get_rsquared_adj(lasso_regressor,X_test,y_test))
y_pred = lasso_regressor.predict(X_test)
mse = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mse}')
#regresija testiranje
data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==round)]
new_X = data_2023.drop(columns=['positionOrder','Top 3 Finish'])
# Predviđanje klasa za 2023. godinu
predicted_classes_2023 = lasso_regressor.predict(new_X)
min_index = np.argmin(predicted_classes_2023)
print(drivers_df.where(drivers_df['driverId']==new_X.iloc[min_index]['driverId'])['surname'].dropna())
print(f'Predicted Classes for 2023: {predicted_classes_2023}')
print("PREDIKCIJA GODINE")
poslednji_red = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year)].iloc[-1]
round_vrednost = poslednji_red['round']
count = int(round_vrednost)+1
result_df = pd.DataFrame({
    'surname': drivers_df['surname'],
    'points': 0 ,
    'driverId':drivers_df['driverId'],
    'real_points':0,
    'constructorId':0,
})
for i in range(len(new_X)):
    result_df.loc[result_df['driverId'] == new_X.iloc[i]['driverId'], 'constructorId'] = new_X.iloc[i]['constructorId']

right=0
right_podium_total=0
right_podium=0

for i in range(2,count):
    counter=0
    print("#########################################################")
    data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round'] == i)]
    new_X = data_2023.drop(columns=['positionOrder', 'Top 3 Finish'])
    predicted_classes_2023 = lasso_regressor.predict(new_X)
    sorted_indices = np.argsort(predicted_classes_2023)
    assigned_values = np.arange(1, len(sorted_indices) + 1)
    mapping_dict = dict(zip(sorted_indices, assigned_values))
    predicted_classes_assigned = np.vectorize(mapping_dict.get)(np.arange(len(predicted_classes_2023)))
    min_index = np.argmin(predicted_classes_assigned)
    winner_position = new_X.iloc[min_index]['driverId'], new_X.iloc[min_index]['grid'].item()
    print("Pozicija pobednika je: {}".format(winner_position))
    winner_surname = drivers_df.loc[drivers_df['driverId'] == new_X.iloc[min_index]['driverId'], 'surname'].item()
    print("Prediktovani POBEDNIK {} runde je: {}".format(i, winner_surname))
    winner_position = new_X.iloc[0]['driverId'], new_X.iloc[0]['grid'].item()
    print("Pozicija stvarnog pobednika je: {}".format(winner_position))
    winner_surname1 = drivers_df.loc[drivers_df['driverId'] == new_X.iloc[0]['driverId'], 'surname'].item()
    print("Stvarni pobednik {} runde je: {}".format(i, winner_surname1))    
    accuracy = accuracy_score(df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==i)]['positionOrder'],predicted_classes_assigned)
    print(f'ACCURACY: {accuracy:.2f}')
    mae = mean_absolute_error(df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==i)]['positionOrder'],predicted_classes_assigned)
    print(f"Mean Absolute Error: {mae:.2f}")
    rezkopija = predicted_classes_assigned.copy()
    for i in range(0,10):
        min_index = np.argmin(rezkopija)
        if (min_index in {0, 1, 2}) and i == 0:
            right_podium = right_podium + 1
        if (min_index in {0, 1, 2}) and i<3:
            counter=counter+1
        if counter==3 and i<3:
            right_podium_total=right_podium_total+1        
        result_df.loc[result_df['driverId'] == new_X.iloc[min_index]['driverId'], 'points'] += points[i]
        result_df.loc[result_df['driverId'] == new_X.iloc[i]['driverId'], 'real_points'] += points[i]
        rezkopija[min_index]=rezkopija[min_index]+20
    if winner_surname==winner_surname1:
        right=right+1
count2=count-2       
print("Procenat pogodjenih pobednika je: {:.2%}".format(right/count2))  
print("Procenat pogodjenih kompletnih podiuma je: {:.2%}".format(right_podium_total/count2))
print("Procenat pogodjenih da je prediktovani pobednik zavrsio na podiumu je: {:.2%}".format(right_podium/count2))   
result_df = result_df[(result_df['real_points'] != 0) | (result_df['points'] != 0)]   
result_df=result_df.sort_values(by='points', ascending=False)   
result_df.to_csv("RezultatiLasso.csv",index=False)
saberi_po_constructorId = result_df.groupby('constructorId')['points'].sum().reset_index()
saberi_po_constructorId = saberi_po_constructorId.sort_values(by='points', ascending=False).reset_index(drop=True)
merged_df = saberi_po_constructorId.merge(constructor_df, on='constructorId', how='left')
print("PREDIKCIJA PORETKA ZA KONSTRUKTORE:")
print(merged_df[['constructorId', 'constructorRef', 'points']])
print("STVARNI POREDAK KONSTRUKTORA")
saberi_po_constructorId = result_df.groupby('constructorId')['real_points'].sum().reset_index()
saberi_po_constructorId = saberi_po_constructorId.sort_values(by='real_points', ascending=False).reset_index(drop=True)
merged_df = saberi_po_constructorId.merge(constructor_df, on='constructorId', how='left')
print(merged_df[['constructorId', 'constructorRef', 'real_points']])
#-----------------------------------------------------------------------------------------------------------------------------------------
print("--------------------------------L2 RIDGE------------------------------------------------------------------------")
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

ridge_regressor = Ridge(alpha=3)  
ridge_regressor.fit(X_train, y_train)
print("R^2 adjusted:")
print(line.get_rsquared_adj(ridge_regressor,X_test,y_test))
y_pred = ridge_regressor.predict(X_test)

mse = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mse}')

# Koeficijenti sa Ridge regresijom
coefficients = ridge_regressor.coef_
print("Koeficijenti sa Ridge regresijom:")
print(coefficients)

#regresija testiranje
data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==round)]
new_X = data_2023.drop(columns=['positionOrder','Top 3 Finish'])
new_X.to_csv("newX.csv",index=False)
# Predviđanje klasa za 2023. godinu
predicted_classes_2023 = ridge_regressor.predict(new_X)
min_index = np.argmin(predicted_classes_2023)
print(drivers_df.where(drivers_df['driverId']==new_X.iloc[min_index]['driverId'])['surname'].dropna())
print(f'Predicted Classes for 2023: {predicted_classes_2023}')
print("PREDIKCIJA GODINE")
right=0
for i in range(2, count):
    print("#########################################################")
    data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round'] == i)]
    new_X = data_2023.drop(columns=['positionOrder', 'Top 3 Finish'])
    predicted_classes_2023 = ridge_regressor.predict(new_X)
    min_index = np.argmin(predicted_classes_2023)
    winner_position = new_X.iloc[min_index]['driverId'], new_X.iloc[min_index]['grid'].item()
    print("Pozicija pobednika je: {}".format(winner_position))
    winner_surname = drivers_df.loc[drivers_df['driverId'] == new_X.iloc[min_index]['driverId'], 'surname'].item()
    print("Prediktovani POBEDNIK {} runde je: {}".format(i, winner_surname))
    winner_position = new_X.iloc[0]['driverId'], new_X.iloc[0]['grid'].item()
    print("Pozicija stvarnog pobednika je: {}".format(winner_position))
    winner_surname1 = drivers_df.loc[drivers_df['driverId'] == new_X.iloc[0]['driverId'], 'surname'].item()
    print("Stvarni pobednik {} runde je: {}".format(i, winner_surname1))
    if winner_surname==winner_surname1:
        right=right+1  
count2=count-2       
print("Procenat pogodjenih pobednika je: {:.2%}".format(right/count2))    
#----------------------------------------------------------------------------------------------------------------------------------------------
print("-----------------------------------------ElasticNet--------------------------------------------------------------")
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


elastic_net_regressor = ElasticNet()

# Definisanje opsega vrednosti za pretragu
param_grid = {
    'alpha': [0.5],
    'l1_ratio': [0.9]
}

# Inicijalizacija GridSearchCV
grid_search = GridSearchCV(estimator=elastic_net_regressor, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=5)

# Pokretanje pretrage
grid_search.fit(X_train, y_train)

# Najbolji parametri
best_params = grid_search.best_params_
print("Najbolji parametri:", best_params)

best_elastic_net = grid_search.best_estimator_
print("R^2 adjusted:")
print(line.get_rsquared_adj(best_elastic_net,X_test,y_test))
# Predviđanje na test setu
y_pred = best_elastic_net.predict(X_test)
# Evaluacija modela
mse = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mse}')

data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year  ) & (df_final_keepPositionOrder['round']==round )]
new_X = data_2023.drop(columns=['positionOrder','Top 3 Finish'])
# Predviđanje klasa za 2023. godinu
predicted_classes_2023 = best_elastic_net.predict(new_X)
indeks_najveceg_reda = np.argmin(predicted_classes_2023)
print("Prediktovani pobednik je:")
print(drivers_df.where(drivers_df['driverId']==new_X.iloc[indeks_najveceg_reda]['driverId'])['surname'].dropna())
print("Stvaran pobednik je:")
print(drivers_df.where(drivers_df['driverId']==new_X.iloc[0]['driverId'])['surname'].dropna())
print(f'Predicted Classes for 2023: {predicted_classes_2023}')
print("--------------------PREDIKCIJA Godine-----------------------")
right=0
for i in range(2, count):
    print("#########################################################")
    data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round'] == i)]
    new_X = data_2023.drop(columns=['positionOrder', 'Top 3 Finish'])
    predicted_classes_2023 = best_elastic_net.predict(new_X)
    indeks_najveceg_reda = np.argmin(predicted_classes_2023)
    winner_surname = drivers_df.loc[drivers_df['driverId'] == new_X.iloc[indeks_najveceg_reda]['driverId'], 'surname'].item()
    print("Prediktovani POBEDNIK {} runde je: {}".format(i, winner_surname))
    winner_surname1 = drivers_df.loc[drivers_df['driverId'] == new_X.iloc[0]['driverId'], 'surname'].item()
    print("Stvaran POBEDNIK {} runde je: {}".format(i, winner_surname1))
    if winner_surname==winner_surname1:
        right=right+1
count2=count-2       
print("Procenat pogodjenih pobednika je: {:.2%}".format(right/count2))

#-------------------------------------------------------------------------------------------------------------------------
print("---------------------------Promena guma predikcija------------------------------")
print("------------------------Linearna Regresija---------------------------------------")
'''
import fastf1
season_df=fastf1.get_session(2021,2)
season_df.load(weather=True)
print(season_df.weather_data)
'''
year=2019
round=10
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
aggregations = {'stop': 'sum', 'weather_warm': 'first','weather_cold':'first', 'weather_dry': 'first','weather_wet':'first','weather_cloudy':'first'}
aggregations2 = {'Group_Count': 'first', 'circuitId': 'first'}
aggregations3 = {'stop': 'sum', 'weather_warm': 'first','weather_cold':'first', 'weather_dry': 'first','weather_wet':'first','weather_cloudy':'first','Group_Count':'first','status_1_count':'first','AVG pitstop':'first'}

pit_with_year = pitstops_df.merge(races_df[['raceId', 'year','round','circuitId']], on='raceId', how='left')
pit_with_year.to_csv("PIT.csv",index=False)
weather_df=weather_df.drop(columns=['circuit_id','weather'])
weather_df = weather_df.rename(columns={'season': 'year'})
weather_df_final = weather_df.merge(pit_with_year[[ 'year','round','stop','driverId','circuitId']], on=['year','round'], how='left')
weather_df_final.dropna(inplace=True)
weather_df_final = weather_df_final.drop_duplicates(subset=['year', 'round', 'driverId'], keep='last')
weather_df_final = weather_df_final.groupby(['year','circuitId','round']).agg(aggregations).reset_index() 
safety_cars_df['Group_Count'] = safety_cars_df.groupby('Race')['Race'].transform('size')
safety_cars_df_final = safety_cars_df.drop_duplicates(subset='Race')
safety_cars_df_final=safety_cars_df_final.drop(columns=['Cause','Deployed','Retreated','FullLaps'])
safety_cars_df_final[['Year', 'Race_Name']] = safety_cars_df['Race'].str.extract(r'(\d{4}) (.+)$')
safety_cars_df_final=safety_cars_df_final.drop(columns=['Race'])
safety_cars_df_final = safety_cars_df_final.rename(columns={'Race_Name': 'name'})
safety_cars_df_final = safety_cars_df_final.rename(columns={'Year': 'year'})
safety_cars_df_final['year'] = safety_cars_df_final['year'].astype(int)
races_df['year'] = races_df['year'].astype(int)
safety_cars_df_final = pd.merge(safety_cars_df_final, races_df[['name','year','circuitId']], on=['name','year'], how='left')
safety_cars_df_final = safety_cars_df_final.groupby(['year','name']).agg(aggregations2).reset_index()
safety_cars_df_final = safety_cars_df_final[(safety_cars_df_final['year'] > 2010) & (safety_cars_df_final['year'] < 2020)]
weather_df_final = pd.merge(weather_df_final, safety_cars_df_final[['year', 'circuitId','Group_Count']], on=['year', 'circuitId'], how='left')
weather_df_final = weather_df_final.fillna(0)
df_final_keepPositionOrder['statusId'] = df_final_keepPositionOrder['statusId'].apply(
    lambda x: 1 if x==1 else 1 if x==2 else 2
)
df_final_keepPositionOrder['status_1_count'] = df_final_keepPositionOrder.groupby(['year', 'round'])['statusId'].transform(lambda x: (x == 1).sum())
df_final_keepPositionOrder_first = df_final_keepPositionOrder.drop_duplicates(['year', 'round'], keep='first')
df_final_keepPositionOrder_first.to_csv("NOVOPROBA.csv",index=False)
weather_df_final['Group_Count']=weather_df_final['Group_Count'].apply(lambda x: 0 if x == 0 else 0 if x==1 else x)
weather_df_final = pd.merge(weather_df_final, df_final_keepPositionOrder_first[['year', 'round','status_1_count']], on=['year', 'round'], how='left')
mean_value_column = weather_df_final['status_1_count'].mean()
weather_df_final['status_1_count'] = weather_df_final['status_1_count'].fillna(mean_value_column)
weather_df_final=weather_df_final.sort_values(by=['year','round'], ascending=True)
weather_df_final['AVG pitstop']=0
print("Dosao")
for i in weather_df_final.index:
    ukupno = 0
    count = 0
    g = i - 1
    circuitId = weather_df_final.at[i, 'circuitId']
    while g >= 0:
        if weather_df_final.at[g, 'circuitId'] == circuitId:
            count += 1
            ukupno += weather_df_final.at[g, 'stop']
        g -= 1
    if count > 0:
        prosek = ukupno / count
        weather_df_final.at[i, 'AVG pitstop'] = int(prosek)
weather_df_final_test = weather_df_final[weather_df_final['year'] < year]
weather_df_final_test = weather_df_final_test.groupby(['year','circuitId','round']).agg(aggregations3).reset_index()
weather_df_final_test=weather_df_final_test.sort_values(by=['year','round'], ascending=True)
safety_cars_df_final.to_csv("Davidimo.csv",index=False)
weather_df_final.to_csv("WET.csv",index=False)
weather_df_final_test.to_csv("WETest.csv",index=False)
y = weather_df_final_test["stop"]
x = weather_df_final_test.drop(columns=["stop"])
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Absolute Error: {mae:.2f}')


data = weather_df_final[(weather_df_final['year'] == year) & (weather_df_final['round']==round)]
new_X = data.drop(columns=['stop'])
print("Koeficijenti atributa:")
predicted_classes_2023 = regressor.predict(new_X)
atributi_nazivi = X_train.columns 
for naziv, coef in zip(atributi_nazivi, regressor.coef_):
    print(f"Atribut '{naziv}': {coef}")

print("Originalni niz:")
print(predicted_classes_2023)
sum_stop_actual = weather_df_final[(weather_df_final['year'] == year) & (weather_df_final['round'] == round)]['stop'].reset_index(drop=True)
sum_stop_predicted = predicted_classes_2023[0]
print(f"Prediktovan broj pitstopova za rundu {round} je: {sum_stop_predicted}")
print(f"Stvaran broj pitstopova za rundu {round} je: {sum_stop_actual}")

poslednji_red = weather_df_final[(weather_df_final['year'] == year)].iloc[-1]
round_vrednost = poslednji_red['round']
count = int(round_vrednost)+1
prosek=0
oduzmioutliere=0
for i in range(1,count):
    print("#########################################################")
    data_2023 = weather_df_final[(weather_df_final['year'] == year) & (weather_df_final['round'] == i)]
    new_X = data_2023.drop(columns=['stop',])  
    predicted_classes_2023 = regressor.predict(new_X)
    sum_stop_actual = weather_df_final[(weather_df_final['year'] == year) & (weather_df_final['round'] == i)]['stop'].reset_index(drop=True)
    sum_stop_predicted = predicted_classes_2023[0]
    print(f"Prediktovan broj pitstopova za rundu {i} je: {sum_stop_predicted}")
    print(f"Stvaran broj pitstopova za rundu {i} je: {sum_stop_actual}")
    if any((sum_stop_actual > sum_stop_predicted) & (sum_stop_predicted / sum_stop_actual < 0.5)):
        oduzmioutliere = oduzmioutliere + 1
    else:
        prosek = prosek + abs((int(sum_stop_predicted) - sum_stop_actual))
print("PROSECNA GRESKA BRE OUTLIERA odnosno ukoliko je predikcija upola ili vise od pola manja od stvarne")
print(f"PROSECNO GRESI ZA {__builtins__.round(prosek/(count-1-oduzmioutliere),1)} PO TRCI")
#-------------------------------------------------------------------------------------------------------------------------
print("------------------------------GRAFICKI PRIKAZ FATALNIH INCIDENATA----------------------------------")
fatal_df=pd.read_csv('fatal_accidents_drivers.csv')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('fatal_accidents_drivers.csv')
# Konvertuj 'Date Of Accident' u tip datuma
df['Date Of Accident'] = pd.to_datetime(df['Date Of Accident'], format='%m/%d/%Y')

# Dodaj kolonu 'Godina'
df['Godina'] = df['Date Of Accident'].dt.year

# Dodaj kolonu 'Decenija'
df['Decenija'] = (df['Godina'] // 10) * 10

# Grupiši po godinama i broj incidenata
godisnji_incidenti = df.groupby('Godina').size().reset_index(name='Broj incidenata')

# Grupiši po decenijama i broj incidenata
decenijski_incidenti = df.groupby('Decenija').size().reset_index(name='Broj incidenata')

# Filtriraj decenije pre 1950. godine
decenijski_incidenti = decenijski_incidenti[decenijski_incidenti['Decenija'] >= 1950]

# Ako želite uključiti i 2020. godinu, dodajte je u DataFrame
godisnji_incidenti = godisnji_incidenti.set_index('Godina').reindex(range(godisnji_incidenti['Godina'].min(), 2021)).reset_index()

# Ako želite uključiti i 2020. godinu, dodajte je u DataFrame
decenijski_incidenti = decenijski_incidenti.set_index('Decenija').reindex(range(decenijski_incidenti['Decenija'].min(), 2021, 10)).reset_index()

# Vizualizuj trend godišnjih incidenata
plt.figure(figsize=(12, 6))
sns.barplot(x='Godina', y='Broj incidenata', data=godisnji_incidenti, palette='viridis')
plt.title('Godišnji trend incidenata na stazi')
plt.xlabel('Godina')
plt.ylabel('Broj incidenata')
plt.show()

# Vizualizuj trend decenijskih incidenata
plt.figure(figsize=(12, 6))
sns.barplot(x='Decenija', y='Broj incidenata', data=decenijski_incidenti, palette='viridis')
plt.title('Trend incidenata na stazi po decenijama')
plt.xlabel('Decenija')
plt.ylabel('Broj incidenata')
plt.show()
#-------------------------------------------------------------------------------------------------------------------------
print("------------------------Vremenska serija---------------------------------------")

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
#weather_df_final_test
weather_df_final_test = weather_df_final_test.drop(columns=['round'])
weather_df_final_test['year'] = pd.to_datetime(weather_df_final_test['year'], format='%Y')
weather_df_final_test = weather_df_final_test.set_index('year')
aggregations = {'stop': 'sum', 'weather_warm': 'first','weather_cold':'first', 'weather_dry': 'first','weather_wet':'first','weather_cloudy':'first'}
weather_df_final_test = weather_df_final_test.groupby(['year','circuitId']).agg(aggregations).reset_index()
weather_df_final_test.to_csv("weter.csv",index=False)
#KORELOGRAM
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(weather_df_final_test['stop'], lags=30)
plt.show()
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(weather_df_final_test['stop'], lags=30)
plt.show()

# Vizualizacija vremenske serije
plt.figure(figsize=(12, 6))
plt.plot(weather_df_final_test['stop'])
plt.title('Vremenska serija broja pitstopova')
plt.xlabel('Datum')
plt.ylabel('Broj pitstopova')
plt.show()

# Dekompozicija vremenske serije
result = seasonal_decompose(weather_df_final_test['stop'], model='additive', period=30)
result.plot()
plt.show()

# Podela podataka na trening i test skup
train_size = int(len(weather_df_final_test) * 0.8)
train, test = weather_df_final_test[0:train_size], weather_df_final_test[train_size:]

train['log10(stop)'] = np.log10(train['stop'])
train['stationary_data'] = train['log10(stop)'].diff()

p_value = adfuller(train['stationary_data'].dropna())[1]
if p_value <= 0.05: print('postoji stacionarnost')
else: print('ne postoji stacionarnost')

plot_pacf(train['stationary_data'].dropna(), lags=30, method='ywm')
plt.show()
p, d, q = 17, 1, 0
ar_model = ARIMA(train['log10(stop)'], order=(p, d, 0)).fit()
print(ar_model.summary())

y_train_pred = ar_model.predict(start=train.index[p+1], end=train.index[-1])

plt.plot(train['log10(stop)'], color='b', linewidth=4, alpha=0.3, label='train')
plt.plot(y_train_pred, color='darkorange', label='AR model prediction')
plt.title('predikcije za log10(stop)')
plt.legend()
plt.show()


test['log10(stop)'] = np.log10(test['stop'])
y_val_pred = ar_model.predict(start=test.index[0], end=test.index[-1])

plt.plot(train['log10(stop)'], color='b', linewidth=4, alpha=0.3, label='train')
plt.plot(test['log10(stop)'], color='mediumblue', linewidth=4, alpha=0.3, label='val')

plt.plot(y_val_pred, color='darkorange', label='AR model prediction')
plt.title('predikcije za log10(stop)')
plt.legend()
plt.show()

plot_acf(train['stationary_data'].dropna(), lags=20)
plt.show()

p, d, q = 17, 1, 2
arima_model = ARIMA(train['log10(stop)'], order=(p, d, q)).fit()

y_pred_arima = arima_model.predict(start=train.index[p+1], end=test.index[-1])
y_pred_arima = np.power(10, y_pred_arima)

plt.plot(train['stop'], color='b', linewidth=4, alpha=0.3, label='train')
plt.plot(test['stop'], color='mediumblue', linewidth=4, alpha=0.3, label='val')
plt.plot(y_pred, color='darkorange', label='AR model prediction')
plt.plot(y_pred_arima, color='k', label='ARIMA model prediction')
plt.title('predikcije za PitStop')
plt.legend()
plt.show()

'''
'''
# Modeliranje vremenske serije (primer sa Exponential Smoothing)
model = ExponentialSmoothing(train['stop'], trend='add', seasonal='add', seasonal_periods=7)
fit_model = model.fit()

# Predviđanje na test skupu
predictions = fit_model.forecast(steps=len(test))
mae = mean_absolute_error(test['stop'], predictions)
print(mae)

# Vizualizacija rezultata
plt.plot(train.index, train['stop'], label='Trening podaci')
plt.plot(test.index, test['stop'], label='Stvarni podaci')
plt.plot(test.index, predictions, label='Predviđeni podaci')
plt.legend()
plt.show()

for i in range(1,count):
    print("#########################################################")
    data_2023 = weather_df_final[(weather_df_final['year'] == year) & (weather_df_final['round'] == i)]
    new_X = data_2023.drop(columns=['stop'])   
    predicted_classes_2023 = fit_model.forecast(steps=len(new_X))
    sum_stop_actual = weather_df_final[(weather_df_final['year'] == year) & (weather_df_final['round'] == i)]['stop'].sum()
    sum_stop_predicted = predicted_classes_2023.sum()
    print(f"Prediktovan broj pitstopova za rundu {i} je: {sum_stop_predicted}")
    print(f"Stvaran broj pitstopova za rundu {i} je: {sum_stop_actual}")
    prosek=prosek+abs((int(sum_stop_predicted)-sum_stop_actual))
    mae_round = mean_absolute_error(weather_df_final[(weather_df_final['year'] == year) & (weather_df_final['round'] == i)]['stop'], predictions)
    print(f"Mean Absolute Error za rundu {i}: {mae_round}")

print(f"PROSECNO GRESI ZA {__builtins__.round(prosek/(count-1),1)} PO TRCI")

'''
