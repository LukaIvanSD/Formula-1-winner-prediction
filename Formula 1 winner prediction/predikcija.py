import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import utils_nans1 as line

#Ucitavanje csv-ova
result_df = pd.read_csv('results.csv')
stats_df = pd.read_csv('status.csv')
drivers_df = pd.read_csv('drivers.csv')
races_df = pd.read_csv('races.csv')
constructor_df = pd.read_csv('constructors.csv')
driver_standings_df = pd.read_csv('driver_standings.csv')
constructor_standings_df = pd.read_csv('constructor_standings.csv')
qualifying_df=pd.read_csv('qualifying.csv')
pitstops_df=pd.read_csv('pit_stops.csv')
weather_df=pd.read_csv('weather.csv')
safety_cars_df=pd.read_csv('safety_cars.csv')
broj_ponavljanja = weather_df['circuit_id'].value_counts()
f1driversds_df=pd.read_csv('F1DriversDataset.csv')
constructor_standings_df = pd.merge(constructor_standings_df, races_df[['raceId', 'year']], on='raceId', how='left')
constructor_standings_df=constructor_standings_df[constructor_standings_df['year']>=2000]
constructor_standings_df = constructor_standings_df.reset_index(drop=True)
constructor_standings_df = constructor_standings_df.loc[constructor_standings_df.groupby('year')['points'].idxmax()]
race_df = races_df[["raceId", "year", "round", "circuitId"]].copy()
#Predprocesiranje
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


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#TOP 3 za vozace prosla godina
df['Top 3 Finish'] = df['positionOrder'].le(3).astype(int)
#print(df.head())

#Racunanje ukupnog broja trka i top 3 zavrsetka za svakog vozaca
driver_yearly_stats = df.groupby(['year', 'driverId']).agg(
    Total_Races=('raceId', 'nunique'),
    Top_3_Finishes=('Top 3 Finish', 'sum')
).reset_index()

#Procenat top 3 za svakog vozaca
driver_yearly_stats['Driver Top 3 Finish Percentage (This Year)'] = (driver_yearly_stats['Top_3_Finishes'] / driver_yearly_stats['Total_Races']) * 100
driver_last_year_stats = driver_yearly_stats.copy()
driver_last_year_stats['year'] += 1
driver_last_year_stats = driver_last_year_stats.rename(columns={'Driver Top 3 Finish Percentage (This Year)': 'Driver Top 3 Finish Percentage (Last Year)'})
df = pd.merge(df, driver_last_year_stats[['year', 'driverId', 'Driver Top 3 Finish Percentage (Last Year)']], on=['year', 'driverId'], how='left')


#TOP 3 za konstruktore prosla godina

#Racunanje srednje vrednosti procenta top 3 zavrsetka za oba u istom timu
constructor_last_year_stats = df.groupby(['year', 'constructorId', 'round']).agg(
    Sum_Top_3_Finishes_Last_Year=('Driver Top 3 Finish Percentage (Last Year)', 'sum')
).reset_index()

#Procenat top 3 zavrsetka za svaki tim prosle godine
constructor_last_year_stats['Constructor Top 3 Finish Percentage (Last Year)'] = constructor_last_year_stats["Sum_Top_3_Finishes_Last_Year"]/2

df = pd.merge(df, constructor_last_year_stats[['year', 'constructorId', 'round', 'Constructor Top 3 Finish Percentage (Last Year)']], on=['year', 'constructorId', 'round'], how='left')


#Vozac top 3 ove godine

#Funkcija za racunanje procenta top 3 zavrsetka pre trenutne runde za vozace
def calculate_driver_top_3_percentage_before_round(row, df):
    previous_races = df[(df['year'] == row['year']) & (df['driverId'] == row['driverId']) & (df['round'] < row['round'])]
    if len(previous_races) == 0:
      return pd.NA

    total_races = previous_races['raceId'].nunique()
    top_3_finishes = previous_races['Top 3 Finish'].sum()

    return (top_3_finishes / total_races) * 100 if total_races > 0 else pd.NA

df['Driver Top 3 Finish Percentage (This Year till last race)'] = df.apply(lambda row: calculate_driver_top_3_percentage_before_round(row, df), axis=1)



#Za konstruktore top 3 do sadasnje trke

#Racunanje srednje vrednosti procenta top 3 zavrsetka za oba vozaca u istom timu u tekucoj godini
constructor_this_year_stats = df.groupby(['year', 'constructorId', 'round']).agg(
    Sum_Top_3_Finishes_This_Year=('Driver Top 3 Finish Percentage (This Year till last race)', 'sum')
).reset_index()

#Racunanje procenta top 3 zavrsetka za svaki tim u tekucoj godini 
constructor_this_year_stats['Constructor Top 3 Finish Percentage (This Year till last race)'] = constructor_this_year_stats["Sum_Top_3_Finishes_This_Year"]/2

df = pd.merge(df, constructor_this_year_stats[['year', 'constructorId', 'round', 'Constructor Top 3 Finish Percentage (This Year till last race)']], on=['year', 'constructorId', 'round'], how='left')

#PROSECNA POZICIJA PROSLE GODINE za vozace

#Racunanje ukupnog broja trka i top 3 zavrsetka za svakog vozaca
driver_yearly_stats = df.groupby(['year', 'driverId']).agg(
    Total_Races=('raceId', 'nunique'),
    Avg_position=('positionOrder', 'mean')
).reset_index()

#Racunanje procenta top 3 zavrsetka za svakog vozaca
driver_yearly_stats['Driver Avg position (This Year)'] = driver_yearly_stats['Avg_position']
driver_last_year_stats = driver_yearly_stats.copy()
driver_last_year_stats['year'] += 1
driver_last_year_stats = driver_last_year_stats.rename(columns={'Driver Avg position (This Year)': 'Driver Avg position (Last Year)'})
df = pd.merge(df, driver_last_year_stats[['year', 'driverId', 'Driver Avg position (Last Year)']], on=['year', 'driverId'], how='left')


#za konstruktore prosecna pozicija prosle godine

#Racunanje srednje vrednosti procenta zavrsetka u top 3 za oba vozaca u istom timu prosle godine
constructor_last_year_stats = df.groupby(['year', 'constructorId', 'round']).agg(
    sum_position_last_year=('Driver Avg position (Last Year)', 'sum')
).reset_index()

#Racunanje procenta top 3 zavrsetka za tim za proslu godinu
constructor_last_year_stats['Constructor Avg position (Last Year)'] = constructor_last_year_stats["sum_position_last_year"]/2
df = pd.merge(df, constructor_last_year_stats[['year', 'constructorId', 'round', 'Constructor Avg position (Last Year)']], on=['year', 'constructorId', 'round'], how='left')

#Za vozace prosecna pozicija ove godine 

def calculate_driver_avg_position_before_round(row, df):
    previous_races = df[(df['year'] == row['year']) & (df['driverId'] == row['driverId']) & (df['round'] < row['round'])]
    if len(previous_races) == 0:
      return pd.NA
    total_races = previous_races['raceId'].nunique()
    positionSum = previous_races['positionOrder'].sum()

    return (positionSum / total_races) if total_races > 0 else pd.NA
df['Driver Average Position (This Year till last race)'] = df.apply(lambda row: calculate_driver_avg_position_before_round(row, df), axis=1)

#---------------------------------------------------------------------------

#Za timove prosecna pozicija ove godine
constructor_this_year_stats = df.groupby(['year', 'constructorId', 'round']).agg(
    sum_Position_Constructor = ('Driver Average Position (This Year till last race)', 'sum')
).reset_index()

constructor_this_year_stats['Constructor Average Position (This Year till last race)'] = constructor_this_year_stats["sum_Position_Constructor"]/2

df = pd.merge(df, constructor_this_year_stats[['year', 'constructorId', 'round', 'Constructor Average Position (This Year till last race)']], on=['year', 'constructorId', 'round'], how='left')

#-----------------------------------------------------------------
df2=pd.DataFrame(df)
df_final = df.drop(labels=["raceId"], axis=1)
print("Number of rows in total:", df_final.shape[0])
initial_count = len(df_final[df_final['year'] != 2000])

#NEDOSTAJUCE VREDNOSTI
df_final['Driver Top 3 Finish Percentage (Last Year)'] = df_final['Driver Top 3 Finish Percentage (Last Year)'].fillna(0.0)
df_final['Constructor Top 3 Finish Percentage (Last Year)'] = df_final['Constructor Top 3 Finish Percentage (Last Year)'].fillna(0.0)
df_final['Driver Avg position (Last Year)'] = df_final['Driver Avg position (Last Year)'].fillna(15.0)
df_final['Constructor Avg position (Last Year)'] = df_final['Constructor Avg position (Last Year)'].fillna(8.0)
# Provera da li imamo nedostajuce vrednosti
final_count = len(df_final[df_final['year'] != 2000])
df_final = df_final.dropna()
rows_dropped = initial_count - final_count
print("Broj redova izbacenih gde je godina veca od 2000:", rows_dropped)



df_final_keepPositionOrder = df_final.copy()
df_final = df_final.drop(["positionOrder"], axis = 1)


#---------------------------------------------------
#Priprema podataka
df_final["Driver Top 3 Finish Percentage (This Year till last race)"] = df_final["Driver Top 3 Finish Percentage (This Year till last race)"].astype(float)
df_final["Constructor Top 3 Finish Percentage (This Year till last race)"] = df_final["Constructor Top 3 Finish Percentage (This Year till last race)"].astype(float)
df_final["Driver Average Position (This Year till last race)"] = df_final["Driver Average Position (This Year till last race)"].astype(float)
df_final["Constructor Average Position (This Year till last race)"] = df_final["Constructor Average Position (This Year till last race)"].astype(float)
df_final_keepPositionOrder["Driver Top 3 Finish Percentage (This Year till last race)"] = df_final_keepPositionOrder["Driver Top 3 Finish Percentage (This Year till last race)"].astype(float)
df_final_keepPositionOrder["Constructor Top 3 Finish Percentage (This Year till last race)"] = df_final_keepPositionOrder["Constructor Top 3 Finish Percentage (This Year till last race)"].astype(float)
df_final_keepPositionOrder["Driver Average Position (This Year till last race)"] = df_final_keepPositionOrder["Driver Average Position (This Year till last race)"].astype(float)
df_final_keepPositionOrder["Constructor Average Position (This Year till last race)"] = df_final_keepPositionOrder["Constructor Average Position (This Year till last race)"].astype(float)

#Prosecna pozicija svake godine
avg_finish_per_year = df.groupby('year')['positionOrder'].mean()
avg_finish_per_year.plot(kind='line')
plt.title('Average Finish Position per Year')
plt.xlabel('Year')
plt.ylabel('Average Finish Position')
print(plt.show())
plt.figure(figsize=(10,7))
sns.heatmap(df_final_keepPositionOrder.corr(), annot=True, mask = False, annot_kws={"size": 7})
print(plt.show())

#Uticaj startne pozicije(grid) na krajnju poziciju(positionOrder)
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
#-------------------------------------------------------------------------------------
#unos godine
year = input("Unesite Godinu (min: 2001 max: 2023):")
year=int(year)
round=input("Unesite rundu: ")
round=int(round)

status1=input("Da li zelite sa status kolonom (y/n): ")
if status1=='y':
    status=1
    status2=input("Da li zelite sa opisom da li je zaostao za vise od 1 kruga (y/n): ")
    if status2=='y':
        status33=1
    else:
        status33=0
else:
    status=0

loop = input("Za koliko prethodnih godina zelite prosek pogodjenih: ")
loop=int(loop)
j=year
if year-loop>2000:
    year=year-loop
else:
    loop=year-2001
    year=2001

#DODAVANJE STATUSID KOLONE
if status==1:
    result_df=result_df.merge(races_df[['raceId', 'year','round']], on='raceId', how='left')
    result_df = result_df.sort_values(by=['year', 'round', 'position'], ascending=[True, True, False])
    df_final_keepPositionOrder = df_final_keepPositionOrder.merge(result_df[['year','round','driverId', 'statusId']], on=['year', 'round', 'driverId'], how='left')
    vrednosti_i_brojevi = df_final_keepPositionOrder['statusId'].value_counts(normalize=True)
    for vrednost, broj in vrednosti_i_brojevi.items():
        if broj>1: 
            print(f'Vrednost: {vrednost}, Broj pojavljivanja: {broj}')
    finished=np.array([1])  
    finishedlater=np.array([11,12,13])
    if status33==1:        
        df_final_keepPositionOrder['statusId'] = df_final_keepPositionOrder['statusId'].apply(
            lambda x: 1 if x in finished else 2 if x in finishedlater else 3
        )
    else:
        df_final_keepPositionOrder['statusId'] = df_final_keepPositionOrder['statusId'].apply(
            lambda x: 1 if x in finished else 1 if x in finishedlater else 2
        )


from ast import literal_eval
import json
import ast

#Dodavanje kolone Champ last year 1 je ukoliko je vozac prosle godine bio sampion i njegov tim isto u suprotnom 0
f1driversds_df['Driver'] = f1driversds_df['Driver'].astype(str)
f1driversds_df[['Ime', 'Prezime']] = f1driversds_df['Driver'].str.rsplit(' ', n=1, expand=True)
f1driversds_df_final = pd.merge(f1driversds_df[['Ime', 'Prezime', 'Championship Years']], drivers_df[['forename','surname', 'driverId']], left_on=['Ime','Prezime'], right_on=['forename','surname'], how='left')
f1driversds_df_final = f1driversds_df_final.fillna(pd.merge(f1driversds_df[['Ime', 'Prezime', 'Championship Years']], drivers_df[['surname', 'forename', 'driverId']], left_on=['Prezime', 'Ime'], right_on=['forename', 'surname'], how='left'))
f1driversds_df_final=f1driversds_df_final.dropna()
f1driversds_df_final.reset_index(drop=True, inplace=True)
constructor_standings_df.reset_index(drop=True, inplace=True)
df_final_keepPositionOrder['Champ Last Year']=0

for index, row in df_final_keepPositionOrder.iterrows():
    if (f1driversds_df_final['driverId'] == df_final_keepPositionOrder.at[index, 'driverId']).any():
        indeks_reda = f1driversds_df_final[f1driversds_df_final['driverId'] == df_final_keepPositionOrder.at[index, 'driverId']].index[0]
        if indeks_reda is not None:
            f1_years_list = f1driversds_df_final.iloc[indeks_reda]['Championship Years']
            lista_godina = ast.literal_eval(f1_years_list)
            if  df_final_keepPositionOrder.loc[index, 'year'] - 1 in lista_godina and (constructor_standings_df['year'] == df_final_keepPositionOrder.at[index, 'year']-1).any():
                indeks_reda2 = constructor_standings_df[constructor_standings_df['year'] == df_final_keepPositionOrder.at[index, 'year']-1].index[0]
                if indeks_reda is not None:
                    champ = constructor_standings_df.iloc[indeks_reda2]['constructorId']
                    if champ==df_final_keepPositionOrder.loc[index, 'constructorId']:
                        df_final_keepPositionOrder.at[index, 'Champ Last Year'] = 1


df_final_keepPositionOrder.to_csv("df_final_keepPositionOrder.csv",index=False)
    


#PRAVLJENJE X I Y PODATAKA

zbirprocenatareg=0
zbirprocenatarfc=0
zbirprocenatarfr=0
zbirprocenatalog=0
zbirprocenataros=0
zbirprocenatalasso=0
zbirprocenataridge=0
zbirprocenataelastic=0

while year<=j:
    print(j)
    print(year)
    filtered_df = df_final_keepPositionOrder[df_final_keepPositionOrder['year'] < year]
    y =  filtered_df["positionOrder"]
    x =filtered_df.drop(columns=["positionOrder","Top 3 Finish"])
    points = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1])
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error

    # Podela podataka na trening i test set
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    #---------------------------------------------------------------------------------------------------
    print("---------------------------REGRESIJA----------------------------------------------------------")

    # Linearna regresija model
    from sklearn.linear_model import Lasso
    regressor = LinearRegression()
    # Treniranje modela
    regressor.fit(X_train, y_train)
    # Evaluacija modela
    y_pred = regressor.predict(X_test)
    print("R^2 adjusted:")
    r_adjusted1=line.get_rsquared_adj(regressor,X_test,y_test)
    print(line.get_rsquared_adj(regressor,X_test,y_test))
    mae1 = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae1}')
    mse1 = mean_squared_error(y_test, y_pred)
    r2_1 = r2_score(y_test, y_pred)
    # Regresija testiranje
    data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==round)]
    new_X = data_2023.drop(columns=['positionOrder','Top 3 Finish'])
    # Prikazivanje koeficijenata atributa zajedno sa nazivima
    print("Koeficijenti atributa:")
    predicted_classes_2023 = regressor.predict(new_X)
    atributi_nazivi = X_train.columns
    regresijacoef=regressor.coef_
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
    # Prikaz greske za izabranu rundu i godinu
    accuracy = accuracy_score(df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==round)]['positionOrder'],predicted_classes_assigned)
    print(f'Accuracy: {accuracy}')
    mae = mean_absolute_error(df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==round)]['positionOrder'],predicted_classes_assigned)
    print(f"Mean Absolute Error: {mae}")
    min_index = np.argmin(predicted_classes_assigned)
    predicted1=drivers_df.where(drivers_df['driverId']==new_X.iloc[min_index]['driverId'])['surname'].dropna()
    realwinner1 = drivers_df.loc[drivers_df['driverId'] == new_X.iloc[0]['driverId'], 'surname'].item()
    print(predicted1)
    print(f'Predicted Classes for 2023: {predicted_classes_2023}')

    # Predikcija za celu izabranu godinu 
    print("----------------PREDIKCIJA Godine------------------")
    poslednji_red = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year)].iloc[-1]
    round_vrednost = poslednji_red['round']
    count = int(round_vrednost)+1
    result_df_reg = pd.DataFrame({
        'surname': drivers_df['surname'],
        'points': 0 ,
        'driverId':drivers_df['driverId'],
        'real_points':0,
        'constructorId':0,
    })
    for i in range(len(new_X)):
        result_df_reg.loc[result_df_reg['driverId'] == new_X.iloc[i]['driverId'], 'constructorId'] = new_X.iloc[i]['constructorId']

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
            result_df_reg.loc[result_df_reg['driverId'] == new_X.iloc[min_index]['driverId'], 'points'] += points[i]
            result_df_reg.loc[result_df_reg['driverId'] == new_X.iloc[i]['driverId'], 'real_points'] += points[i]
            rezkopija[min_index]=rezkopija[min_index]+20
        if winner_surname==winner_surname1:
            right=right+1
    count2=count-2 
    accuracyreg=(right/count2)*100
    zbirprocenatareg=zbirprocenatareg+accuracyreg     
    print("Procenat pogodjenih pobednika je: {:.2%}".format(right/count2))  
    accuracyreg1=(right_podium_total/count2)*100
    print("Procenat pogodjenih kompletnih podiuma je: {:.2%}".format(right_podium_total/count2))
    accuracyreg2=(right_podium/count2)*100
    print("Procenat pogodjenih da je prediktovani pobednik zavrsio na podiumu je: {:.2%}".format(right_podium/count2))   
    result_df_reg = result_df_reg[(result_df_reg['real_points'] != 0) | (result_df_reg['points'] != 0)]   
    result_df_reg=result_df_reg.sort_values(by='points', ascending=False)   
    result_df_reg.to_csv("RezultatiReg.csv",index=False)
    saberi_po_constructorId = result_df_reg.groupby('constructorId')['points'].sum().reset_index()
    saberi_po_constructorId = saberi_po_constructorId.sort_values(by='points', ascending=False).reset_index(drop=True)
    merged_df = saberi_po_constructorId.merge(constructor_df, on='constructorId', how='left')
    print("PREDIKCIJA PORETKA ZA KONSTRUKTORE:")
    print(merged_df[['constructorId', 'constructorRef', 'points']])
    print("STVARNI POREDAK KONSTRUKTORA")
    saberi_po_constructorId = result_df_reg.groupby('constructorId')['real_points'].sum().reset_index()
    saberi_po_constructorId = saberi_po_constructorId.sort_values(by='real_points', ascending=False).reset_index(drop=True)
    merged_df = saberi_po_constructorId.merge(constructor_df, on='constructorId', how='left')
    print(merged_df[['constructorId', 'constructorRef', 'real_points']])


    #---------------------------------------------------------------------------------------------------------
    print("--------------------------------RANDOM FOREST CLASSIFIER-------------------------------------------")

    from sklearn.metrics import precision_score
    classifier = RandomForestClassifier(n_estimators=50,max_depth=5,random_state=42)
    y=y.apply(lambda x: 1 if x == 1 else 2 if x==2 else 3 if x==3 else 4)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.2, random_state=42)
    classifier.fit(X_train1, y_train1)

    # Evaluacija modela
    print("R^2 adjusted:")
    r2_adjustedrfc=line.get_rsquared_adj(classifier,X_test1,y_test1)
    print(line.get_rsquared_adj(classifier,X_test1,y_test1))
    y_pred = classifier.predict(X_test1)
    accuracy = accuracy_score(y_test1, y_pred)
    print(f'Accuracy: {accuracy}')
    df = pd.DataFrame({'y_test': y_test1, 'y_pred': y_pred})
    correct_predictions=0
    incorrect_predictions=0
    for index, row in df.iterrows():
        if row['y_test'] == 1 and row['y_test'] == row['y_pred']:
            correct_predictions += 1
        elif row['y_test'] != 1 and row['y_pred'] == 1:
            incorrect_predictions += 1
    total_y_test_1 = len(df[df['y_test'] == 1])
    accuracy_for_y_test_1 = correct_predictions / total_y_test_1 if total_y_test_1 > 0 else 0
    print(f'Preciznost za y_test == 1: {__builtins__.round(accuracy_for_y_test_1*100,1)}%')

    total_y_test_2 = len(df[df['y_pred'] == 1])
    accuracy_for_y_test_2 = incorrect_predictions / total_y_test_2 if total_y_test_2 > 0 else 0
    print(f'Preciznost za y_pred == 1: {__builtins__.round(accuracy_for_y_test_2*100,1)}%')

    # Predikcija za klassifier=
    data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year  ) & (df_final_keepPositionOrder['round']==round )]
    new_X = data_2023.drop(columns=['positionOrder','Top 3 Finish'])
    predicted_classes_2023 = classifier.predict_proba(new_X)

    # Prikazivanje važnosti atributa
    importancesrfc = classifier.feature_importances_
    feature_names = X_train1.columns
    for feature, importance in zip(feature_names, importancesrfc):
        print(f"{feature}: {importance}")

    indeks_najveceg_reda = np.argmax(predicted_classes_2023[:, 0])
    print("Prediktovani pobednik je:")
    predicted2=drivers_df.where(drivers_df['driverId']==new_X.iloc[indeks_najveceg_reda]['driverId'])['surname'].dropna()
    print(predicted2)
    print("Stvaran pobednik je:")
    print(drivers_df.where(drivers_df['driverId']==new_X.iloc[0]['driverId'])['surname'].dropna())
    print(f'Predicted Classes for 2023: {predicted_classes_2023}')

    # Predikcija za celu izabranu godinu
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
    accuracyrfc=(right/count2)*100 
    zbirprocenatarfc= zbirprocenatarfc+accuracyrfc  
    print("Procenat pogodjenih pobednika je: {:.2%}".format(right/count2))
    accuracyrfc1=(right_podium_total/count2)*100
    print("Procenat pogodjenih kompletnih podiuma je: {:.2%}".format(right_podium_total/count2))
    accuracyrfc2=(right_podium/count2)*100
    print("Procenat pogodjenih da je prediktovani pobednik zavrsio na podiumu je: {:.2%}".format(right_podium/count2))
    #----------------------------------------------------------------------------------------------------------
    print("-----------------------------------------RANDOM FOREST REGRESSOR-----------------------------------")
    from sklearn.ensemble import RandomForestRegressor

    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_train, y_train)
    # Evaluacija modela
    print("R^2 adjusted:")
    r_adjusted2=line.get_rsquared_adj(regressor,X_test,y_test)
    print(line.get_rsquared_adj(regressor,X_test,y_test))
    y_pred = regressor.predict(X_test)
    mae2 = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae2}')
    mse2=mean_squared_error(y_test, y_pred)
    r2_2=r2_score(y_test, y_pred)
    # rfr testiranje
    data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==round)]
    new_X = data_2023.drop(columns=['positionOrder','Top 3 Finish'])
    predicted_classes_2023 = regressor.predict(new_X)
    #Vaznost atributa
    print("Feature importances:")
    atributi_nazivi = X_train.columns
    importancesrfr=regressor.feature_importances_
    for naziv, importance in zip(atributi_nazivi, regressor.feature_importances_):
        print(f"Feature '{naziv}': {importance}")

    # SORTIRANJE FINALNIH POZICIJA
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
    predicted3=drivers_df.where(drivers_df['driverId']==new_X.iloc[min_index]['driverId'])['surname'].dropna()
    print(predicted3)
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
    accuracyrfr=(right/count2)*100  
    zbirprocenatarfr= zbirprocenatarfr+accuracyrfr  
    print("Procenat pogodjenih pobednika je: {:.2%}".format(right/count2))  
    accuracyrfr1=(right_podium_total/count2)*100 
    print("Procenat pogodjenih kompletnih podiuma je: {:.2%}".format(right_podium_total/count2))
    accuracyrfr2=(right_podium/count2)*100 
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
    saberi_po_constructorId = resultforest_df.groupby('constructorId')['real_points'].sum().reset_index()
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
    model = LogisticRegression(random_state=42)
    # Treniranje modela
    model.fit(X_train4, y_train4)
    r_adjustedlog=line.get_rsquared_adj(model,X_test4,y_test4)
    y_pred = model.predict(X_test4)
    labels_proba = model.predict_proba(X_test4)[:, 1]
    y_pred_proba=model.predict_proba(X_test4)
    # Evaluacija performansi modela
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    correct_predictions=0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and np.argmax(y_pred_proba[i])==1:
            correct_predictions += 1
    total_y_test_1 = len(y_test4[y_test4 == 1])
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


    # Predikcija za izabranu godinu i rundu
    data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==round)]
    new_X = data_2023.drop(columns=['positionOrder','Top 3 Finish'])
    predicted_classes_2023 = model.predict_proba(new_X)
    indeks_najveceg_reda = np.argmax(predicted_classes_2023[:, 1])
    print("Prediktovani pobednik je:")
    predicted4=drivers_df.where(drivers_df['driverId']==new_X.iloc[indeks_najveceg_reda]['driverId'])['surname'].dropna()
    print(predicted4)
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
    accuracylog=(right/count2)*100 
    zbirprocenatalog=zbirprocenatalog+accuracylog     
    print("Procenat pogodjenih pobednika je: {:.2%}".format(right/count2))
    #-------------------------------------------------------------------------------------------------------------------
    print("-------------------------------------ROS-----------------------------------------------------")
    from imblearn.over_sampling import RandomOverSampler
    from collections import Counter

    ros = RandomOverSampler(random_state=42)

    X_train_resampled, y_train_resampled = ros.fit_resample(X_train1, y_train1)

    print("Before oversampling: ", Counter(y_train1))
    print("After oversampling: ", Counter(y_train_resampled))

    classifier = RandomForestClassifier(n_estimators=50,max_depth=5,random_state=42)
    classifier.fit(X_train_resampled, y_train_resampled)

    # Evaluacija modela
    print("R^2 adjusted:")
    r_adjustedross=line.get_rsquared_adj(classifier,X_test1,y_test1)
    print(line.get_rsquared_adj(classifier,X_test1,y_test1))
    y_pred = classifier.predict(X_test1)
    accuracy = accuracy_score(y_test1, y_pred)
    print(f'Accuracy: {accuracy}')
    df = pd.DataFrame({'y_test': y_test1, 'y_pred': y_pred})
    correct_predictions=0
    incorrect_predictions=0
    for index, row in df.iterrows():
        if row['y_test'] == 1 and row['y_test'] == row['y_pred']:
            correct_predictions += 1
        elif row['y_test'] != 1 and row['y_pred'] == 1:
            incorrect_predictions += 1
    total_y_test_1 = len(df[df['y_test'] == 1])
    accuracy_for_y_test_1 = correct_predictions / total_y_test_1 if total_y_test_1 > 0 else 0
    print(f'Preciznost za y_test == 1: {__builtins__.round(accuracy_for_y_test_1*100,1)}%')
    total_y_test_2 = len(df[df['y_pred'] == 1])
    accuracy_for_y_test_2 = incorrect_predictions / total_y_test_2 if total_y_test_2 > 0 else 0
    print(f'Preciznost za y_pred == 1: {__builtins__.round(accuracy_for_y_test_2*100,1)}%')
    #Predikcija za klassifier

    data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year  ) & (df_final_keepPositionOrder['round']==round )]
    new_X = data_2023.drop(columns=['positionOrder','Top 3 Finish'])
    predicted_classes_2023 = classifier.predict_proba(new_X)

    # Prikazivanje važnosti atributa
    importancesros = classifier.feature_importances_
    feature_names = X_train1.columns
    for feature, importance in zip(feature_names, importancesros):
        print(f"{feature}: {importance}")

    indeks_najveceg_reda = np.argmax(predicted_classes_2023[:, 0])
    print("Prediktovani pobednik je:")
    predicted5=drivers_df.where(drivers_df['driverId']==new_X.iloc[indeks_najveceg_reda]['driverId'])['surname'].dropna()
    print(predicted5)
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
    accuracyros=(right/count2)*100 
    zbirprocenataros=zbirprocenataros+accuracyros
    print("Procenat pogodjenih pobednika je: {:.2%}".format(right/count2))
    accuracyros1=(right_podium_total/count2)*100 
    print("Procenat pogodjenih kompletnih podiuma je: {:.2%}".format(right_podium_total/count2))
    accuracyros2=(right_podium/count2)*100 
    print("Procenat pogodjenih da je prediktovani pobednik zavrsio na podiumu je: {:.2%}".format(right_podium/count2))
    #-------------------------------------------------------------------------------------------------------------------
    print("---------------------------LASSO-----------------------------------------------------")
    from sklearn.model_selection import cross_val_score
    alphas = [0.01, 0.05, 0.1, 0.15, 1.0, 5.0, 10.0]

    lasso_regressor = Lasso(alpha=0.5,random_state=42)
    lasso_regressor.fit(X_train, y_train)
    print("R^2 adjusted:")
    r_adjusted3=line.get_rsquared_adj(lasso_regressor,X_test,y_test)
    print(line.get_rsquared_adj(lasso_regressor,X_test,y_test))
    y_pred = lasso_regressor.predict(X_test)
    lassocoef = lasso_regressor.coef_
    mae3 = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae3}')
    mse3 = mean_squared_error(y_test, y_pred)
    r2_3= r2_score(y_test, y_pred)
    # Lasso testiranje
    data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==round)]
    new_X = data_2023.drop(columns=['positionOrder','Top 3 Finish'])
    predicted_classes_2023 = lasso_regressor.predict(new_X)
    min_index = np.argmin(predicted_classes_2023)
    predictedlas=drivers_df.where(drivers_df['driverId']==new_X.iloc[min_index]['driverId'])['surname'].dropna()
    print(predictedlas)
    print(f'Predicted Classes for 2023: {predicted_classes_2023}')
    print("PREDIKCIJA GODINE")
    poslednji_red = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year)].iloc[-1]
    round_vrednost = poslednji_red['round']
    count = int(round_vrednost)+1
    result_df_lasso = pd.DataFrame({
        'surname': drivers_df['surname'],
        'points': 0 ,
        'driverId':drivers_df['driverId'],
        'real_points':0,
        'constructorId':0,
    })
    for i in range(len(new_X)):
        result_df_lasso.loc[result_df_lasso['driverId'] == new_X.iloc[i]['driverId'], 'constructorId'] = new_X.iloc[i]['constructorId']

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
            result_df_lasso.loc[result_df_lasso['driverId'] == new_X.iloc[min_index]['driverId'], 'points'] += points[i]
            result_df_lasso.loc[result_df_lasso['driverId'] == new_X.iloc[i]['driverId'], 'real_points'] += points[i]
            rezkopija[min_index]=rezkopija[min_index]+20
        if winner_surname==winner_surname1:
            right=right+1
    count2=count-2       
    accuracylasso=(right/count2)*100
    zbirprocenatalasso=zbirprocenatalasso+accuracylasso
    print("Procenat pogodjenih pobednika je: {:.2%}".format(right/count2)) 
    accuracylasso1=(right_podium_total/count2)*100 
    print("Procenat pogodjenih kompletnih podiuma je: {:.2%}".format(right_podium_total/count2))
    accuracylasso2=(right_podium/count2)*100 
    print("Procenat pogodjenih da je prediktovani pobednik zavrsio na podiumu je: {:.2%}".format(right_podium/count2))   
    result_df_lasso = result_df_lasso[(result_df_lasso['real_points'] != 0) | (result_df_lasso['points'] != 0)]   
    result_df_lasso=result_df_lasso.sort_values(by='points', ascending=False)   
    result_df_lasso.to_csv("RezultatiLasso.csv",index=False)
    saberi_po_constructorId = result_df_lasso.groupby('constructorId')['points'].sum().reset_index()
    saberi_po_constructorId = saberi_po_constructorId.sort_values(by='points', ascending=False).reset_index(drop=True)
    merged_df = saberi_po_constructorId.merge(constructor_df, on='constructorId', how='left')
    print("PREDIKCIJA PORETKA ZA KONSTRUKTORE:")
    print(merged_df[['constructorId', 'constructorRef', 'points']])
    print("STVARNI POREDAK KONSTRUKTORA")
    saberi_po_constructorId = result_df_lasso.groupby('constructorId')['real_points'].sum().reset_index()
    saberi_po_constructorId = saberi_po_constructorId.sort_values(by='real_points', ascending=False).reset_index(drop=True)
    merged_df = saberi_po_constructorId.merge(constructor_df, on='constructorId', how='left')
    print(merged_df[['constructorId', 'constructorRef', 'real_points']])
    #-----------------------------------------------------------------------------------------------------------------------------------------
    print("--------------------------------L2 RIDGE------------------------------------------------------------------------")
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    ridge_regressor = Ridge(alpha=3,random_state=42)  
    ridge_regressor.fit(X_train, y_train)
    print("R^2 adjusted:")
    r_adjusted4=line.get_rsquared_adj(ridge_regressor,X_test,y_test)
    print(line.get_rsquared_adj(ridge_regressor,X_test,y_test))
    y_pred = ridge_regressor.predict(X_test)

    mae4 = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae4}')
    mse4 = mean_squared_error(y_test, y_pred)
    r2_4 = r2_score(y_test, y_pred)
    # Koeficijenti sa Ridge regresijom
    ridgecoef = ridge_regressor.coef_
    print("Koeficijenti sa Ridge regresijom:")
    print(ridgecoef)

    # Ridge testiranje
    data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round']==round)]
    new_X = data_2023.drop(columns=['positionOrder','Top 3 Finish'])
    predicted_classes_2023 = ridge_regressor.predict(new_X)
    min_index = np.argmin(predicted_classes_2023)
    predicted6=drivers_df.where(drivers_df['driverId']==new_X.iloc[min_index]['driverId'])['surname'].dropna()
    print(predicted6)
    print(f'Predicted Classes for 2023: {predicted_classes_2023}')
    print("PREDIKCIJA GODINE")

    poslednji_red = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year)].iloc[-1]
    round_vrednost = poslednji_red['round']
    count = int(round_vrednost)+1
    result_df_ridge = pd.DataFrame({
        'surname': drivers_df['surname'],
        'points': 0 ,
        'driverId':drivers_df['driverId'],
        'real_points':0,
        'constructorId':0,
    })
    for i in range(len(new_X)):
        result_df_ridge.loc[result_df_ridge['driverId'] == new_X.iloc[i]['driverId'], 'constructorId'] = new_X.iloc[i]['constructorId']

    right=0
    right_podium_total=0
    right_podium=0
    for i in range(2, count):
        counter=0
        print("#########################################################")
        data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year) & (df_final_keepPositionOrder['round'] == i)]
        new_X = data_2023.drop(columns=['positionOrder', 'Top 3 Finish'])
        predicted_classes_2023 = ridge_regressor.predict(new_X)
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
            result_df_ridge.loc[result_df_ridge['driverId'] == new_X.iloc[min_index]['driverId'], 'points'] += points[i]
            result_df_ridge.loc[result_df_ridge['driverId'] == new_X.iloc[i]['driverId'], 'real_points'] += points[i]
            rezkopija[min_index]=rezkopija[min_index]+20
        if winner_surname==winner_surname1:
            right=right+1
    count2=count-2 
    accuracyridge=(right/count2)*100
    zbirprocenataridge=zbirprocenataridge+ accuracyridge  
    print("Procenat pogodjenih pobednika je: {:.2%}".format(right/count2)) 
    accuracyridge1=(right_podium_total/count2)*100   
    print("Procenat pogodjenih kompletnih podiuma je: {:.2%}".format(right_podium_total/count2))
    accuracyridge2=(right_podium/count2)*100   
    print("Procenat pogodjenih da je prediktovani pobednik zavrsio na podiumu je: {:.2%}".format(right_podium/count2))   
    result_df_ridge = result_df_ridge[(result_df_ridge['real_points'] != 0) | (result_df_ridge['points'] != 0)]   
    result_df_ridge=result_df_ridge.sort_values(by='points', ascending=False)   
    result_df_ridge.to_csv("RezultatiRidge.csv",index=False)
    saberi_po_constructorId = result_df_ridge.groupby('constructorId')['points'].sum().reset_index()
    saberi_po_constructorId = saberi_po_constructorId.sort_values(by='points', ascending=False).reset_index(drop=True)
    merged_df = saberi_po_constructorId.merge(constructor_df, on='constructorId', how='left')
    print("PREDIKCIJA PORETKA ZA KONSTRUKTORE:")
    print(merged_df[['constructorId', 'constructorRef', 'points']])
    print("STVARNI POREDAK KONSTRUKTORA")
    saberi_po_constructorId = result_df_ridge.groupby('constructorId')['real_points'].sum().reset_index()
    saberi_po_constructorId = saberi_po_constructorId.sort_values(by='real_points', ascending=False).reset_index(drop=True)
    merged_df = saberi_po_constructorId.merge(constructor_df, on='constructorId', how='left')
    print(merged_df[['constructorId', 'constructorRef', 'real_points']])   
    #----------------------------------------------------------------------------------------------------------------------------------------------
    print("-----------------------------------------ElasticNet--------------------------------------------------------------")
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GridSearchCV


    elastic_net_regressor = ElasticNet(random_state=42)

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
    r_adjusted5=line.get_rsquared_adj(best_elastic_net,X_test,y_test)
    print(line.get_rsquared_adj(best_elastic_net,X_test,y_test))
    # Predviđanje na test setu
    y_pred = best_elastic_net.predict(X_test)
    # Evaluacija modela
    mae5 = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mse}')
    mse5 = mean_squared_error(y_test, y_pred)
    r2_5 = r2_score(y_test, y_pred)
    elasticoef = best_elastic_net.coef_
    data_2023 = df_final_keepPositionOrder[(df_final_keepPositionOrder['year'] == year  ) & (df_final_keepPositionOrder['round']==round )]
    new_X = data_2023.drop(columns=['positionOrder','Top 3 Finish'])
    predicted_classes_2023 = best_elastic_net.predict(new_X)
    indeks_najveceg_reda = np.argmin(predicted_classes_2023)
    print("Prediktovani pobednik je:")
    predicted7=drivers_df.where(drivers_df['driverId']==new_X.iloc[indeks_najveceg_reda]['driverId'])['surname'].dropna()
    print(predicted7)
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
    accuracyelastic=(right/count2)*100
    zbirprocenataelastic=zbirprocenataelastic+accuracyelastic       
    print("Procenat pogodjenih pobednika je: {:.2%}".format(right/count2))
    year=year+1
    print(year)
year=year-1
print("LOOOP")
print(loop)
zbirprocenatareg=zbirprocenatareg/(loop+1)
zbirprocenatarfr=zbirprocenatarfr/(loop+1)
zbirprocenatalog=zbirprocenatalog/(loop+1)
zbirprocenatarfc=zbirprocenatarfc/(loop+1)
zbirprocenataros=zbirprocenataros/(loop+1)
zbirprocenatalasso=zbirprocenatalasso/(loop+1)
zbirprocenataridge=zbirprocenataridge/(loop+1)
zbirprocenataelastic=zbirprocenataelastic/(loop+1)
#-------------------------------------------------------------------------------------------------------------------------
print("------------------------Linearna Regresija Predvidjanje izlaska safety car-a---------------------------------------")
if status==0:
    result_df=result_df.merge(races_df[['raceId', 'year','round']], on='raceId', how='left')
    result_df = result_df.sort_values(by=['year', 'round', 'position'], ascending=[True, True, False])
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
year1=2019
round=10
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
aggregations = {'stop': 'sum', 'weather_warm': 'first','weather_cold':'first', 'weather_dry': 'first','weather_wet':'first','weather_cloudy':'first'}
aggregations2 = {'Group_Count': 'first', 'circuitId': 'first'}
aggregations3 = {'stop': 'sum', 'weather_warm': 'first','weather_cold':'first', 'weather_dry': 'first','weather_wet':'first','weather_cloudy':'first','Group_Count':'first','status_1_count':'first','AVG pitstop':'first'}

pit_with_year = pitstops_df.merge(races_df[['raceId', 'year','round','circuitId']], on='raceId', how='left')
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
#----------------------------------------------------------------------------------------------------------------------------------------------------
print("----------------------------------Predikcija Broja izlaska safety car-a na trci----------------------------------------------")
df_final_keepPositionOrder1=df_final_keepPositionOrder.drop(columns=['driverId','constructorId','grid','positionOrder','Top 3 Finish','Driver Top 3 Finish Percentage (Last Year)','Constructor Top 3 Finish Percentage (Last Year)','Driver Top 3 Finish Percentage (This Year till last race)','Constructor Top 3 Finish Percentage (This Year till last race)','Driver Avg position (Last Year)','Constructor Avg position (Last Year)','Driver Average Position (This Year till last race)','Constructor Average Position (This Year till last race)','Champ Last Year'])
df_final_keepPositionOrder1 = pd.merge(df_final_keepPositionOrder1,safety_cars_df_final [['year','circuitId','Group_Count']], on=['year', 'circuitId'], how='left')
df_final_keepPositionOrder1=df_final_keepPositionOrder1.fillna(0)
df_final_keepPositionOrder1 = pd.merge(df_final_keepPositionOrder1,weather_df_final [['year','circuitId','weather_warm','weather_cold','weather_dry','weather_wet','weather_cloudy']], on=['year', 'circuitId'], how='left')
df_final_keepPositionOrder1=df_final_keepPositionOrder1.dropna()
novo_ime = {'Group_Count': 'SafetyCar'}
df_final_keepPositionOrder1 = df_final_keepPositionOrder1.rename(columns=novo_ime)
df_final_keepPositionOrder1.loc[df_final_keepPositionOrder1['SafetyCar'] > 0, 'SafetyCar'] = 1
df_final_keepPositionOrder1['statusId'] = df_final_keepPositionOrder1['statusId'].apply(
    lambda x: 0 if x==1 else 0 if x==2 else 1
)
df_final_keepPositionOrder1 = df_final_keepPositionOrder1.groupby(['year', 'round']).agg({
    'SafetyCar': 'first',
    'weather_warm': 'first',
    'weather_cold': 'first',
    'weather_dry': 'first',
    'weather_wet': 'first',
    'weather_cloudy': 'first',
    'circuitId': 'first',
    'statusId': 'sum'    
}).reset_index()
filtered_df1 = df_final_keepPositionOrder1[df_final_keepPositionOrder1['year'] < 2019]
x=filtered_df1.drop(columns='SafetyCar')
y=filtered_df1['SafetyCar']
X_train6, X_test6, y_train6, y_test6 = train_test_split(x, y, test_size=0.3, random_state=42)


classifier = RandomForestClassifier(n_estimators=50,max_depth=5,random_state=42)
classifier.fit(X_train6, y_train6)
# Evaluacija modela
print("R^2 adjusted:")
r2_adjustedsafety=line.get_rsquared_adj(classifier,X_train6,y_train6)
print(line.get_rsquared_adj(classifier,X_train6,y_train6))
y_pred = classifier.predict(X_test6)
accuracysafety = accuracy_score(y_test6, y_pred)
print(f'Accuracy: {accuracysafety}')
df = pd.DataFrame({'y_test': y_test6, 'y_pred': y_pred})
correct_predictions=0
incorrect_predictions=0
for index, row in df.iterrows():
    if row['y_test'] == 1 and row['y_test'] == row['y_pred']:
        correct_predictions += 1
    elif row['y_test'] != 1 and row['y_pred'] == 1:
        incorrect_predictions += 1
total_y_test_1 = len(df[df['y_test'] == 1])
accuracy_for_y_test_1 = correct_predictions / total_y_test_1 if total_y_test_1 > 0 else 0
print(f'Preciznost za y_test == 1: {__builtins__.round(accuracy_for_y_test_1*100,1)}%')

total_y_test_2 = len(df[df['y_pred'] == 1])
accuracy_for_y_test_2 = incorrect_predictions / total_y_test_2 if total_y_test_2 > 0 else 0
print(f'Preciznost za y_pred == 1: {__builtins__.round(accuracy_for_y_test_2*100,1)}%')

#Predikcija za klassifier

data_2023 = df_final_keepPositionOrder1[(df_final_keepPositionOrder1['year'] == 2019  ) & (df_final_keepPositionOrder1['round']==7 )]
new_X = data_2023.drop(columns=['SafetyCar'])
predicted_classes_2023 = classifier.predict(new_X)

# Prikazivanje važnosti atributa
importancessafety = classifier.feature_importances_
feature_names = X_train6.columns
for feature, importance in zip(feature_names, importancessafety):
    print(f"{feature}: {importance}")
print("Predikcija je tacna:")
if predicted_classes_2023==data_2023['SafetyCar'].values:
    print("DA")
else:
    print("NE")
print(f'Predicted Classes for 2019: {predicted_classes_2023}')
right=0
for i in range(2, 22):
    print("#########################################################")
    data_2023 = df_final_keepPositionOrder1[(df_final_keepPositionOrder1['year'] == 2019) & (df_final_keepPositionOrder1['round'] == i)]
    new_X = data_2023.drop(columns=['SafetyCar'])
    predicted_classes_2023 = classifier.predict(new_X)
    print("Predikcija je:")
    print(predicted_classes_2023)
    print("Stvarano je:")
    print(data_2023['SafetyCar'].values)
    if predicted_classes_2023==data_2023['SafetyCar'].values:
        right=right+1
procenat1=(right/20)*100     
print("Procenat pogodjenih izlaska safety car-a je: {:.2%}".format(right/20))

#----------------------------------------------------------------------------------------------------------------------------------------------------
print("---------------------------Promena guma predikcija------------------------------")
safety_cars_df_final1 = safety_cars_df_final[(safety_cars_df_final['year'] > 2010) & (safety_cars_df_final['year'] < 2020)]
weather_df_final = pd.merge(weather_df_final, safety_cars_df_final1[['year', 'circuitId','Group_Count']], on=['year', 'circuitId'], how='left')
weather_df_final = weather_df_final.fillna(0)
df_final_keepPositionOrder['statusId'] = df_final_keepPositionOrder['statusId'].apply(
    lambda x: 1 if x==1 else 1 if x==2 else 2
)
df_final_keepPositionOrder['status_1_count'] = df_final_keepPositionOrder.groupby(['year', 'round'])['statusId'].transform(lambda x: (x == 1).sum())
df_final_keepPositionOrder_first = df_final_keepPositionOrder.drop_duplicates(['year', 'round'], keep='first')
weather_df_final['Group_Count']=weather_df_final['Group_Count'].apply(lambda x: 0 if x == 0 else 0 if x==1 else x)
weather_df_final = pd.merge(weather_df_final, df_final_keepPositionOrder_first[['year', 'round','status_1_count']], on=['year', 'round'], how='left')
mean_value_column = weather_df_final['status_1_count'].mean()
weather_df_final['status_1_count'] = weather_df_final['status_1_count'].fillna(mean_value_column)
weather_df_final=weather_df_final.sort_values(by=['year','round'], ascending=True)
weather_df_final['AVG pitstop']=0
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
weather_df_final_test = weather_df_final[weather_df_final['year'] < year1]
weather_df_final_test = weather_df_final_test.groupby(['year','circuitId','round']).agg(aggregations3).reset_index()
weather_df_final_test=weather_df_final_test.sort_values(by=['year','round'], ascending=True)
y = weather_df_final_test["stop"]
x = weather_df_final_test.drop(columns=["stop"])
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)

regressor = RandomForestRegressor(n_estimators=100,max_depth=5)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse=mean_squared_error(y_test, y_pred)
r2_adjustedpitstop=line.get_rsquared_adj(regressor,X_test,y_test)
print("R^2 adjusted:")
print(line.get_rsquared_adj(regressor,X_test,y_test))
print(f'Mean Absolute Error: {mae:.2f}')


data = weather_df_final[(weather_df_final['year'] == year1) & (weather_df_final['round']==round)]
new_X = data.drop(columns=['stop'])
print("Feature importances:")
atributi_nazivi = X_train.columns
importancespitstop=regressor.feature_importances_
for naziv, importance in zip(atributi_nazivi, regressor.feature_importances_):
    print(f"Feature '{naziv}': {importance}")


print("Originalni niz:")
print(predicted_classes_2023)
sum_stop_actual = weather_df_final[(weather_df_final['year'] == year1) & (weather_df_final['round'] == round)]['stop'].reset_index(drop=True)
sum_stop_predicted = predicted_classes_2023[0]
print(f"Prediktovan broj pitstopova za rundu {round} je: {sum_stop_predicted}")
print(f"Stvaran broj pitstopova za rundu {round} je: {sum_stop_actual}")

poslednji_red = weather_df_final[(weather_df_final['year'] == year1)].iloc[-1]
round_vrednost = poslednji_red['round']
count = int(round_vrednost)+1
prosek=0
oduzmioutliere=0
stvaranprosek=0
for i in range(1,count):
    print("#########################################################")
    data_2023 = weather_df_final[(weather_df_final['year'] == year1) & (weather_df_final['round'] == i)]
    new_X = data_2023.drop(columns=['stop',])  
    predicted_classes_2023 = regressor.predict(new_X)
    sum_stop_actual = weather_df_final[(weather_df_final['year'] == year1) & (weather_df_final['round'] == i)]['stop'].reset_index(drop=True)
    sum_stop_predicted = predicted_classes_2023[0]
    print(f"Prediktovan broj pitstopova za rundu {i} je: {sum_stop_predicted}")
    print(f"Stvaran broj pitstopova za rundu {i} je: {sum_stop_actual}")
    if any((sum_stop_actual > sum_stop_predicted) & (sum_stop_predicted / sum_stop_actual < 0.5)):
        oduzmioutliere = oduzmioutliere + 1
    else:
        prosek = prosek + abs((int(sum_stop_predicted) - sum_stop_actual))
        stvaranprosek=stvaranprosek+sum_stop_actual
print("PROSECNA GRESKA BRE OUTLIERA odnosno ukoliko je predikcija upola ili vise od pola manja od stvarne")
pitstopavg=prosek/(count-1-oduzmioutliere)
stvaranprosek=stvaranprosek/(count-1-oduzmioutliere)
pitstop_avg_in_percentage=(pitstopavg/stvaranprosek)*100
print(f"PROSECNO GRESI ZA {__builtins__.round(prosek/(count-1-oduzmioutliere),1)} PO TRCI")
#-------------------------------------------------------------------------------------------------------------------------
print("------------------------------GRAFICKI PRIKAZ FATALNIH INCIDENATA----------------------------------")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('fatal_accidents_drivers.csv')
df['Date Of Accident'] = pd.to_datetime(df['Date Of Accident'], format='%m/%d/%Y')
df['Godina'] = df['Date Of Accident'].dt.year
df['Decenija'] = (df['Godina'] // 10) * 10
godisnji_incidenti = df.groupby('Godina').size().reset_index(name='Broj incidenata')
decenijski_incidenti = df.groupby('Decenija').size().reset_index(name='Broj incidenata')
decenijski_incidenti = decenijski_incidenti[decenijski_incidenti['Decenija'] >= 1950]
godisnji_incidenti = godisnji_incidenti.set_index('Godina').reindex(range(godisnji_incidenti['Godina'].min(), 2021)).reset_index()
decenijski_incidenti = decenijski_incidenti.set_index('Decenija').reindex(range(decenijski_incidenti['Decenija'].min(), 2021, 10)).reset_index()

# Vizualizuj trend decenijskih incidenata
plt.figure(figsize=(12, 6))
sns.barplot(x='Decenija', y='Broj incidenata', data=decenijski_incidenti, palette='viridis')
plt.title('Trend incidenata na stazi po decenijama')
plt.xlabel('Decenija')
plt.ylabel('Broj incidenata')
plt.show()
#-------------------------------------------------------------------------------------------------------------------------
#UPOREDJIVANJE MODELA PREKO GRAFIKA
model_names = ['LinearReg', 'RFC', 'RFR', 'LogisticReg', 'ROS', 'Lasso', 'Ridge', 'Elastic']
r2_values = [r_adjusted1, r2_adjustedrfc, r_adjusted2, r_adjustedlog, r_adjustedross, r_adjusted3, r_adjusted4, r_adjusted5]
colors = ['blue', 'green', 'orange', 'yellow', 'black']
plt.bar(model_names, r2_values, color=colors)
plt.ylim(0, 1) 
plt.title('R^2 adjusted vrednosti za različite modele')
plt.xlabel('Modeli')
plt.ylabel('R^2 vrednost')
for i, value in enumerate(r2_values):
    plt.text(i, value + 0.02, f'{value:.2f}', ha='center', va='bottom', fontsize=10, color='black')
plt.show()

models = ['LinearReg', 'RFC', 'RFR', 'LogisticReg', 'Ros', 'Lasso', 'Ridge','Elastic']
predicted_values = [predicted1.values, predicted2.values, predicted3.values, predicted4.values, predicted5.values,predictedlas.values, predicted6.values, predicted7.values]
plt.bar(models, [1 if realwinner1 == pred else 0 for pred in predicted_values], color='green', alpha=0.7, label='Tačno predviđanje')
plt.bar(models, [1 if realwinner1 != pred else 0 for pred in predicted_values], color='red', alpha=0.7, label='Pogrešno predviđanje')


for i, model in enumerate(models):
    plt.text(i, 1.05, f'{realwinner1}', ha='center', va='center', color='black',fontweight='bold', fontsize=10)
    plt.text(i, 0.95, f'{predicted_values[i][0]}', ha='center', va='center', color='black',fontweight='bold', fontsize=10)

plt.xlabel('Modeli')
plt.ylabel('Predviđanje')
plt.title('Prikaz stvarnog i predviđenog pobednika za svaki model')
plt.legend()
plt.show()


model_names = ['LinearReg', 'RFR', 'Lasso', 'Ridge', 'Elastic']
metric_names = ['R^2', 'MSE', 'MAE']
r2_values = [r2_1, r2_2, r2_3, r2_4, r2_5]
mse_values = [mse1, mse2, mse3, mse4, mse5]
mae_values = [mae1, mae2, mae3, mae4, mae5]
bar_width = 0.2  
index = np.arange(len(model_names))  
plt.bar(index - bar_width, r2_values, width=bar_width, label='R^2')
plt.bar(index, mse_values, width=bar_width, label='MSE')
plt.bar(index + bar_width, mae_values, width=bar_width, label='MAE')
for i in range(len(model_names)):
    plt.text(i - bar_width, r2_values[i] + 0.02, f'{r2_values[i]:.2f}', ha='center', va='bottom', fontsize=8)
    plt.text(i, mse_values[i] + 0.02, f'{mse_values[i]:.2f}', ha='center', va='bottom', fontsize=8)
    plt.text(i + bar_width, mae_values[i] + 0.02, f'{mae_values[i]:.2f}', ha='center', va='bottom', fontsize=8)

plt.title('Performanse modela po metrikama')
plt.xlabel('Modeli')
plt.ylabel('Vrednosti metrike')
plt.xticks(index, model_names)
plt.legend()
plt.show()


models = ['LinearReg', 'RFC', 'RFR', 'LogisticReg', 'Ros', 'Lasso', 'Ridge','Elastic']
procenat_pogodaka = [accuracyreg, accuracyrfc, accuracyrfr, accuracylog,accuracyros, accuracylasso, accuracyridge,accuracyelastic] 
plt.bar(models, procenat_pogodaka, color='skyblue')
for i, procenat in enumerate(procenat_pogodaka):
    plt.text(i, procenat + 1, f'{procenat:.2f}%', ha='center', va='bottom', fontsize=10)

plt.xlabel('Modeli')
plt.ylabel(f'Procenat pogodaka za godinu {year}')
plt.title('Procenat pogodaka za svaki model')
plt.ylim(0, 100)
plt.show()

models = ['LinearReg', 'RFC', 'RFR', 'Ros', 'Lasso', 'Ridge']
procenat_pogodaka = [accuracyreg1, accuracyrfc1, accuracyrfr1,accuracyros1, accuracylasso1, accuracyridge1]
plt.bar(models, procenat_pogodaka, color='skyblue')
for i, procenat in enumerate(procenat_pogodaka):
    plt.text(i, procenat + 1, f'{procenat:.2f}%', ha='center', va='bottom', fontsize=10)

plt.xlabel('Modeli')
plt.ylabel('Procenat pogodaka')
plt.title(f'Procenat pogodaka TOP 3 za {year}. godinu')
plt.ylim(0, 100)
plt.show()

models = ['LinearReg', 'RFC', 'RFR', 'Ros', 'Lasso', 'Ridge']
procenat_pogodaka = [accuracyreg2, accuracyrfc2, accuracyrfr2,accuracyros2, accuracylasso2, accuracyridge2]
plt.bar(models, procenat_pogodaka, color='skyblue')
for i, procenat in enumerate(procenat_pogodaka):
    plt.text(i, procenat + 1, f'{procenat:.2f}%', ha='center', va='bottom', fontsize=10)
plt.xlabel('Modeli')
plt.ylabel('Procenat pogodaka')
plt.title(f'Procenat da je predvidjeni pobednik zavrsio u TOP 3 za {year}. godinu')
plt.ylim(0, 100)
plt.show()
#----------------------------------------------------------------------------------------
#ZA SAFETY CAR
odel_names = ['RFC']
metric_names = ['R^2', 'ACCURACY']
model1_metrics = [r2_adjustedsafety, accuracysafety]
bar_width = 0.2
plt.bar(metric_names, model1_metrics, color=['skyblue', 'lightgreen'], width=bar_width)
for i, value in enumerate(model1_metrics):
    plt.text(i, value + 0.02, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
plt.title('Performanse modela po metrikama za safety car')
plt.xlabel('Metrike')
plt.ylabel('Vrednosti metrike')
plt.show()

models = ['RFC']
procenat_pogodaka = [procenat1]
bar_width = 0.2
plt.bar(models, procenat_pogodaka, color='skyblue', width=bar_width)
for i, procenat in enumerate(procenat_pogodaka):
    plt.text(i, procenat + 1, f'{procenat:.2f}%', ha='center', va='bottom', fontsize=10)
plt.xlabel('Modeli')
plt.ylabel('Procenat pogodaka za safety car')
plt.title(f'Procenat tačnosti izlaska safety car-a za {year1}. godinu')
plt.ylim(0, 100)
plt.show()

#----------------------------------------------------------------------------------------
#ZA Promenu guma

model_names = ['RFR']
r2_adjusted_values = [r2_adjustedpitstop]
bar_width = 0.2
plt.bar(model_names, r2_adjusted_values, color=['blue'], width=bar_width)
for i, value in enumerate(r2_adjusted_values):
    plt.text(i, value + 0.02, f'{value:.2f}', ha='center', va='bottom', fontsize=10)
plt.ylim(0, 1) 
plt.title('R^2 adjusted za promenu guma (pitstop)')
plt.xlabel('Modeli')
plt.ylabel('R^2 vrednost')
plt.show()

model_names = ['RFR']
metric_names = ['MAE', 'MSE']
model1_metrics = [mae, mse]
bar_width = 0.2
plt.bar(metric_names, model1_metrics, color=['skyblue', 'lightgreen'], width=bar_width)
for i, value in enumerate(model1_metrics):
    plt.text(i, value + 0.02, f'{value:.2f}', ha='center', va='bottom', fontsize=10)
plt.title('Performanse modela po metrikama za promenu guma')
plt.xlabel('Metrike')
plt.ylabel('Vrednosti metrike')
plt.show()

models = ['RFR']
procenat_pogodaka = [pitstop_avg_in_percentage.values[0]]
bar_width = 0.2
plt.bar(models, procenat_pogodaka, color='skyblue', width=bar_width)
for i, procenat in enumerate(procenat_pogodaka):
    plt.text(i, procenat + 1, f'{procenat:.2f}%', ha='center', va='bottom', fontsize=10)
plt.xlabel('Modeli')
plt.ylabel('Procenat prosečne greške')
plt.title(f'Procenat prosečne greške za promenu guma za {year1}. godinu')
plt.ylim(0, 100)
plt.text(0.5, -5, 'Ovo je komentar', fontsize=10, color='red', ha='center', va='center')
plt.show()
#---------------------------------------------------------------------
#SVAKI OD OVIH PROCENA ALI ZA n GODINA + uneta
models = ['LinearReg', 'RFC', 'RFR', 'LogisticReg', 'Ros', 'Lasso', 'Ridge','Elastic']
procenat_pogodaka = [zbirprocenatareg, zbirprocenatarfc, zbirprocenatarfr, zbirprocenatalog,zbirprocenataros, zbirprocenatalasso, zbirprocenataridge,zbirprocenataelastic] 
plt.bar(models, procenat_pogodaka, color='skyblue')
for i, procenat in enumerate(procenat_pogodaka):
    plt.text(i, procenat + 1, f'{procenat:.2f}%', ha='center', va='bottom', fontsize=10)

plt.xlabel('Modeli')
plt.ylabel('Prosecan procenat pogodjenih pobednika')
plt.title(f'Prosecan Procenat pogodjenih pobednika za prethodnih {loop} godina: {year-loop} - {year}')
plt.ylim(0, 100)
plt.show()
#---------------------------------------------------------------------
#Prikaz championshipa
colors = np.where(result_df_reg['real_points'] == np.minimum(result_df_reg['real_points'], result_df_reg['points']), 'orange', 'blue')
plt.figure(figsize=(10, 6))
plt.barh(result_df_reg['surname'], np.minimum(result_df_reg['real_points'], result_df_reg['points']), color=colors, alpha=1, label='Stvarni poeni')
plt.barh(result_df_reg['surname'], abs(result_df_reg['real_points'] - result_df_reg['points']),left=np.minimum(result_df_reg['real_points'], result_df_reg['points']), color=np.where(colors == 'orange', 'blue', 'orange'), alpha=1, label='Predviđeni poeni')
plt.xlabel('Vozači')
plt.ylabel('Poeni')
plt.title('Predviđeni vs. Stvarni poeni za svakog vozača Regresija')
plt.legend()
plt.show()

colors = np.where(resultforest_df['real_points'] == np.minimum(resultforest_df['real_points'], resultforest_df['points']), 'orange', 'blue')
plt.figure(figsize=(10, 6))
plt.barh(resultforest_df['surname'],np.minimum(resultforest_df['real_points'], resultforest_df['points']), color=colors, label='Stvarni poeni')
plt.barh(resultforest_df['surname'], abs(resultforest_df['real_points'] - resultforest_df['points']),left=np.minimum(resultforest_df['real_points'], resultforest_df['points']), color=np.where(colors == 'orange', 'blue', 'orange'), label='Predviđeni poeni')
plt.xlabel('Vozači')
plt.ylabel('Poeni')
plt.title('Predviđeni vs. Stvarni poeni za svakog vozača Forest')
plt.legend()
plt.show()

colors = np.where(result_df_lasso['real_points'] == np.minimum(result_df_lasso['real_points'], result_df_lasso['points']), 'orange', 'blue')
plt.figure(figsize=(10, 6))
plt.barh(result_df_lasso['surname'], np.minimum(result_df_lasso['real_points'], result_df_lasso['points']), color=colors, label='Stvarni poeni')
plt.barh(result_df_lasso['surname'], abs(result_df_lasso['real_points'] - result_df_lasso['points']),left=np.minimum(result_df_lasso['real_points'], result_df_lasso['points']),color=np.where(colors == 'orange', 'blue', 'orange'), label='Predviđeni poeni')
plt.xlabel('Vozači')
plt.ylabel('Poeni')
plt.title('Predviđeni vs. Stvarni poeni za svakog vozača Lasso')
plt.legend()
plt.show()

colors = np.where(result_df_ridge['real_points'] == np.minimum(result_df_ridge['real_points'], result_df_ridge['points']), 'orange', 'blue')
plt.figure(figsize=(10, 6))
plt.barh(result_df_ridge['surname'], np.minimum(result_df_ridge['real_points'], result_df_ridge['points']), color=colors, label='Stvarni poeni')
plt.barh(result_df_ridge['surname'], abs(result_df_ridge['real_points'] - result_df_ridge['points']),left=np.minimum(result_df_ridge['real_points'], result_df_ridge['points']), color=np.where(colors == 'orange', 'blue', 'orange'), label='Predviđeni poeni')
plt.xlabel('Vozači')
plt.ylabel('Poeni')
plt.title('Predviđeni vs. Stvarni poeni za svakog vozača Ridge')
plt.legend()
plt.show()
#---------------------------------------------------------------------
#GRAFICKI PRIKAZ IMPORTANCA
width = 0.2 
feature_names = X_train1.columns
plt.barh(feature_names, regresijacoef, tick_label=feature_names)
plt.ylabel('Atributi')
plt.xlabel('Značajnost atributa')
plt.title('Značajnost atributa za Linearnu Regresiju')
plt.subplots_adjust(left=0.3)
plt.show()

plt.barh(range(len(importancesrfc)), importancesrfc, tick_label=feature_names)
plt.ylabel('Atributi')
plt.xlabel('Značajnost atributa')
plt.title('Značajnost atributa za Random Forest Classifier')
plt.subplots_adjust(left=0.3)
plt.show()

plt.barh(range(len(importancesrfr)), importancesrfr, tick_label=feature_names)
plt.ylabel('Atributi')
plt.xlabel('Značajnost atributa')
plt.title('Značajnost atributa za Random Forest Regresor')
plt.subplots_adjust(left=0.3)
plt.show()

plt.barh(range(len(importancesros)), importancesros, tick_label=feature_names)
plt.ylabel('Atributi')
plt.xlabel('Značajnost atributa')
plt.title('Značajnost atributa za Ros')
plt.subplots_adjust(left=0.3)
plt.show()

plt.barh(range(len(lassocoef)), lassocoef, tick_label=feature_names)
plt.ylabel('Atributi')
plt.xlabel('Značajnost atributa')
plt.title('Značajnost atributa za Lasso')
plt.subplots_adjust(left=0.3)
plt.show()

plt.barh(range(len(ridgecoef)), ridgecoef, tick_label=feature_names)
plt.ylabel('Atributi')
plt.xlabel('Značajnost atributa')
plt.title('Značajnost atributa za Ridge')
plt.subplots_adjust(left=0.3)
plt.show()

plt.barh(range(len(elasticoef)), elasticoef, tick_label=feature_names)
plt.ylabel('Atributi')
plt.xlabel('Značajnost atributa')
plt.title('Značajnost atributa za Elastic Net')
plt.subplots_adjust(left=0.3)
plt.show()

feature_names = X_train6.columns
plt.barh(range(len(importancessafety)), importancessafety, tick_label=feature_names)
plt.ylabel('Atributi')
plt.xlabel('Značajnost atributa')
plt.title('Značajnost atributa za Random Forest Classifier za Predikciju Safety car-a')
plt.subplots_adjust(left=0.3)
plt.show()

feature_names = X_train.columns
plt.barh(range(len(importancespitstop)), importancespitstop, tick_label=feature_names)
plt.ylabel('Atributi')
plt.xlabel('Značajnost atributa')
plt.title('Značajnost atributa za Random Forest Regresion za Predikciju promene guma')
plt.subplots_adjust(left=0.3)
plt.show()