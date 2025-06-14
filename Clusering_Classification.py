# -*- coding: utf-8 -*-
"""
Created on Wed May  7 23:14:56 2025

@author: david
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.colors import to_rgba
from statsbombpy import sb
from mplsoccer import Pitch, FontManager,VerticalPitch, pitch
from mplsoccer.statsbomb import * 
#from mplsoccer.statsbomb import read_event, EVENT_SLUG
import seaborn as sns
from matplotlib.cm import get_cmap
import pandas as pd
import os
from pandas import json_normalize
import re
from function_for_clustering import *
import ast

"""A) Import Data"""
Players_Statistics=pd.read_excel(r'C:\Users\david\OneDrive\Football Analytics\Calcio\Dati e Progetti\Miei Progetti\Clustering_Teams_and_Players\Statistiche cumulate e normalizzate\Players\Full_Players_Norm_Stats.xlsx')

"""B) Create column with Most frequency Role and convert Statsbomb Role in Classic Role"""

"""Create columns with most frequency role for every plyer"""
#Convert in list
Players_Statistics['Position_Abbreviation']= Players_Statistics['Position_Abbreviation'].apply(ast.literal_eval)
Players_Statistics['Position_Frequency']= Players_Statistics['Position_Frequency'].apply(ast.literal_eval)

#Create columns
Players_Statistics['Position_Frequency_Max'] = Players_Statistics.apply(keep_max_element, axis=1)

# Change position of column
col_data = Players_Statistics.pop('Position_Frequency_Max') 
Players_Statistics.insert(3, 'Position_Frequency_Max', col_data)

"""Aggregate Statsbomb role in generical role"""
position_map = {
    'GK': 'Goalkeeper',
    'RB': 'Full Back', 'LB': 'Full Back', 'RWB': 'Full Back', 'LWB': 'Full Back',
    'CB': 'Center Back', 'RCB': 'Center Back', 'LCB': 'Center Back',
    'CDM': 'Defensive Midfielder','LDM': 'Defensive Midfielder','RDM': 'Defensive Midfielder',
    'CM': 'Central Midfielder', 'LCM': 'Central Midfielder', 'RCM': 'Central Midfielder',
    'LM': 'External Midfielder', 'RM': 'External Midfielder',
    'CAM': 'Attacking Midfielder','LAM': 'Attacking Midfielder','RAM': 'Attacking Midfielder',
    'LW': 'Winger', 'RW': 'Winger',
    'CF': 'Forward', 'LCF': 'Forward', 'RCF': 'Forward', 'SS': 'Forward'
}

#Map Classic Role
Players_Statistics['Classic_Role'] = Players_Statistics['Position_Frequency_Max'].map(position_map)

# Change position of column
col_data2 = Players_Statistics.pop('Classic_Role') 
Players_Statistics.insert(4, 'Classic_Role', col_data2)



"""C) Select only Most Important Features"""
#Create columns vriable
columns=list(Players_Statistics.columns)

#Select only Stats that define style of play.
Players_Statistics_features_filt=Players_Statistics[['Player_Name','Classic_Role','Team','AdjP90opXG',
                                                                       'opXGs','AdjP90spXG','spXGs','AdjP90npShots','Mean_Shots_Lenght',
                                                                       'AdjP90Foot_Shots','AdjP90Head_Shots',
                                                                       'AdjP90Dribling_Shots','AdjP90npGol','Shot_Touch_%',
                                                                       'Average_Passes_Distance','AdjP90Attempted_Passes_Under_Pressure',
                                                                       'Being_Pressured_Change_in_Pass','AdjP90Passes_in_Final_Third',
                                                                       '%_of_Forward_Passes_in_Final_Third',
                                                                       'AdjP90Attempted_Progressive_Passes',
                                                                       'AdjP90Attempted_Short_Passes','AdjP90Attempted_Middle_Passes','AdjP90Attempted_Long_Passes','AdjP90Attempted_Long_Passes_Underpressure',
                                                                       'AdjP90Attempted_Throw','AdjP90Key_Passes','AdjP90Through_Ball',
                                                                       'AdjP90Corners','AdjP90Attempted_Cross','AdjP90Passes_Into_Box','AdjP90Cross_Into_Box',
                                                                       'Box_Cross_%','AdjP90Passes_Inside_Box','AdjP90Deep_Pass_Completion','AdjP90Deep_Cross_Completion','AdjP90Attempted_Goal_Kick','AdjP90Ball_Receipt_Under_Pressure','AdjP90Pass_into_Danger',
                                                                       'AdjP90Touches','AdjP90Touches_in_Final_Third','AdjP90Touches_in_Box','AdjP90Attempted_Dribbling',
                                                                       'AdjP90Progressive_Carries_Number','Mean_Progressive_Carries_Distance','AdjP903/4_Carries',
                                                                       'AdjP90Inside_area_Carries','AdjP90Deep_Progressions','AdjP90Turnover','AdjP90Foul Won','AdjP90Pressures','Mean_Pressures_Height','AdjP90Conterpressures',
                                                                       'AdjP90Final_third_Balls_Recovery','Mean_Recovery_Height','AdjP90Pressure_Regains','AdjP90Aggression','AdjP90Tackles_Attempted','AdjP90Tackles_Attempted_in_Area','AdjP90Total_Aereal_Duel','AdjP90Total_Aereal_Duel_in_Area',
                                                                       'Dribble_Stopped_%','AdjP90Interceptions','AdjP90Clearance',
                                                                       'AdjP90Fouls_committed','AdjP90Outside_Area_Defensive_Actions','AdjP90Inside_Area_Defensive_Actions',
                                                                       'Times']]

#Select columns filt
columns_filt=list(Players_Statistics_features_filt.columns)

#find columns unselected
columns_unselected=set(columns)-set(columns_filt)


#Fill nan with zero
Players_Statistics_features_filt=Players_Statistics_features_filt.fillna(0)


"""D) Filt for Played Time"""
Players_Statistics_features_final=Players_Statistics_features_filt[Players_Statistics_features_filt['Times']>=900]

"""Set players name like a index"""
Players_Statistics_features_final.index=Players_Statistics_features_final['Player_Name']


"""E) Normalize and calculate VIF values for multicollinearity"""

from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from sklearn.preprocessing import StandardScaler

"""Create only features Dataframe"""
Players_Statistics_only_features=Players_Statistics_features_final.drop(["Player_Name", "Classic_Role", "Team",'Times'], axis=1)


"""Calculation"""
#Standard scaler
Players_Statistics_only_features_standaraize = StandardScaler().fit_transform(Players_Statistics_only_features)

#Vif values calculation
vif_data = pd.DataFrame()
vif_data["feature"] = Players_Statistics_only_features.columns
vif_data["VIF"] = [variance_inflation_factor(Players_Statistics_only_features_standaraize, i) for i in range(Players_Statistics_only_features_standaraize.shape[1])]




"""F) RUN UMAP"""
import umap

"""Test different parameters for umap to select the best"""
n_neighbors=[ 30, 40, 50]
min_dist= [0.2, 0.3, 0.4]

for n in n_neighbors:
    for m in min_dist:
        reducer = umap.UMAP(n_neighbors=n, min_dist=m, n_components=2, random_state=42)
        X_umap = reducer.fit_transform(Players_Statistics_only_features_standaraize)

        #Show UMAP
        umap_df = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])

        #Add index and role columns
        umap_df.index=Players_Statistics_only_features.index
        umap_df['Role']=Players_Statistics_features_final['Classic_Role']

        # Plot
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Role', palette='tab10', s=50)
        plt.title('UMAP ')
        plt.legend(title=f'Umap with {n} neighbors and {m} min dist')
        plt.show()


        


"""Run final UMAP with best parameters selected from for cicle"""
#RUN UMAP
n=10
m_d=0.1
reducer = umap.UMAP(n_neighbors=n, min_dist=m_d, n_components=2, random_state=42)
X_umap = reducer.fit_transform(Players_Statistics_only_features_standaraize)

#Show UMAP
umap_df = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'])

#Add index and role columns
umap_df.index=Players_Statistics_only_features.index
umap_df['Role']=Players_Statistics_features_final['Classic_Role']

# Plot
plt.figure(figsize=(10, 7))
sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Role', palette='tab10', s=30)
    
plt.title('UMAP \n Classic Roles',fontsize=16)
plt.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,fontsize=10)
plt.tight_layout()  # Per evitare tagli nel grafico)
plt.show()



"""G) RUN DBSCAN"""
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

#Calculation of Elbow Plot
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_umap)
distances, indices = neighbors_fit.kneighbors(X_umap)

# # Sort the distance to 5° neighbour
k_distances = np.sort(distances[:, -1])  # Last column = 10° Neighbor

# Sort the distance to 5° neighbor
distances = np.sort(distances[:, 4])
plt.plot(distances)
plt.ylabel("Distance to 5° neighbor")
plt.xlabel("Index")
plt.title("K-distance Graph (10-nearest neighbors)")
plt.grid()
plt.show()

"""Elbow calculation"""
x = np.arange(len(k_distances))
y = k_distances

# Segmento tra il primo e l'ultimo punto
start = np.array([x[0], y[0]])
end = np.array([x[-1], y[-1]])

# Distanze dei punti dalla linea (prodotto vettoriale)
line_vec = end - start
line_vec_norm = line_vec / np.linalg.norm(line_vec)

vec_from_start = np.stack([x, y], axis=1) - start
scalar_proj = np.dot(vec_from_start, line_vec_norm)
proj = np.outer(scalar_proj, line_vec_norm)
distance_to_line = np.linalg.norm(vec_from_start - proj, axis=1)

# Trova l'indice con distanza massima → gomito
elbow_idx = np.argmax(distance_to_line)
elbow_eps = y[elbow_idx]

plt.plot(y)
plt.scatter(elbow_idx, y[elbow_idx], color='red', label=f"Gomito (eps={elbow_eps:.2f})")
plt.legend()
plt.grid(True)
plt.title("k-distance plot con gomito automatico")
plt.show()

print(f"Valore suggerito di eps: {elbow_eps:.2f}")


"""Start DBSCAN"""
clustering = DBSCAN(eps=0.223, min_samples=5)
clusters = clustering.fit_predict(X_umap)

#Insert new role
umap_df['Cluster_Role']=clusters

# Salva i colori per ogni label
umap_df['color'] = umap_df['Cluster_Role'].map(label_color_map)

# Mappa colori: cluster_name → colore
palette_dict = dict(zip(
    umap_df['Cluster_Role'].unique(),
    umap_df.groupby('Cluster_Role')['color'].first()
))

#Visualize clusters with new role
points_to_highlights = ['Leonardo Bonucci', 'Arjen Robben']

# Plot
plt.figure(figsize=(10, 7))
sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Cluster_Role', palette=label_to_color , s=30)
# Add the Arrows for outiliers players
for idx in points_to_highlights:
    x = umap_df.loc[idx, 'UMAP1']
    y = umap_df.loc[idx, 'UMAP2']
    
    # Position of Arrow
    x_base = x - 1.5
    y_base = y + 4

    # Arrow
    plt.arrow(x_base, y_base, x - x_base, y - y_base,
              head_width=0.1, length_includes_head=True, color='blue')

    # Label of player name close to arrow bottom
    plt.text(x_base, y_base - 0.2, idx, color='blue', fontsize=12)
    
plt.title('UMAP + DBSCAN \n Raw Cluster Roles ',fontsize=16)
plt.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,fontsize=10)
plt.tight_layout()
plt.show()



"""PLOT SINGLE CLUSTER"""
clusters_unique=list(set(clusters))

for c_r in clusters_unique:

    umap_cluster_filt_df=umap_df[umap_df['Cluster_Role']==c_r]
    # Plot
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', s=30)
    plt.scatter( umap_cluster_filt_df['UMAP1'], umap_cluster_filt_df['UMAP2'], color='r', s=30)
    plt.title(f'UMAP cluster {c_r}')
    plt.show()


"""H) CLUSTER ANALYSIS WITH Z-SCORE"""

#ADD DBSCAN cluster labels to original df
Players_Statistics_features_final_dbscancluster = Players_Statistics_features_final.join(umap_df['Cluster_Role'])

#Z-Score for every cluster respect the mean of dataset

#Find cluster
#Define container for every z-score of every cluster.
cluster_zscore=[]
#Define container for players list in every cluster
cluster_players=[]
#New Cluster classic role frequency
classic_role_frequency_list=[]

#start cicle for every cluster
for cluster_id in clusters_unique:
    # Cluster target 
    #cluster_id = 9
    if cluster_id!=-1:
        
        cluster_data_full = Players_Statistics_features_final_dbscancluster[Players_Statistics_features_final_dbscancluster['Cluster_Role'] == cluster_id]
                
        #Calculation for every cluster 
        classic_role_frequency=cluster_data_full["Classic_Role"].value_counts()
        classic_role_frequency_list.append(classic_role_frequency)
        
        cluster_data = cluster_data_full.drop(["Player_Name", "Classic_Role", "Team",'Times','Cluster_Role'], axis=1)
        
        #Delate cluster role
        Players_Statistics_features_final_dbscancluster_dropped = Players_Statistics_features_final_dbscancluster.drop(["Player_Name", "Classic_Role", "Team",'Times','Cluster_Role'], axis=1)
        
        # Calcolo media e std globali
        global_mean = Players_Statistics_features_final_dbscancluster_dropped.mean()
        global_std = Players_Statistics_features_final_dbscancluster_dropped.std()
        
        
        # Media delle feature nel cluster
        cluster_mean = cluster_data.mean()
        
        # Z-score tra media cluster e media globale
        cluster_zscores = (cluster_mean - global_mean) / global_std
        
        # Ordina per impatto
        cluster_zscores.sort_values(key=abs, ascending=False).head(15)
        
        #insert in container z-score
        cluster_zscore.append(cluster_zscores)
        
        #Insert in container players cluster list
        cluster_players.append(cluster_data_full)

    else:
        continue


#Select noise cluster
cluster_data_noise_full = Players_Statistics_features_final_dbscancluster[Players_Statistics_features_final_dbscancluster['Cluster_Role'] == -1]

"""I) MAP DBSCAN Cluster with New Cluster name."""

"""Create New Cluster Name"""
# Dizionario per mappare i valori a nomi
new_cluster_name_map = {
    -1: 'Unique Style Players',
    0: 'Finalizers',
    1: 'Aggressive Interdictors',
    2: 'Man-Marking Defenders',
    3: 'Goalkeeper',
    4: 'Chance-Creators',
    5: 'Box-to-Box Wide Players',
    6: 'Playmakers',
    7: 'Defensive Carriers',
    8: 'Goalkeeper',
    9: 'Self-Sufficient Offensive Players',
    10: 'Self-Made Strikers',
    11: 'Goalkeeper',
    12: 'Creative Shooters',
    13: 'Dribblers'
}

# Create new cluster name mapping DBSCAN label with new role name
Players_Statistics_features_final_dbscancluster['New Cluster Name'] = Players_Statistics_features_final_dbscancluster['Cluster_Role'].map(new_cluster_name_map)


"""VISUALIZE CLUSTER WITH NEW CLUSTER NAME"""
#Insert new role
umap_df['New Cluster Role']= umap_df['Cluster_Role'].map(new_cluster_name_map)

label_to_color = dict(
    umap_df.drop_duplicates('Cluster_Role')[['Cluster_Role', 'color']].values
)

label_to_name = dict(
    umap_df.drop_duplicates('Cluster_Role')[['Cluster_Role', 'New Cluster Role']].values
)

# Costruisci mappa finale: nome cluster → colore
name_to_color = {
    label_to_name[label]: color
    for label, color in label_to_color.items()
}

#Visualize clusters with new role
points_to_highlights = ['Leonardo Bonucci', 'Arjen Robben']

# Plot
plt.figure(figsize=(12, 7))
sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='New Cluster Role', palette=name_to_color, s=30)
# Add the Arrows for outiliers players
for idx in points_to_highlights:
    x = umap_df.loc[idx, 'UMAP1']
    y = umap_df.loc[idx, 'UMAP2']
    
    # Position of Arrow
    x_base = x - 3.1
    y_base = y + 4

    # Arrrow
    plt.arrow(x_base, y_base, x - x_base, y - y_base,
              head_width=0.1, length_includes_head=True, color='blue')

    # Label of player name close to arrow bottom
    plt.text(x_base, y_base - 0.2, idx, color='blue', fontsize=12)
    
plt.title('UMAP + DBSCAN \n Styles Roles',fontsize=16)
plt.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,fontsize=10)
plt.tight_layout()
plt.show()


"""L) CREATE RESUME TABLE WITH CLUSTER DESCRIPTION"""
#DICT with INFO
data = {
    "Cluster": [-1,0, 1, 2,[3, 8, 11],4,5,6,7,9,10,12,13],
    "Label": ["Unique Style Players", "Finalizers", "Aggressive Interdictors","Man-Marking Defenders","Goalkeeper",
              "Chance-Creators","Box-to-Box Wide Players","Playmakers","Defensive Carriers","Self-Sufficient Offensive Players","Self-Made Strikers","Creative Shooters","Dribblers"],
    "Prototype Players": ["Bonucci, Robben, Khedira", "Lewandowski, Müller, Ronaldo", "Kanté, Nainggolan, Vidal", "Chiellini, Barzagli, Ramos", "Buffon, Neuer, Courtois", "Çalhanoğlu, De Bruyne, Özil", "Lahm, Alaba, Lichtsteiner", "Thiago Alcântara, Xabier Alonso, Tony Kross", "Guarín, Soriano", "Messi, Neymar, Insigne, Mertens", "Ibrahimović, Harry Kane, Bale", "Iličić, Coutinho, Berardi", "Sadio Mané"],
    "Key Features": ["It depends on the proximity to the cluster of which they are outliers",
        "Shot Touch %, Touches in Box, opXG",
        "Aggression, Pressure Regains, Tackle Attempted",
        "Clearance, Inside Area Defensive Action, Total Areal Duel in Area",
        "Attempted Goal Kick, Average Passes Distance, Attempted Throw",
        "Deep Pass Completition, Key Passes, Progressive Carriers Number",
        "Box Cross %, Deep cross Completition, Dribble Stopped %",
        "Deep Pass Completition, Passes into Box, Through Ball",
        "Aggression, Attempted Dribbling, Progressive Carriers Number",
        "Through Ball, npGoal, Attempted Dribbling",
        "Dribbling Shots, Head Shots, Shot Touch %",
        "Dribbling Shots, Deep Pass Completition, Attempted Dribbling",
        "Attempted Dribbling, Turnover, Inside Area Carriers"
    ],
    "Description": [
        "Players not belonging to any cluster.",
        "Offensive players who are very good at finishing the action.",
        "Players who are skilled at completing the opponent's action and recovering balls.",
        "Classic goal-scoring defender.",
        "Goalkeeper",
        "Players capable of creating goal-scoring opportunities through dribbling, passing and ball control.",
        "External players employed in both the defensive and offensive phases.",
        "Game creators.","Players capable of breaking the opponent's action and transforming it into an offensive action.",
        "Self-sufficient players who create play and finish on their own, out of scale with the others.",
        "Forwards capable of creating goal-scoring opportunities on their own.",
        "Offensive players capable of scoring and finishing the action.",
        "Players who dribble continuously."
    ]
}

#Make table a dataframe
df_description = pd.DataFrame(data)

"""Create Table Plot"""
#Define plot
fig, ax = plt.subplots(figsize=(16, 4))  # allarga per testi lunghi
ax.axis('off')

# Create table
table = ax.table(
    cellText=df_description.values,
    colLabels=df_description.columns,
    cellLoc='left',
    colLoc='left',
    loc='center'
)

# Format
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2)  # scala colonne e righe

#Plot table
plt.tight_layout()
plt.show()

"""M) PLAYERS PROFILATION"""

"""Single player dot in UMAP PLOT"""
#Define coordinate of player
umap_player_cord_df=umap_df.loc['Leonardo Bonucci']

#Plot All
plt.figure(figsize=(10, 7))
sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', s=30)
plt.scatter( umap_player_cord_df['UMAP1'], umap_player_cord_df['UMAP2'], color='r', s=30)
plt.title(f'UMAP Robben Position')
plt.show()


"""Single player Z-score calculation with similar cluster of players.
Bonucci Example"""

#ADD DBSCAN cluster labels to original df
Players_Statistics_features_final_dbscancluster = Players_Statistics_features_final.join(umap_df['Cluster_Role'])

#Check Bonucci outlier reasons

#Select players in DBSCAN cluster 0.
dc_cluster = Players_Statistics_features_final_dbscancluster[(Players_Statistics_features_final_dbscancluster["Cluster_Role"] == 2)]
dc_cluster = dc_cluster.drop(["Player_Name", "Classic_Role", "Team",'Times','Cluster_Role'], axis=1)
#Scaled
scaler = StandardScaler()
dc_scaled = scaler.fit_transform(dc_cluster)


#Select Bonucci Features
Bonucci_cluster = pd.DataFrame(Players_Statistics_features_final_dbscancluster.loc['Leonardo Bonucci']).T
Bonucci_cluster = Bonucci_cluster.drop(["Player_Name", "Classic_Role", "Team",'Times','Cluster_Role'], axis=1)
#Scaled
bonucci_scaled = scaler.transform(Bonucci_cluster)

#Z-score cluster
# Differenza in z-score
z_diff = abs(bonucci_scaled[0] - dc_scaled.mean(axis=0))
z_diff_df = pd.Series(z_diff, index=dc_cluster.columns).sort_values(ascending=False)

print(z_diff_df.head(10))  # Le feature che lo rendono un outlier

z_diff_df=pd.DataFrame(z_diff_df,columns=['Leonardo Bonucci vs Man-Marking Defenders'])
z_diff_df.index.name='TOP 10 Stats Z-Score'
"""Creation of RadarPlot with the mean of discriminant Features for Robben"""

#A) For outliers Player
#Get top 10 discriminant features between outliers players and closest cluster
params=z_diff_df.index[:10].to_list()
#Get values
outliers_players_values=list(Bonucci_cluster[params].iloc[0].values)



