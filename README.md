# Clustering-Players 
Overcome Classic Roles in Football using Dimensional Reduction plus Clustering algorithms to find Style of Playing for players.

## Project Goal (Classic Roles vs Styles Roles)
In modern Football defining players through **Classic Roles** such as striker, midfielder, full-back, etc. is limiting because the evolution of the game has produced players who have tasks on the field rather than roles.

This generates players who, even with the same position covered on the field, have completely different ways of interpreting the game and with different functions.

What has just been described is the definition of a player's playing style, which I define as **Styles Roles**.

That is, the tasks, functions and characteristics that each player has beyond the position on the field covered.

**This is the principle on which this project is based:**

Starting from the Raw Data of the 2015/2016 season for the Top 5 European leagues (Italy, England, Germany, France and Spain) released for free by Statsbomb, it was decided to calculate several advanced statistics for each player and use them to cluster them in different groups no longer based on their role, but on their playing style.

## Development and Realization.

### Statistics calculation:

As anticipated, starting from the Raw Data released for free by Statsbomb for the Top 5 European leagues of the 2015/2016 season, more than 300 advanced statistics, representatives of the entire season, were calculated for each player employed in that season and for each Team.

**To give an overview some of the statistics calculated are as follows:**

1) Progressive Passes.

2) Deep Progression.

3) Aggression.

4) Box Cross %.

5) Total xG.

6) Tackle/Dribbled_Past %.

7) Touches In Box.

8) Shot Touch %.

9) Shot Stopping %.

### Statistics Normalization:

Once the statistics were calculated, a normalization process was performed.

Each cumulative statistic (so no statistic derived from a ratio or product of others) was normalized for the **Average Possession** held by the Team to which the player belonged using the **formula used by Statsbomb** shown below.

![possession_adjustment](https://github.com/user-attachments/assets/a7bbd7c9-b7bc-4fe0-97e7-fd5f5be2d17c)

_Sigmoid function for statistics (x in the formula) normalization with respect to possession (possession in the formula) attributing everything to 50 (0.5 in the formula) which would be the match-by-match average_.

Finally, the normalized statistics were normalized again for 90 minutes.

After these two normalizations, players who played a different amount of minutes in the season and in teams with different levels and styles of play with the ball are comparable.

### Player Filtering:

At this point, the dataset was filtered by excluding the Teams and keeping only the players who played at least 900 minutes in the season (10 games in total) in order to have a statistical sample representative of the style and what the various players did on the pitch.

### Statistics Filtering:

Finally, only the statistics that were representative of a player's playing style were kept, excluding all those that indicated their effectiveness, such as **% Passes Completed** or **% Tackles Won**.

All statistics exclusive to goalkeepers were also excluded.

In this way, we went from more than 300 statistics to 67.

### Role Aggregation:

Since Statsbomb does not use the classic separation of roles (Forward, Midfielder, Defender, Full Back, etc.) to assign to players, but has a more granular division of roles, as shown in the image below derived from the Statsbomb Datasheet, it was decided to aggregate them in the classic roles according to the following logic:

**Logic of Role Aggregation:**

* 'Goalkeeper': 'GK'.
* 'Full Back': 'RB', 'LB', 'RWB', 'LWB'.
* 'Center Back': 'CB', 'RCB', 'LCB'.
* 'Defensive Midfielder': 'CDM', 'LDM', 'RDM'.
* 'Central Midfielder': 'CM', 'LCM', 'RCM'.
* 'External Midfielder': 'LM', 'RM'.
* 'Attacking Midfielder': 'CAM', 'LAM', 'RAM'.
* 'Winger': 'LW', 'RW'.
* 'Forward': 'CF', 'LCF', 'RCF', 'SS'.

**Statsbomb Roles:**

_Statsbomb Roles Arrangement on the Field._

![statsbomb role pitch](https://github.com/user-attachments/assets/4084b675-c918-4280-9bff-c119bcf1d57e)

_Statsbomb Role Description Table._

![statsbomb role table 1](https://github.com/user-attachments/assets/9c1069ca-0d31-4959-8f22-3d254a4ab62e)

![statsbomb role table](https://github.com/user-attachments/assets/07fdd546-7194-4ac1-a473-4d36958841fb)

**Role frequency Selection:**

During the season and in the same game players play multiple roles so statsbomb in its Event data for the same player assigns multiple roles.

To decide which role to assign to each player, the number of times he played a role in the season was calculated and finally the one with the highest frequency was assigned to him.

### Dimensional Reduction (UMAP):

To aggregate players with similar playing styles based on advanced statistics, a dimensionality reduction technique called UMAP was chosen.

** UMAP (Uniform Manifold Approximation and Projection)** is a dimensionality reduction algorithm used to project high-dimensional data into a lower-dimensional space (usually 2D or 3D), while maintaining as much of the local and global structure of the data as possible.

**Local Structure:**

* Points that are close together in the original space (similar to each other) are also projected close together in the reduced space.

* This allows us to identify coherent clusters: the points within a cluster in the projection are truly similar in the original space.

**Global Structure:**

* UMAP does a good job of maintaining relationships between clusters: if two clusters are far apart in the original space, it tends to keep them far apart in the map as well.

This allows you to view in 2D or 3D similar players, close points, otherwise not easily identifiable unless you compare each statistic.

**UMAP needs several parameters.**

**1) n-neighbours:** Defines how much "local structure" to consider.

What it does:

* Determines how many neighbors to use to build the local graph around each point.

* Controls how "local" or "global" the analysis is:

  * low values ​​5-10 leads the algorithm to focus on the local structure.
  
  * High values ​​40-50 leads it to focus on the global one.

**2) min_dist :** Defines how "tight" the clusters are.

What it does:

* Influences how much points can be compressed together in the final representation.

* Control the visual "shape" and "density" of the groups:
  
  * Values ​​between  0.1 – 0.3: Dense and compact clusters, more "separated".
 
  * Values ​​between 0.4–0.8: More relaxed, less separated, more nuanced clusters.

**In summary:**

Low n-neighbors and m_dist: Very compact and separate clusters. Great for highlighting subgroups.

High n-neighbors and m_dist: More continuous, less segmented map. Useful for seeing transitions or gradualities in data.

In our case we are interested in seeing players with similar styles, so getting compact and separate clusters was the goal so small values ​​were selected:

* n_neighbours: 10
* min_dist: 0.1

**UMAP with Classic Roles:**

The following graph shows the result of the UMAP with the colors positioned based on the classic role associated with each player:

![Classic_Roles_UMAP](https://github.com/user-attachments/assets/6cb459af-26af-4f21-9a52-748f7703e8f3)

From the color scheme, you can see that the central defenders and the full-backs have two completely separate clusters.

You can see that some full-backs are in the defenders cluster and some defenders in the full-backs cluster and that some defensive midfielders are in the central defenders cluster.

They are probably adapted players or with different playing styles from their classic role.

The goalkeepers cluster is also separated from everything else as you would expect, as there are no specific statistics for them, and therefore all the statistics of the outfield players are much less marked except, obviously, those relating to the throws made and the average distance of the passes due to the goalkeepers' tendency to throw long and to kick from the bottom.

The reason why some goalkeepers distance themselves from the main cluster does not come from a different style, but from values ​​of throws made and average distance of the passes different from the other goalkeepers.

This gives more of a vision of the team's style than of the player himself.

The other roles, however, tend to be arranged in a more logical way.

The midfielders are close to each other with the offensive ones closer to the strikers and wingers, who mix well with the external midfielders, while the defensive ones are closer to the central defenders.

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise):

At this point we try to group the players based on their proximity, playing style, in the UMAP.

To do this we need to use a clustering algorithm on the reduced data.

The choice ended up on DBSCAN which is a density-based clustering algorithm, which allows to identify groups of dense points separated by less dense regions (noise or outliers).

DBSCAN groups points that are close together if they are in high-density areas and ignores (labels as "noise") isolated points.

**DBSCAN also has some adjustable parameters:**

* **eps:** radius (maximum distance) to consider a point as "close".
  
  The value of eps is calculated using the elbow point method which works as follows.

  **1. Calculate the distances of the neighbors.**
  
    For each point in the dataset, calculate the distance to its k-th neighbor.

   **2. Sort the distances.**
   
     Sort these distances in ascending order.
     
     You get a curve that grows slowly at first (points in dense regions), then rapidly (isolated points).

 
   **3. Find the "elbow" of the curve.**
   
     Plot the ordered distances.
     
     Find the point where the curve has a significant inflection, that is, where it starts to rise more steeply.
     
     This point is called the "elbow," and is a good candidate for ε.

   
  The method works because the **elbow** value allows you to define the point at which points go from being **neighbors**, and therefore part of the cluster, to **noise**.
  
  ![image](https://github.com/user-attachments/assets/25626134-83a3-4b2e-84c2-d246467810de)
  
  _Plot of distance with elbow value used in this project_

* **min_samples:** minimum number of points required within eps to consider a dense area.

**DBSCAN works by classifying points into 3 categories:**

* **Core point:**
A point that has at least min_samples points (including itself) in its eps radius.

It is at the center of a dense region.

* **Border point:**
It does not have enough points in its eps, but is close to a core point.

It is at the edge of a cluster.

* **Noise point (outlier):**
  
It is neither a core nor a border.

It is isolated, it does not belong to any cluster.

**The procedure by which DBSCAN finds clusters is as follows:**

1) For each point, check its neighbors within radius eps.

2) If the point is a core point, create (or expand) a cluster.

3) Include neighbors (core and border) in the cluster.

4) Continue until all points are labeled or marked as noise.


**The advantages of this algorithm are:**

1) It does not require specifying the number of clusters a priori (like K-means).

2) It can find clusters of arbitrary shape (not only spherical).

3) It is robust to noise and outliers.


In the following graph you can see how the clusters found by **DBSCAN** and projected onto **UMAP** appear:

![Raw_Roles_UMAP](https://github.com/user-attachments/assets/de9dd276-2f0c-4426-a40c-01bec3735476)

#### Players Profilation:

**DBSCAN Cluster specification:**

The label -1 represents the noise cluster, where players with unique styles that cannot be grouped into any cluster end up.

This group is colored in light blue.

I have inserted two arrows on points of this group with the relative names of the players (Leonardo Bonucci and Arjen Robben) to show an example of what type of players end up in this group (We will come back to this later).

All the other labels represent clusters of different shapes and sizes compared to how those of the classic roles appeared, a symptom that players of the same role are in different groups and vice versa.

**Cluster Analysis:**

At this point, understanding what led the algorithm to create these clusters will tell us the playing style of the players in them.

To do this, we will use statistics, in this case the Z-Score.

The z-score is a statistical measure that indicates how many standard deviations a given value is from the mean of a set of data.

![z-score](https://github.com/user-attachments/assets/b1d27ba6-efaf-4a52-b8cf-1403591e479d)

**𝑥** = data value.

**𝜇** = sample or population mean.

**𝜎** = standard deviation.

With this value we can understand how much the statistics of each cluster differ from the others.

If a statistic has Z-score:

= 0, the value is exactly equal to the mean.

= ± 1, the value is one standard deviation above or below the mean (different).

= ± 2, the value is two standard deviations above or below the mean (unusually different).

= ± 3, the value is two standard deviations above or below the mean (Extremely different).

**Analyzing statistics with Z-Scores that have an absolute value >=1 allows us to understand what characterizes the playing style of the players in each cluster:**

The following image now shows the clusters, previously defined by numbers, with names that represent the playing style of the players within them.

A table gives an overview of why those names.

![Styles_Roles_UMAP](https://github.com/user-attachments/assets/f92d644f-faa5-4c4d-8d13-4efb5ce6db18)

![Cluster Table Description](https://github.com/user-attachments/assets/2e99c7d0-de34-4ebb-b8e6-c7a0cdd91e14)

A special mention goes to Lorenzo Insigne and Dries Mertens who share the cluster with Messi and Neymar.

This cluster includes players who have offensive statistics, both in finishing and play creation, that are out of scale compared to the other players taken into consideration.

This cluster has in fact been called **"Self-Sufficient Offensive Players"**.

This indicates that these two players have had an extraordinary year at the level of the best, certainly helped by a coach like Maurizio Sarri who has been able to enhance their talent with magnificent football leading Napoli to earn 91 points in the championship, almost managing to win the Serie A.

### Case studies of two outliers:

As anticipated, the analysis is shown that allows you to find players with unique styles inserted into the noise because of this.

**1) Leonardo Bonucci**

![Bonucci vs Man-Marking Defenders](https://github.com/user-attachments/assets/342a3375-221a-447b-ae8c-6f5b9e92452d)

Bonucci, defined as central defender in classic roles definition, is an outlier in the central defenders cluster as shown by his position in the UMAP.

To understand what makes him different from players similar to him, the Z-Score of his parameters was calculated compared to that of the central defenders cluster.

As you might imagine given Bonucci's reputation, what differentiates him is his game with the ball, which sees him statistically superior in through balls, xG produced by placed defenders and in other statistics related to ball management.

This is a clear example of how the role does not indicate the style.

Despite covering the same role as his BBC teammates (Barzagli and Chiellini) he has a completely different way of interpreting it.

**2) Arjen Robben:**

![Robben Vs Chance-Creators](https://github.com/user-attachments/assets/f7a06178-bcea-4473-8c58-5576bbb532f6)

![Robben vs Finalizers](https://github.com/user-attachments/assets/3927f8c3-6dd3-4c92-bd8d-bd98c0e11736)

Robben's case, defined as winger in classic roles definition, is slightly different.

If Bonucci was an outlier attributable to a single cluster, Robben is an intracluster one, therefore straddling two clusters (In this case among Chance-Creator and Finalizers).

From the two Z-Score tables you can see how compared to the Chance-Creators he is more of an attacker as he has clearly higher values ​​than them for shots, xG in Open Play and touches in the area.

While on the contrary he is more of a creator of play compared to the cluster of pure finishers having higher values ​​in the statistics regarding the ball's rise in dangerous areas.

This is the classic example of a versatile and complete offensive player capable of making the difference in various aspects of the game exactly like a phenomenon like Robben.






