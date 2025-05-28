# Clustering-Players
Overcome Classic Roles in Football using Dimensional Reduction plus Clustering algorithms to find Style of Playing for players.

## Project Goal
In modern Football defining players through classic roles such as striker, midfielder, full-back, etc. is limiting because the evolution of the game has produced players who have tasks on the field rather than roles.

This generates players who, even with the same position covered on the field, have completely different ways of interpreting the game and with different functions.

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

![statsbomb role pitch](https://github.com/user-attachments/assets/4084b675-c918-4280-9bff-c119bcf1d57e)

![statsbomb role table 1](https://github.com/user-attachments/assets/9c1069ca-0d31-4959-8f22-3d254a4ab62e)

![statsbomb role table](https://github.com/user-attachments/assets/07fdd546-7194-4ac1-a473-4d36958841fb)









