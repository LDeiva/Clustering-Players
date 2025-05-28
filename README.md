# Clustering-Players
Overcome Classic Roles in Football using Dimensional Reduction plus Clustering algorithms to find Style of Playing for players.

## Project Goal
In modern Football defining players through classic roles such as striker, midfielder, full-back, etc. is limiting because the evolution of the game has produced players who have tasks on the field rather than roles.

This generates players who, even with the same position covered on the field, have completely different ways of interpreting the game and with different functions.

**This is the principle on which this project is based:**

Starting from the Raw Data of the 2015/2016 season for the Top 5 European leagues (Italy, England, Germany, France and Spain) released for free by Statsbomb, it was decided to calculate several advanced statistics for each player and use them to cluster them in different groups no longer based on their role, but on their playing style.

## Development and Realization

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

Once the statistics were calculated, a normalization process was performed.

Each cumulative statistic (so no statistic derived from a ratio or product of others) was normalized for the **Average Possession** held by the Team to which the player belonged using the **formula used by Statsbomb** shown below.

![possession_adjustment](https://github.com/user-attachments/assets/a7bbd7c9-b7bc-4fe0-97e7-fd5f5be2d17c)

_Sigmoid function for statistics (x in the formula) normalization with respect to possession (possession in the formula) attributing everything to 50 (0.5 in the formula) which would be the match-by-match average_.









