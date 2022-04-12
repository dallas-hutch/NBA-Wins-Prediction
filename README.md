
# Predicting NBA Wins after All-Star Break

The post All-Star stretch is arguably one of the most important
 parts of an NBA Season. This is the time that each team has their
 final roster after the NBA trade deadline. Each team is trying to
  fight for playoff positioning, or “fighting” for the worst 
  record and draft lottery positioning. The goal for this project
  is to predict how many wins each NBA team will have during this
  stretch and ultimately, where each team will land in the
  standings.


## Table of Contents
[Authors](#authors)

[Introduction](#introduction)

[Part 1. Web Scraping](#1-web-scraping)

[Part 2. Data Cleaning](#2-data-cleaning)

[Part 3. Exploratory Data Analysis](#3-exploratory-data-analysis)

[Part 4. Regression Modeling](#4-regression-modeling)

[Findings](#findings)

## Authors

- [@Dallas Hutchinson](https://github.com/dallas-hutch)
- [@Andrew Marion](https://github.com/andrewmarion)

## Introduction
Today, 02/24/2022, the end of the All-Star Break, we would like
to build a regression model to predict how many wins each team
will have to end the 2021-2022 NBA season. Teams go through ups
and downs over the course of a long NBA season so we want to
utilize both season-wide statistics and team performance heading
into the break as model predictors. This project was handled in four
parts: web scraping, data cleaning, EDA, and
regression modeling. 
## 1. Web Scraping

We want to pull data from the past 10 NBA seasons, starting from 
the 2012–13 season until the current one. The [stats.nba.com](https://www.nba.com/stats/)
page has team statistics across each season along with filters to slice 
the results how you wish. In order to scrape this page, we need to
identify the NBA API endpoint. This [article](https://jedong.medium.com/using-python-to-scrape-nba-individual-player-stats-in-less-than-20-lines-44b149e21434) 
provides a great tutorial on how to find the endpoint within the HTML code 
and see the row set parameters. The function below makes a request 
to the NBA API for a specific endpoint based on the filters given, 
stores the table result set and then combines each into a data frame. 
A “season” column is also added to keep track of which NBA season 
each table is coming from.
```python
def pull_stats(seasons, datefrom, measuretype, segment, numcol, columns):
    dfs = []
    for season, date in zip(seasons, datefrom):
        url = "https://stats.nba.com/stats/leaguedashteamstats?Conference=&DateFrom={}&DateTo=&Division=&GameScope=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType={}&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season={}&SeasonSegment={}&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision="
        res = requests.get(url=url.format(date, measuretype, season, segment), headers=headers)
        res.raise_for_status()
        data = res.json()['resultSets'][0]['rowSet']
        extract = [data[i][0:numcol+1] for i in range(len(data))]
        df = pd.DataFrame(extract, columns=columns)
        df['Season'] = season
        dfs.append(df)
    return pd.concat(dfs, sort=False)
```
The first slice of team data we want is Four Factors stats for each 
team Pre All-Star break. Four Factors stats are derived metrics 
focused on measuring a team’s strengths and weaknesses related to 
shooting, turnovers, free throws, and rebounding.
```python
datefrom = [''] * 10
columns_ff = ["TeamID","Team","GP","W","L","WIN%","MIN","EFG%","FTA RATE","TOV%","OREB%","OPP EFG%","OPP FTA RATE","OPP TOV%","OPP OREB%"]
ff_stats = pull_stats(seasons, datefrom, 'Four+Factors', 'Pre+All-Star', 14, columns_ff)
```
![alt text](https://github.com/dallas-hutch/NBA-Wins-Prediction/blob/main/images/ff_table.png)

This looks right, 30 teams x 10 NBA seasons = 300 rows with the 
columns labelled appropriately. The next slice of data will try to 
capture how well a team is performing leading into the All-Star 
break. We will try to grab the ~15 games of metrics per team based 
on the date of the All-Star break that particular season. If a team 
is surging (or tanking) headed into the break, that will definitely 
influence how a team may perform post-break.
```python
last_15 = pull_stats(seasons, as_break_dates, 'Advanced', 'Pre+All-Star', 12, columns_last15)
```
![alt text](https://github.com/dallas-hutch/NBA-Wins-Prediction/blob/main/images/last15_table.png)

The “last_15” table looks to check out too. The final slice of data 
that’s needed is our actual target or outcome variable. So far we 
have just scraped pre All-Star break data. But we are trying to 
predict post All-Star break wins. Let’s pull some basic post 
All-Star break stats including wins for the past 10 seasons 
(minus the ongoing season).
```python
post_as = pull_stats(seasons[0:9], datefrom, 'Base', 'Post+All-Star', 5, columns_post_as)
```
![alt text](https://github.com/dallas-hutch/NBA-Wins-Prediction/blob/main/images/post_as_table.png)

Looks good! 270 rows makes sense since the 2021–2022 season does 
not have any post All-Star break data yet. Let’s save these three 
data frames as csvs to use in the data cleaning and analysis portion. 
We now have two tables regarding pre All-Star stats, one with 
season-to-break four factors metrics and the other with metrics for 
the 15-ish games heading into break. The third table has post All-Star 
break metrics from which we will extract wins to use in our 
regression model.
## 2. Data Cleaning

We will first read in our three csv files as data frames using 
pandas. For each data frame we will check and validate the column 
data types, null values in each column, and season counts. 
Everything seems to look as expected. Then, we drop unnecessary 
columns and rename a few others for more clear interpretation.
```python
ff = pd.read_csv("four_factors.csv")
last_15 = pd.read_csv("last_15.csv")
post_as = pd.read_csv("post_as.csv")

# Check data types, null values, Season counts
last_15.dtypes
last_15.isnull().sum()
last_15['Season'].value_counts()

# Drop + rename columns
last_15.rename(columns={'GP': 'GP_15', 'W': 'W_15', 'L': 'L_15', 'WIN%': 'WIN%_15'}, inplace=True)
last_15.drop(columns=['TeamID', 'MIN', 'EOFFRTG', 'EDEFRTG', 'ENETRTG'], inplace=True)
```
To combine the tables, we’ll use pd.merge() twice utilizing an 
inner join on the “Team” and “Season” columns. Before the second 
merge, the last 30 rows of the data frame corresponding to the 
current 2021–2022 season will be split off and saved as itself. 
This “curr_season” data frame will be used at the end to make 
predictions on how many wins teams this season will have.
```python
# Merge all dfs, split current NBA season off as that will be used as a test case
season_stats = pd.merge(ff, last_15, how='inner', on=['Team','Season'])
curr_season = season_stats.iloc[-30:]
past_seasons = season_stats.iloc[:-30]
past_season_stats = pd.merge(past_seasons, post_as, how='inner', on=['Team', 'Season'])
past_season_stats.to_csv('past_season_stats.csv', index=False)
curr_season.to_csv('curr_season_stats.csv', index=False)
```
## 3. Exploratory Data Analysis

When looking at our data, we noticed many variables that were highly 
correlated. We decided to drop Number of Wins (W), Number of 
Losses (L), Minutes Played (MIN), Number of Wins in the most 
recent 15 games (W_15), Number of Losses in the most recent 15 
games (L_15) from the model, as they were too highly correlated 
with other variables in the model.
![correlation matrix](https://github.com/dallas-hutch/NBA-Wins-Prediction/blob/main/images/corr_matrix.png)
<p align = "center">
Figure 1. Pearson's correlation matrix
</p>

Looking at the distribution of the remaining variables, no variables 
stood out enough to be removed from the model. However, we found 
that the previous two seasons (2019–2020 and 2020–2021) season 
added variation in the data that may cause an issue with our 
predictions, as they both had unusual game schedules due to the 
COVID-19 pandemic.
![alt text](https://github.com/dallas-hutch/NBA-Wins-Prediction/blob/main/images/pairplot.png)
<p align = "center">
Figure 2. Four Factors data pairplots
</p>

![alt text](https://github.com/dallas-hutch/NBA-Wins-Prediction/blob/main/images/pairplot2.png)
<p align = "center">
Figure 3. Last 15 data pairplots
</p>

Notice the linear relationships between predictors/outcome variables. Time to do some modeling!
## 4. Regression Modeling

We decided to use a Partial Least Squares Regression, as it is 
especially useful when your predictors are highly collinear. In our 
case, many predictors we wanted to use were highly correlated.

__Predictor Variables__: Games Played (GP), Win Percentage (WIN%), 
Expected Field Goal Percentage (EFG%), Free Throw Rate (FTA RATE), 
Turnover Percentage (TOV%), Offensive Rebounding Percentage (OREB%), 
Opponent Expected Field Goal Percentage (OPP EFG%), 
Opponent Free Throw Rate (OPP FTA RATE), Opponent 
Turnover Percentage (OPP TOV%), Opponent Offensive Rebounding 
Percentage (OPP OREB%), Games Played for the previous 
month (GP_15), Win Percentage for the previous month (WIN%_15), 
Offensive Rating for the previous month (OFFRTG_15), 
Defensive Rating for the previous month (DEFRTG_15), 
Net Rating for the previous month (NETRTG_15)

__Outcome Variable__: Wins Post All-Star Break (W_Post)

Our model was built using Partial Least Squares Regression and 
Cross Validation to make the best fitting model. Next, we optimized 
our model by testing the number of variables in our model to 
minimize the Mean Squared Error (MSE) and maximize the R Squared 
(R2). Our plots of the MSE and R2 are used to check that our model 
is working effectively. Finally, we plotted our regression line vs 
the expected regression line (y vs y) to compare our model to the 
theoretical best model.

**Partial Least Squares using the past 9 seasons:**

In the model, the optimal number of variables left in the model was 
found to be only 3 variables, leaving us with an R2 = 0.5113 and 
a MSE = 14.9085. This means that based on our model, predictions 
will be off on average 3.86 wins.
![alt text](https://github.com/dallas-hutch/NBA-Wins-Prediction/blob/main/images/r2_mse_plots.png)
![alt text](https://github.com/dallas-hutch/NBA-Wins-Prediction/blob/main/images/r2_mse_rpd_plot.png)
<p align = "center">
Figure 4. Nine NBA season model evaluation
</p>

**Partial Least Squares NOT using the previous 2 seasons 
(2019–2020 and 2020–2021):**

As we noted earlier, there are two seasons affected by the COVID-19 
pandemic, 2019–2020 and 2020–2021. We decided to remove those 
seasons from the model to see if they are affecting it.
In the model, the optimal number of variables left in the model was 
found to be 6 variables, leaving us with a R2 = 0.5782 and a 
MSE = 10.5266. Based on this model, predictions will be off on 
average 3.24 wins.
![alt text](https://github.com/dallas-hutch/NBA-Wins-Prediction/blob/main/images/optimize_r2_mse.png)
![alt text](https://github.com/dallas-hutch/NBA-Wins-Prediction/blob/main/images/optimize_r2_mse_rpd.png)
<p align = "center">
Figure 5. Seven NBA season model evaluation
</p>

## Findings

Not using the previous 2 seasons (2019–2020 and 2020–2021) 
helped improve our model, as this year is expected to be more of a 
typical year in terms of games played after the All-Star Break. 
With this PLS model trained on 7 NBA seasons of data, we are 
ideally able to predict post All-Star break wins within ~3 wins. 
Let’s try it out.

**Final 2021-2022 Season Projections:**

![alt text](https://github.com/dallas-hutch/NBA-Wins-Prediction/blob/main/images/final_predictions.png)

__Key__: Team (NBA Team Name), G_Left (number of games left in the 
season), Pred_Wins (teams predicted number of wins in their 
remaining games), Pred_Win% (teams predicted win percentage in 
their remaining games), Pred_Total_W (teams current wins + predicted 
wins)

Thank you for reading!
