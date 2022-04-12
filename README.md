
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
