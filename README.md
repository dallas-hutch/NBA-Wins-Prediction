
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

<script src="https://gist.github.com/dallas-hutch/5475ee6c0ccc56d0221c12c934a87637.js"></script>

The first slice of team data we want is Four Factors stats for each 
team Pre All-Star break. Four Factors stats are derived metrics 
focused on measuring a team’s strengths and weaknesses related to 
shooting, turnovers, free throws, and rebounding.

<script src="https://gist.github.com/dallas-hutch/b3f4bb600c74b923d8bf6e77a01a05ea.js"></script>

![alt text](https://github.com/dallas-hutch/NBA-Wins-Prediction/blob/main/images/ff_table.png)

This looks right, 30 teams x 10 NBA seasons = 300 rows with the 
columns labelled appropriately. The next slice of data will try to 
capture how well a team is performing leading into the All-Star 
break. We will try to grab the ~15 games of metrics per team based 
on the date of the All-Star break that particular season. If a team 
is surging (or tanking) headed into the break, that will definitely 
influence how a team may perform post-break.

<script src="https://gist.github.com/dallas-hutch/b7222652b798e95141f9766ae4ea4e59.js"></script>

<insert last15_table image>

The “last_15” table looks to check out too. The final slice of data 
that’s needed is our actual target or outcome variable. So far we 
have just scraped pre All-Star break data. But we are trying to 
predict post All-Star break wins. Let’s pull some basic post 
All-Star break stats including wins for the past 10 seasons 
(minus the ongoing season).

<script src="https://gist.github.com/dallas-hutch/a2bb5e710541aa8bf180b161944dc1ce.js"></script>

<insert post_as_table image>

Looks good! 270 rows makes sense since the 2021–2022 season does 
not have any post All-Star break data yet. Let’s save these three 
data frames as csvs to use in the data cleaning and analysis portion. 
We now have two tables regarding pre All-Star stats, one with 
season-to-break four factors metrics and the other with metrics for 
the 15-ish games heading into break. The third table has post All-Star 
break metrics from which we will extract wins to use in our 
regression model.
