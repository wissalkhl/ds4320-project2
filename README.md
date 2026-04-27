# DS 4320 Project 2: Predicting Swing State Election Outcomes from Economic Indicators

This repository contains a MongoDB-backed data science project analyzing whether economic indicators like unemployment rate can predict presidential election outcomes in key swing states. The project includes a data creation pipeline, MongoDB Atlas database, machine learning model, and visualizations.

**Name:** Wissal Khlouf  
**NetID:** hta4yb  
**DOI:** 10.5281/zenodo.19824789  
**Press Release:** [press_release.md](./press_release.md)  
**Pipeline:** [pipeline.ipynb](./pipeline.ipynb)  
**License:** MIT — [LICENSE](./LICENSE)  

## Problem Definition

**General Problem:** Predicting election results.

**Specific Problem:** Can economic indicators such as unemployment rate predict 
presidential election outcomes in swing states?

### Motivation
The fate of presidential elections often rests on just a few crucial states. But it remains difficult to predict how swing states will vote. One hypothesis is that the economy directly affects how people vote — if the economy is in bad shape, the party in power will lose votes. If we can identify which economic predictors correlate most with swing state performance, the press, candidates, and citizens can have better expectations for election outcomes.

### Rationale for Refinement
Predicting elections in general is too broad — national popular vote forecasts are very different from state-level outcomes. Swing states are the most important ones to focus on as they have the highest uncertainty, making them the most interesting to model. Limiting our scope to economics keeps our feature set small and rooted in established political science theory.

### Press Release
[Can GDP and Unemployment Predict Who Wins Swing States?](./press_release.md)

## Domain Exposition

### Terminology

| Term | Definition |
|---|---|
| Swing state | A state where both parties have a realistic chance of winning |
| GDP growth | Percentage change in Gross Domestic Product |
| Unemployment rate | Percentage of labor force actively seeking work |
| Median household income | Middle value of all household incomes in an area |
| Incumbent | The party currently holding the presidency |
| Two-party vote share | Percentage of votes split between only Democrat and Republican |

### Domain Overview
This project exists at the nexus of political science and data science. Prediction of elections has been studied for decades. Economists and political scientists have created models to predict elections using measurements of economic fundamentals — the theory being that voters reward or punish the party in power based on their personal economic wellbeing. MongoDB lends itself well to this project because we are combining economic and election data from multiple disparate sources with different structures.

### Background Reading

| Title | Description | Link |
|---|---|---|
| How Does Economic Performance Affect US Presidential Elections? | Academic study examining how GDP, unemployment, inflation, and stock market performance affected election outcomes from 1948–2020 | [Link](./background_reading/economic_performance.pdf) |
| How the Economy Is Doing in the Swing States | Time Magazine analysis of economic conditions in the 7 key swing states during the 2024 election cycle | [Link](./background_reading/How_the_Economy_is_Doing_in_the_Swing_States.pdf) |
| Predicting How Swing Voters in Battleground Counties Will Vote | Oxford Economics county-level forecast showing how unemployment and income affect swing state outcomes | [Link](./background_reading/Predicting-how-swing-voters-in-battleground-US-counties.pdf) |
| Can Economic Indicators Or Betting Markets Predict Election Results? | Analysis of the Misery Index, GDP growth, and stock market as predictors of incumbent party success | [Link](./background_reading/Can_Economic_Indicators_Or_Betting_Markets_Predict_Election_Results_-_Articles_-_Advisor_Perspectives.pdf) |
| Prediction Markets + Polls + Economic Indicators: Better Election Forecasting? | UCLA Anderson research combining polling, economic data, and prediction markets for improved forecasting | [Link](./background_reading/Prediction_Markets___Polls___Economic_Indicators__Better_Election_Forecasting__-_UCLA_Anderson_Review.pdf) |

## Data Creation

### Provenance
Election data was sourced from the MIT Election Data and Science Lab's U.S. President 1976–2020 dataset, which compiles official state-level vote totals from the U.S. House Clerk's office. Unemployment data was sourced from the Federal Reserve Economic Data (FRED) database, specifically the monthly state unemployment rate series for each of the seven swing states (PA, MI, WI, AZ, NV, GA, NC). Both datasets are publicly available and well maintained.

The data creation script merges these two sources by matching election year and state, computing two-party vote shares, and embedding the October/November average unemployment rate for each state in each election year into a single MongoDB document per state per election cycle.

### Code

| File | Description |
|---|---|
| [data/create_data.py](./data/create_data.py) | Loads CSVs, merges election and unemployment data, builds documents, inserts into MongoDB Atlas |

### Critical Decisions and Uncertainty
The unemployment rate used is the average of October and November of the election year, representing economic conditions closest to Election Day. This is a judgment call, using the full-year average would smooth out short-term fluctuations but may be less representative of voter sentiment at the time of voting. Using only October/November introduces uncertainty if those months are anomalous relative to the broader year.

### Bias Identification
The dataset only includes two major parties (Democrat and Republican), excluding third party candidates. This introduces a simplification bias. Additionally, swing state selection is based on modern political classifications, so states like Georgia and Arizona were not considered swing states in earlier decades, which may introduce selection bias.

### Bias Mitigation
Third party votes are excluded from two-party vote share calculations, which is standard practice in political science for this type of analysis. The limitation is noted in the pipeline. Future versions could include third party vote share as a separate feature.

## Metadata

### Implicit Schema
Each document in the MongoDB collection represents one swing state in one presidential election year. Every document is expected to contain the following fields: year (integer), state name (string), state postal abbreviation (string), raw vote totals for each major party (integer), total votes cast (integer), two-party vote shares (float), two-party margin (float), winner (string), and the unemployment rate averaged over October and November of that election year (float). The data_sources field is a list of strings identifying provenance. All fields are expected to be present, there are no optional fields in this schema.

### Data Summary

| Property | Value |
|---|---|
| Total documents | 84 |
| States covered | PA, MI, WI, AZ, NV, GA, NC |
| Election years | 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020 |
| Database | MongoDB Atlas |
| Collection | election_economics |

### Data Dictionary

| Field | Type | Description | Example | Uncertainty |
|---|---|---|---|---|
| year | int | Presidential election year | 2020 | None — exact |
| state | string | Full state name in uppercase | PENNSYLVANIA | None |
| state_po | string | Two-letter postal abbreviation | PA | None |
| democrat_votes | int | Raw votes cast for Democrat candidate | 2328677 | Minor reporting delays in some states |
| republican_votes | int | Raw votes cast for Republican candidate | 2205604 | Minor reporting delays in some states |
| total_votes | int | Total votes cast in that state that year | 4620787 | Excludes third party votes in margin calc |
| democrat_vote_share | float | Democrat votes as % of total | 50.40 | ±0.01% rounding |
| republican_vote_share | float | Republican votes as % of total | 47.73 | ±0.01% rounding |
| two_party_margin | float | Democrat share minus Republican share | 2.67 | ±0.02% rounding |
| winner | string | Winning party in that state that year | DEMOCRAT | None — derived from vote totals |
| unemployment_rate_election_season | float | Avg unemployment rate Oct–Nov of election year | 7.95 | ±0.1% BLS measurement error |
| data_sources | list | List of data source names | [MIT Election Lab, FRED] | None |