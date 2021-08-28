# Advanced Analytics and Application 2021 - Anacondas

Advanced Analyitcs and Application 2021 team assignement by team "Anacondas"

This project provides analysis of taxi demand patterns in the City of Chicago in 2015 to provide valuable information for the market entry of an all electric ridehailing service of an german car manufacturer. 

## Group

* Nina Annika Erlacher
* Simon Maximilian Wolf
* Christian Bergen
* Nico Liedmeyer 
* Sepehr Salemirad

## Setup

We recommend to setup a virtual environment or conda environment with python version 3.8 and using the provided requirements.txt to install all necessary packages.

Most of the used data is to large to upload on GitHub. Therefore you have to donwload the required datasets from:

- Chicago 2015 parquet file: [Download](https://filedn.eu/lvIIS1QB2KmSUjz5Gvx9LYb/Taxi_Trips.parquet)
- Weather 2015 parquet file: [Download](https://filedn.eu/lvIIS1QB2KmSUjz5Gvx9LYb/Weather.parquet)
- Cleaned Chicagoe dataset: [Download](https://filedn.eu/lvIIS1QB2KmSUjz5Gvx9LYb/Taxi_Trips_Cleaned.parquet)
- 15 Percent Sample Data (Clustering): [Download](https://filedn.eu/lvIIS1QB2KmSUjz5Gvx9LYb/chicago_taxi_trips_15percent_sample.parquet)


Put the datasets into /data to run our notebooks.

    
## Notebook Description

### Task 1 - Data Collection and Preperation

* **[interactive_statistics](notebooks/interactive_statistics.ipynb):** Dataset exploration using naive, preimplemented data validation tools. 
* **[Cleaning](notebooks/Cleaning.ipynb):** Removing invalid data and outliers from raw trip dataset.
* **[h3](notebooks/h3.ipynb):** Presentation of h3-uber library

*Require datasets [Chicago 2015](https://filedn.eu/lvIIS1QB2KmSUjz5Gvx9LYb/Taxi_Trips.parquet) and [Weather 2015](https://filedn.eu/lvIIS1QB2KmSUjz5Gvx9LYb/Weather.parquet)*

### Task 2 - Descriptive (Spatial) Analytics

Analysis of demand patterns of taxi trips with focus on diffrent temporal-spatial resolutions:

* **[Eda_Total_Amount](notebooks/Eda_Total_Amount.ipynb):** Total fare amount of taxi trips.
* **[eda_idle_time](notebooks/eda_idle_time.ipynb):** Idle time of taxis between trips.
* **[eda_location](notebooks/eda_location.ipynb):** Most popular locations based on starting and ending taxi trips.
* **[eda_starttime](notebooks/eda_starttime.ipynb):** Starting time of taxi trips.
* **[eda_trip_length](notebooks/eda_trip_length.ipynb):** Duration of taxi trips in chicago.

 *Require dataset [Cleaned Chicago](https://filedn.eu/lvIIS1QB2KmSUjz5Gvx9LYb/Taxi_Trips_Cleaned.parquet)*
 
 ### Task 3 - Cluster Analysis

* **[clustering](notebooks/clustering.ipynb):** Soft- and Hard-Clustering approaches to find prevalent usage/trip clusters.

 *Require dataset [15 Percent Sample Data](https://filedn.eu/lvIIS1QB2KmSUjz5Gvx9LYb/chicago_taxi_trips_15percent_sample.parquet)*

### Task 4 - Predictiv Analytics

Predicting taxi demand in diffrent spatia-temporal resolutions using two diffrent trained and tuned prediction models: 

* **[prediction_svm](notebooks/prediction_svm.ipynb):** Support Vector Machine
* **[prediction_nn](notebooks/prediction_nn.ipynb):** Neural Network

 *Require dataset [Cleaned Chicago](https://filedn.eu/lvIIS1QB2KmSUjz5Gvx9LYb/Taxi_Trips_Cleaned.parquet)*


### Task 5 - Optimal Allocation of Chargin Stations 

* **[optimization](notebooks/optimization.ipynb):** Formulation and solving of an mathematical optimization problem to allocate the optimal locations for charging stations in chicago.

 *Require dataset [Cleaned Chicago](https://filedn.eu/lvIIS1QB2KmSUjz5Gvx9LYb/Taxi_Trips_Cleaned.parquet)*


![Anacondas](https://miro.medium.com/max/1000/1*h1h6j08kIHv3ywN3MCfKLg.jpeg)
