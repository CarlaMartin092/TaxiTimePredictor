# TaxiTimePredictor ✈️

Predicts the time laps between the departure of a plane from the dock and its take-off time.

To run the package:

```python main.py --mode "test" --model "FRBS"```

The package predicts the taxi time using airport data and weather data. We implemented 3 models: a 60-days moving average, a multilinear regression and a Fuzzy Rule-Based System.

The performances of the models is measured using 3 metrics. The R2, to assess the amount of variance that is captured by the model. The Mean Absolute Error allows us to know how many seconds away our prediction is from ground truth on average. Similarly, we implemented a metric we called accuracy, which returns the percentage of predictions that are within a specified number of minutes of the actual taxi times. 
