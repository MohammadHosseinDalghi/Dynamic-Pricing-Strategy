# Dynamic Pricing for Ride-Sharing Services

## Overview

This project implements a dynamic pricing strategy for ride-sharing services based on supply and demand conditions. It utilizes machine learning techniques to predict ride costs, considering factors such as the number of riders, number of drivers, vehicle type, and expected ride duration.

## Features

- Exploratory Data Analysis (EDA) using Seaborn and Matplotlib
- Data preprocessing including encoding categorical variables, handling outliers, and scaling features
- Implementation of dynamic pricing based on demand-supply multipliers
- Machine learning model (Random Forest Regressor) for price prediction
- Performance evaluation through visualization and comparison of actual vs. predicted values

## Dataset

The dataset used (`dynamic_pricing.csv`) contains the following features:

- `Number_of_Riders`: Total number of passengers requesting rides
- `Number_of_Drivers`: Total number of available drivers
- `Vehicle_Type`: Type of vehicle (Premium or Economy)
- `Expected_Ride_Duration`: Estimated duration of the ride (in minutes)
- `Historical_Cost_of_Ride`: Previous ride cost

## Dynamic Pricing Strategy

The project adjusts ride costs based on demand and supply conditions:

1. **Demand Multiplier**: If the number of riders is above the 75th percentile, the value is divided by the 75th percentile. If below, it is divided by the 25th percentile.
2. **Supply Multiplier**: If the number of drivers is above the 25th percentile, the adjustment factor is computed as (75th percentile / number of drivers). Otherwise, (25th percentile / number of drivers) is used.
3. **Adjusted Ride Cost**: Calculated by multiplying the historical ride cost with demand and supply multipliers.

## Model Training

The dataset is split into training and testing sets (80/20). A Random Forest Regressor is trained using the following input features:

- `Number_of_Riders`
- `Number_of_Drivers`
- `Vehicle_Type` (encoded as 1 for Premium, 0 for Economy)
- `Expected_Ride_Duration`

The target variable is `adjusted_ride_cost`.

## Prediction Function

The model can predict ride costs using the function:

```python
predict_price(number_of_riders, number_of_drivers, vehicle_type, Expected_Ride_Duration)
```

Example usage:

```python
predicted_price = predict_price(50, 25, "Economy", 30)
print("Predicted Price:", predicted_price)
```

## Visualizations

- **Regression Plot**: Relationship between ride duration and ride cost
- **Box Plot**: Distribution of historical ride costs by vehicle type
- **Correlation Matrix**: Heatmap of feature correlations
- **Pie Chart**: Distribution of profitable vs. loss rides
- **Actual vs. Predicted Values**: Regression plot comparing model predictions

## Installation & Usage

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/dynamic-pricing-ride-sharing.git
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the script:
   ```sh
   python dynamic_pricing.py
   ```

## Dependencies

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## License

This project is licensed under the MIT License.

## Author

[Your Name]

