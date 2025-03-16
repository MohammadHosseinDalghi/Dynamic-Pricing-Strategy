# Add Necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Adding a dataset
df = pd.read_csv('dynamic_pricing.csv')

# print(df.head())
# print(df.info())

# relationship between expected ride duration and the historical cost of the ride
sns.regplot(
    data=df,
    x='Expected_Ride_Duration',
    y='Historical_Cost_of_Ride',
    line_kws=dict(color='C1'),
)
plt.title('Ride Duration Vs. Cost of Ride');
# plt.show()
# plt.close()

# distribution of the historical cost of rides based on the vehicle type
sns.boxplot(
    data=df,
    x='Vehicle_Type',
    y='Historical_Cost_of_Ride'
)
plt.title('Historical cost of ride by vehicle type');
# plt.show()
# plt.close()

# correlation matrix
df_encoded = df.copy()

for col in df.select_dtypes(include='object').columns:
    df_encoded[col] = LabelEncoder().fit_transform(df[col])

df_corr = df_encoded.corr()

sns.heatmap(data=df_corr)
plt.title('Correlation Matrix');
# plt.show()
# plt.close()

'''
If the number of passengers is greater than the 75% percentile:
we divide it by the 75th percentile.
If it is less, we divide it by the 25th percentile.
'''

high_demand_percentile = np.percentile(df['Number_of_Riders'], 75)
low_demand_percentile = np.percentile(df['Number_of_Riders'], 25)

df['demand_multiplier'] = np.where(
    df['Number_of_Riders'] > high_demand_percentile,
    df['Number_of_Riders'] / high_demand_percentile,
    df['Number_of_Riders'] / low_demand_percentile)

'''
If the number of drivers is greater than the 25th percentile:
We calculate the multiplication as (75th percentile divided by the number of drivers). 
Otherwise, we use the value (25th percentile divided by the number of drivers).
'''

high_supply_percentile = np.percentile(df['Number_of_Drivers'], 75)
low_supply_percentile = np.percentile(df['Number_of_Drivers'], 25)

df['supply_multiplier'] = np.where(
    df['Number_of_Drivers'] > low_supply_percentile,
    high_supply_percentile / df['Number_of_Drivers'],
    low_supply_percentile / df['Number_of_Drivers'])

'''
If demand is high and supply is low, the cost of travel increases,
If demand is low and supply is high, the cost of travel decreases.
'''

df['adjusted_ride_cost'] = df['Historical_Cost_of_Ride'] * (
    np.maximum(df['demand_multiplier'], 0.8) *
    np.maximum(df['supply_multiplier'], 0.8)
)

# Calculate the profit percentage for each ride
df['profit_percentage'] = ((df['adjusted_ride_cost'] - df['Historical_Cost_of_Ride']) / df['Historical_Cost_of_Ride']) * 100

# Identify profitable rides where profit percentage is positive
profitable_rides = df[df['profit_percentage'] > 0]

# Identify loss rides where profit percentage is negative
loss_rides = df[df['profit_percentage'] < 0]

profitable_count = len(profitable_rides)
loss_count = len(loss_rides)

# Create a pie chart to show the distribution of profitable and loss rides
values = [profitable_count, loss_count]
labels = ['Profitable Rides', 'Loss Rides']

fig, ax = plt.subplots()
ax.pie(
    x=values,
    labels=labels,
    autopct='%1.1f%%',
    radius=1.2,
    
    
)
plt.title('Expected Ride Duration vs. Cost of Ride');
# plt.show()
# plt.close()

# relationship between the expected ride duration and the cost of the ride based on the dynamic pricing strategy
sns.regplot(
    data=df,
    x='Expected_Ride_Duration',
    y='adjusted_ride_cost',
    line_kws=dict(color='C1')
)
plt.title('Expected Ride Duration vs. Cost of Ride');
# plt.show()
# plt.close()

def data_preprocessing_pipline(df):

    numeric_features = df.select_dtypes(include=['float', 'int']).columns
    categorical_features = df.select_dtypes(include='object').columns

    '''
    In this dataset we do not have missing values,
    but if we had them in another dataset,
    we can handle missing values ​​with this command.
    '''
    # df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())

    # detect and handle outliers in numeric features using IQR
    for feature in numeric_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        df[feature] = np.where(
            (df[feature] < lower_bound) | (df[feature] > upper_bound),
            df[feature].mean(),
            df[feature]
        )

    #Handle missing values in categorical features
    df[categorical_features] = df[categorical_features].fillna(df[categorical_features].mode().iloc[0])

    return df

df['Vehicle_Type'] = df['Vehicle_Type'].map({
    "Premium": 1,
    'Economy': 0
})

x = np.array(df[['Number_of_Riders', 'Number_of_Drivers', 'Vehicle_Type', 'Expected_Ride_Duration']])
y = np.array(df[['adjusted_ride_cost']])

X_train, X_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Reshape y to 1D array
y_train = y_train.ravel()
y_test = y_test.ravel()

model = RandomForestRegressor()
model.fit(X_train, y_train)

def get_vehicle_type_numeric(vehicle_type):
    vehicle_type_mapping = {
        'Premium': 1,
        'Economy': 0
    }

    vehicle_type_numeric = vehicle_type_mapping.get(vehicle_type)
    return vehicle_type_numeric

def predict_price(number_of_riders, number_of_drivers, vehicle_type, Expected_Ride_Duration):
    vehicle_type_numeric = get_vehicle_type_numeric(vehicle_type)
    if vehicle_type_numeric is None:
        raise ValueError("Invalid vehicle type")
    
    input_data = np.array([[number_of_riders, number_of_drivers, vehicle_type_numeric, Expected_Ride_Duration]])
    predicted_price = model.predict(input_data)
    return predicted_price

# Example prediction using user input values
user_number_of_riders = 50
user_number_of_drivers = 25
user_vehicle_type = "Economy"
Expected_Ride_Duration = 30
predicted_price = predict_price(user_number_of_riders, user_number_of_drivers, user_vehicle_type, Expected_Ride_Duration)
print('predicted price:', predicted_price)

# comparison of the actual and predicted results
y_pred = model.predict(X_test)

y_test_flat = y_test.flatten()
y_pred_flat = y_pred.flatten()

plt.figure(figsize=(8, 6))
sns.regplot(
    x=y_test_flat,
    y=y_pred_flat,
    scatter_kws={'alpha': 0.7},
    line_kws={'color': 'red', 'linestyle': 'dashed'},
    ci=None
)

plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values');
# plt.show()
# plt.close()