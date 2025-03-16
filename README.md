# 🚖 **Dynamic Pricing Model for Ride-Sharing**  

## 📌 **Project Overview**
This project implements a **dynamic pricing model** for ride-sharing platforms using **Python** and **machine learning**. The model predicts ride costs based on **demand and supply fluctuations** using **Random Forest Regression**.

---

## 📊 **Dataset**
The dataset (`dynamic_pricing.csv`) contains ride details, including:
- `Number_of_Riders` - The number of passengers requesting a ride.
- `Number_of_Drivers` - The available drivers in the area.
- `Vehicle_Type` - Type of vehicle (Economy or Premium).
- `Expected_Ride_Duration` - Estimated ride duration in minutes.
- `Historical_Cost_of_Ride` - Past ride cost.

---

## 💡 **Pricing Strategy**
The pricing model follows these key steps:
1. **Demand Multiplier**: Adjusts ride costs based on rider demand.
2. **Supply Multiplier**: Modifies pricing based on driver availability.
3. **Final Cost Calculation**:
   - If **demand is high & supply is low**, prices increase 📈
   - If **demand is low & supply is high**, prices decrease 📉

---

## ⚙️ **Installation**
To run this project locally, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/dynamic-pricing-model.git

# Navigate to the project directory
cd dynamic-pricing-model

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 **Usage**
### 1️⃣ **Run the Model**
```bash
python dynamic_pricing.py
```
### 2️⃣ **Predict Ride Cost**
You can use the model to predict ride costs:
```python
from dynamic_pricing import predict_price

predicted_price = predict_price(50, 25, "Economy", 30)
print(f"Predicted Ride Cost: ${predicted_price[0]:.2f}")
```
---

## 📈 **Model Performance**
The model's accuracy is assessed by comparing **actual vs predicted values** using visualization techniques:
- **Regression plots** to evaluate prediction trends.
- **Heatmaps** to analyze feature correlations.
- **Boxplots** to examine cost distributions.

---

## 🤝 **Contributing**
We welcome contributions! 🚀 To contribute:
1. **Fork** this repository 🍴
2. **Create** a new branch `feature-branch` 🌿
3. **Commit** changes 💾
4. **Push** to GitHub 🚀
5. **Submit** a pull request 🔥

---

## 📜 **License**
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

💡 _Built with passion for AI and Machine Learning ❤️_

