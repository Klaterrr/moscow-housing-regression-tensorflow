# Moscow Housing Price Prediction Project 🏠💵

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-%23F37626.svg)](Moscow_Housing_Price_Prediction.ipynb)

This repository contains a Jupyter Notebook that explores the **Moscow Housing Price Dataset** to predict apartment prices in Moscow and the Moscow Oblast region using machine learning. The project encompasses a comprehensive **Exploratory Data Analysis (EDA)**, **model building**, **hyperparameter tuning**, and a **user-friendly demo** to predict apartment prices based on user input.

## Dataset 📊

The project utilizes the [Moscow Housing Price Dataset](https://www.kaggle.com/datasets/egorkainov/moscow-housing-price-dataset) from Kaggle, collected in November 2023.  The dataset includes features crucial for predicting housing costs such as:

*   **Price:** The target variable - the price of the apartment.
*   **Apartment type:** Type of apartment (e.g., studio, 1-bedroom, etc.).
*   **Metro station:** Nearest metro station.
*   **Minutes to metro:** Walking time to the metro.
*   **Region:** Moscow or Moscow Oblast.
*   **Number of rooms:** Number of rooms in the apartment.
*   **Area:** Total area in square meters.
*   **Living area:** Living area in square meters.
*   **Kitchen area:** Kitchen area in square meters.
*   **Floor:** Apartment floor.
*   **Number of floors:** Total floors in the building.
*   **Renovation:** Level of renovation (e.g., no renovation, cosmetic, euro, designer).

## Jupyter Notebook Content 📝

The [Moscow_Housing_Price_Prediction.ipynb](Moscow_Housing_Price_Prediction.ipynb) notebook is structured to guide you through the entire process of analyzing the dataset and building a predictive model:

**1. Data Understanding & Cleaning:**
*   Loading the dataset and initial exploration using Pandas.
*   Handling missing values and duplicates.
*   Data type analysis and correction.

**2. Univariate Analysis:**
*   Analyzing the distribution of the target variable 'Price'.
*   Exploring the distributions of numerical and categorical features individually using histograms, boxplots, countplots, and descriptive statistics.

**3. Bivariate & Multivariate Analysis:**
*   Investigating relationships between 'Price' and numerical features using scatter plots and correlation matrices.
*   Analyzing the relationship between 'Price' and categorical features using boxplots and bar plots.
*   Checking for multicollinearity among numerical features.

**4. Model Building:**
*   **Baseline Model (Linear Regression):** Building and evaluating a simple linear regression model.
*   **Random Forest Regressor:** Implementing and evaluating a Random Forest model, demonstrating improved performance.
*   **Hyperparameter Tuning with RandomizedSearchCV:** Optimizing the Random Forest model using RandomizedSearchCV to find the best hyperparameters and further enhance performance.

**5. Model Evaluation:**
*   Using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R2) to assess model performance.
*   Comparing the performance of Linear Regression, Baseline Random Forest, and Tuned Random Forest.

**6. Interactive Demo:**
*   A user-friendly demo at the end of the notebook allows you to input apartment characteristics and get price predictions from the trained models (Linear Regression and Tuned Random Forest).
*   Trained models and preprocessors are saved using `joblib` and loaded for real-time predictions.

## Models Used 🤖

The project explores the following machine learning models for price prediction:

*   **Linear Regression:** A simple baseline model for linear relationships.
*   **Random Forest Regressor:** A powerful ensemble method that captures non-linear relationships and provides improved prediction accuracy.
*   **Tuned Random Forest Regressor:**  A Random Forest model with hyperparameters optimized using RandomizedSearchCV, achieving the best performance in this project.

The **Tuned Random Forest model** demonstrated the highest R-squared score of **~0.799** and the lowest error metrics (RMSE and MAE), indicating its superior performance in predicting Moscow housing prices compared to the baseline models.

## Key EDA Insights 🔍

The Exploratory Data Analysis revealed several key insights:

*   **Price Distribution:** Apartment prices exhibit a right-skewed distribution with a significant number of outliers at the higher end.
*   **Area & Rooms:** Strong positive correlation between price and apartment area (total, living, kitchen) and the number of rooms.
*   **Region & Apartment Type:**  Apartments in Moscow are significantly more expensive than in Moscow Oblast. "Secondary" apartments tend to be pricier on average than "New building" apartments.
*   **Renovation:** "Designer" renovation and "Without renovation" categories are associated with higher average prices compared to "Euro-style" or "Cosmetic" renovations.
*   **Metro Proximity:**  Linear correlation between price and "Minutes to metro" is weak, suggesting a more complex relationship or influence of other factors.

## Hyperparameter Tuning ⚙️

Hyperparameter tuning for the Random Forest model was performed using **RandomizedSearchCV**. This process involved:

*   Defining a parameter grid (`param_grid_rf`) with ranges of values for key hyperparameters like `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features`.
*   Using 3-fold cross-validation to evaluate different hyperparameter combinations.
*   Identifying the hyperparameter set that yielded the best performance, resulting in the "Best Random Forest Model".

## Interactive Demo 🕹️

At the end of the notebook, you'll find an interactive demo that allows you to predict apartment prices:

1.  **Run the demo cell:** Execute the code cell in the "Interactive Demo" section.
2.  **Input apartment features:** The demo will prompt you to enter characteristics of an apartment (type, metro station, minutes to metro, region, etc.) through text input.
3.  **Get price predictions:** The demo will output predicted prices from the Linear Regression, Baseline Random Forest, and the Tuned Random Forest models, allowing you to compare predictions.

## Repository Structure after executing the jupiter-notebook 🗂️

```
Moscow_Housing_Price_Prediction/
├── Moscow_Housing_Price_Prediction.ipynb     # Jupyter Notebook with EDA, modeling, and demo
├── moscow_housing_price.csv                  # Dataset file
├── lr_model.joblib                           # Saved Linear Regression model
├── lr_preprocessor.joblib                    # Saved preprocessor for Linear Regression
├── rf_model.joblib                           # Saved Baseline Random Forest model
├── rf_preprocessor.joblib                    # Saved preprocessor for Baseline Random Forest
├── rf+_model.joblib                          # Saved Tuned Random Forest model
└── rf+_preprocessor.joblib                   # Saved preprocessor for Tuned Random Forest
```

## How to Run the Notebook 🚀

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Klaterrr/moscow-housing-regression-tensorflow.git
    cd moscow-housing-regression-tensorflow
    ```
2.  **Install required libraries:** It's recommended to create a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook Moscow_Housing_Price_Prediction.ipynb
    ```
4.  **Follow the notebook:** Execute cells sequentially to reproduce the EDA, model building, and run the interactive demo.

## Conclusion and Further Improvements ✨

This project provides a comprehensive analysis and prediction model for Moscow housing prices. The Tuned Random Forest model offers a significant improvement over a simple linear regression.

Potential further improvements include:

*   **Feature Engineering:** Creating more informative features from existing data.
*   **Exploring other models:** Trying advanced models like Gradient Boosting (XGBoost, LightGBM, CatBoost) or Neural Networks.
*   **More Extensive Hyperparameter Tuning:**  Conducting a more exhaustive search or using Bayesian optimization.
*   **Adding Geographical Data:** Incorporating more detailed geographical information for potentially better predictions.

Feel free to explore the notebook, run the demo, and contribute to further enhance this project!
