# Aeropolis 🚁

**Team Members:** 
- Maria Maggiora (297281)
- Alessia Maria Cercaci (299981)

## Introduction 🖍️

In Aeropolis, a bustling futuristic metropolis, autonomous delivery drones are pivotal in maintaining efficient logistics operations. This project aims to leverage machine learning techniques to predict the cargo capacity of drones under varying operational and environmental conditions. 

Therefore our project addresses the challenge of optimizing drone logistics in Aeropolis by predicting cargo capacity under diverse conditions, thereby improving delivery efficiency and resource utilization. By analyzing a rich dataset encompassing 20 variables, we seek to enhance drone performance, optimize resource allocation, and contribute to the development of smarter urban logistics solutions. 

The dataset is sourced from synthetic simulations of drone operations, capturing a variety of factors affecting cargo capacity, such as weather conditions and operational settings.

### Key Features of the Dataset 🔑
- **Cargo Capacity (Target Variable):** The dependent variable predicting the payload capacity of drones.
- **Wind Speed:** A critical environmental factor that significantly affects drone stability and capacity.
- **Air Temperature:** Can influence the drone's battery efficiency and aerodynamic performance.
- **Package Type:** Encodes information about the type of cargo being transported.
- **Terrain Type:** Captures operational challenges associated with various terrains.
- **Vertical Landing Support:** A binary feature indicating whether drones can land vertically, impacting operational flexibility.

# Midificare questa parte 

This dataset provides a comprehensive view of the factors influencing drone cargo capacity, making it an excellent foundation for predictive modeling and logistics optimization.

## [Section 2] Methods 🔎

### Data Overview
The Aeropolis dataset comprises 20 features that span environmental, operational, and logistical factors. Below is a preview of the dataset used in this project:

![Dataset Preview](images/dataset_preview.png "Dataset Preview")

Our project began with a thorough analysis of the Aeropolis dataset, employing Python and its powerful ecosystem of libraries such as Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn. This exploratory phase was essential for understanding the dataset’s unique characteristics, including its structure and the distribution of key variables. It has permitted to laid a strong foundation for predictive modeling.

### Key Steps in Methodology
1. **Data Acquisition and Inspection:**
   - The Aeropolis dataset was imported, inspected for completeness, and checked for anomalies. A comprehensive overview of its 20 features revealed relationships that could influence cargo capacity predictions. 
2. **Data Cleaning:**
   - Missing values were addressed through imputation methods tailored to the feature type (e.g., mean for numerical variables and mode for categorical variables). Outliers were removed based on domain knowledge and statistical thresholds to ensure data quality.
3. **Exploratory Data Analysis (EDA):** --> cambiare tittolo 
   - Visualizations, including heatmaps, histograms, and scatter plots, were created to explore correlations between features and cargo capacity. These helped identify trends, relationships, and potential preprocessing needs.
   - Example visualizations include:
     - Heatmap of feature correlations:
       ![Heatmap](images/heatmap.png "Heatmap of Feature Correlations")
     - Distribution of Cargo Capacity:
       ![Cargo Capacity Distribution](images/cargo_distribution.png "Cargo Capacity Distribution")
4. **Feature Engineering:**
   - Features were scaled or transformed to ensure uniformity and enhance model performance.
   - Key variables such as wind speed and terrain type were emphasized due to their significant impact on drone operations.
   - Categorical features, like package type and weather status, were one-hot encoded to enable compatibility with machine learning models.
5. **Model Selection and Training:**
   - Multiple models, including Linear Regression, Random Forest, and Gradient Boosting, were tested to determine the most suitable approach for predicting cargo capacity. Models were selected based on their ability to handle numerical and categorical data, scalability, and interpretability.
6. **Optimization:**
   - Hyperparameter tuning was performed using Grid Search Cross-Validation to refine model accuracy and efficiency. Specific parameters such as maximum depth, learning rate, and number of estimators were optimized for each model.

### Workflow Visualization
The project workflow can be summarized as follows:

```markdown
1. **Data Loading:** Import and inspect the Aeropolis dataset.
2. **EDA:** Perform exploratory data analysis with visualizations.
3. **Preprocessing:** Handle missing data, remove outliers, and encode features.
4. **Modeling:** Train and evaluate machine learning models.
5. **Optimization:** Hyperparameter tuning of the best-performing model.
6. **Results:** Analyze performance metrics and derive insights.
```

![Workflow Visualization](images/workflow.png "Workflow Visualization")

### Optimization and Computational Efficiency
The computational efficiency of the models was evaluated, and the time taken by each model is shown below:

![Time Complexity Comparison](images/time_complexity.png "Comparative Analysis of Time Complexity Across Models")

From the chart, it is evident that Linear Regression is the most efficient model in terms of computational time, followed by Hist Gradient Boosting and Random Forest. This highlights the suitability of Linear Regression for scenarios where speed is critical.

### Tools and Environment
- **Programming Language:** Python (v3.x)
- **Libraries Used:**
  - pandas (v1.x): Data manipulation and analysis
  - numpy (v1.x): Numerical computations
  - scikit-learn (v1.x): Machine learning algorithms and preprocessing
  - matplotlib (v3.x) and seaborn (v0.x): Data visualization
- **Environment Setup:**
  - To ensure reproducibility, the project environment can be recreated using the following command:
    ```
    conda env create -f environment.yml
    ```
- **Dataset:** The Aeropolis dataset (`aeropolis.csv`), containing variables such as Wind Speed, Air Temperature, and Cargo Capacity, forms the core of this analysis.

By following this systematic approach, we ensured a robust foundation for predictive modeling, enabling the extraction of actionable insights and the development of reliable machine learning models.

## [Section 3] Experimental Design 🔬

### Main Purpose
The primary goal of this project is to evaluate the performance of various machine learning models in predicting drone cargo capacity. Specific objectives include:
- Identifying the best-performing model in terms of accuracy and computational efficiency.
- Exploring the impact of preprocessing techniques on model performance.

### Baselines
The following models were used as baselines:
- **Linear Regression:** A simple and interpretable model ideal for understanding linear relationships.
- **Random Forest:** A tree-based ensemble method capable of capturing non-linear relationships and handling diverse data types.
- **Gradient Boosting:** A powerful boosting technique known for its efficiency in regression tasks.

### Evaluation Metrics
To assess model performance, we used:
- **Root Mean Squared Error (RMSE):** Highlights large prediction errors and emphasizes the magnitude of error.
- **R-squared (R²):** Measures the proportion of variance in the target variable explained by the model, indicating its explanatory power.
- **MAE** 

### Experimental Details
- **Data Partitioning:** The dataset was split into training (80%) and test (20%) subsets.
- **Hyperparameter Tuning:** Performed using Grid Search Cross-Validation for Gradient Boosting. Key parameters included:
  - Learning Rate
  - Maximum Depth
  - Number of Estimators

By comparing baseline performances and applying optimization techniques, we aimed to achieve the best possible predictive performance for the models.

## [Section 4] Results 🏆

### Main Findings
1. **Linear Regression** achieved a balanced trade-off between simplicity, computational efficiency, and predictive performance.
2. While **Gradient Boosting** and **Random Forest** demonstrated slightly higher train-set performance, Linear Regression performed equivalently or better on the test set, ensuring its robustness against overfitting.

### Visualizations and Tables
- **Feature Correlations Heatmap:** Visualizing relationships between key features:
  ![Feature Correlations Heatmap](images/heatmap.png "Feature Correlations Heatmap")

- **Model Performance Comparison:** Bar chart showing RMSE and R² for each model:
  ![Model Performance Comparison](images/model_metrics.png "Model Evaluation Metrics Comparison")

- **Predicted vs. Actual Values:** Scatter plot of predicted vs actual cargo capacities:
  ![Predicted vs Actual](images/predicted_vs_actual.png "Predicted vs Actual Cargo Capacities")

### Model Performance Table
| Model              | RMSE   | R²   |
|--------------------|--------|-------|
| Linear Regression  | 0.939  | 0.694 |
| Random Forest      | 0.957  | 0.682 |
| Gradient Boosting  | 0.939  | 0.693 |

### Insights
Linear Regression emerged as the best model due to its simplicity, computational efficiency, and comparable accuracy to more complex models. It is especially suitable for scenarios requiring interpretable models and faster computations.

## [Section 5] Conclusions 🖋️

### Summary
This study demonstrates the feasibility of leveraging machine learning to predict drone cargo capacity in Aeropolis. By utilizing models such as Linear Regression, we effectively identified critical factors influencing drone operations and optimized predictive accuracy. The results highlight the importance of preprocessing and feature selection in achieving robust model performance.

### Future Directions
Future research could include:
- Integrating real-time drone telemetry data for dynamic predictions.
- Exploring deep learning models for capturing higher-dimensional patterns.
- Evaluating the impact of additional environmental variables, such as precipitation or wind gust patterns.

By expanding the scope of data and methodologies, this project could further enhance drone logistics in Aeropolis and similar urban environments.
