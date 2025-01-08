# Aeropolis 🚁

**Team Members:** 
- Maria Maggiora (297281)
- Alessia Maria Cercaci (299981)

## Introduction 🖍️

In the futuristic city of Aeropolis, autonomous delivery drones are revolutionizing the way goods are transported across the sprawling metropolis, ensuring fast and efficient delivery. These drones play a pivotal role in maintaining the city's dynamic pace, with their performance evaluated by the amount of cargo they can deliver per flight. However, optimizing drone performance is no simple task, as it depends on a multitude of factors, including weather conditions, flight altitude, terrain type, and other operational variables.

Therefore our project addresses the challenge of optimizing drone logistics in Aeropolis by predicting cargo capacity under diverse conditions, thereby improving delivery efficiency and resource utilization. By analyzing a rich dataset encompassing 20 variables, we seek to enhance drone performance, optimize resource allocation, and contribute to the development of smarter urban logistics solutions. 

Our approach is built around a structured data science workflow. We begin with an in-depth Exploratory Data Analysis (EDA) to uncover insights within the dataset. This phase includes visualizing trends, detecting anomalies, and identifying the most critical factors influencing drone performance. The insights gained here serve as the foundation for all subsequent steps in the analysis.

Next, we move to the data preprocessing phase, where the raw dataset is transformed into a format suitable for machine learning models. This includes handling missing values, scaling numerical variables, and encoding categorical features. These tailored preprocessing steps not only prepare the data for analysis but also help highlight hidden relationships that could enhance the predictive power of our models.

Finally, we embark on a detailed experimentation process, testing various machine learning algorithms to identify the most effective approach for predicting drone cargo capacity. Each model is carefully fine-tuned, and its performance is evaluated to ensure it meets the high standards required for Aeropolis’s dynamic delivery ecosystem. This iterative process ensures that our final model is both robust and accurate, ready to contribute to the optimization of Aeropolis’s autonomous delivery systems.

The results will enable Aeropolis to continue leading the way in futuristic logistics and urban innovation! 🥳


## Methods 🔎

Our project began with a thorough analysis of the Aeropolis dataset, employing Python and its powerful ecosystem of libraries such as Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn. This exploratory phase was essential for understanding the dataset’s unique characteristics, including the distribution of key variables and its structure. It has permitted to laid a strong foundation for predictive modeling.

### Dataset Analysis

The EDA started with a thorough examination of the dataset, focusing on understanding its structure, the distribution of variables, checked for anomalies, and any initial trends or patterns.

The Aeropolis dataset comprises 20 features that span environmental, operational, and logistical factors. Below is a preview of the dataset used in this project:

Therefore it consists of:

- **Target Variable**: `Cargo_Capacity_kg` (continuous).
- **Features**: Numerical (e.g., `Flight_Hours`, `Vertical_Max_Speed`) and categorical (e.g., `Weather_Status`, `Package_Type`).

We've also focused on understanding the distribution of both numerical and categorical variables. These univariate analyses provided valuable insights, revealing multimodal distributions in some variables and skewed distributions in others. Additionally, this analysis informed our strategy for imputing missing values later in the data preprocessing pipeline.

![PHOTO-2025-01-05-17-37-15](https://github.com/user-attachments/assets/2675b8c0-eacf-4656-9db1-68a08f9ee17a)

![PHOTO-2025-01-05-17-37-34](https://github.com/user-attachments/assets/adab6640-2677-489e-8f12-7b12bf11a8b3)


### Identification of Anomalies

During the initial Exploratory Data Analysis (EDA), we identified several anomalies within the dataset. 

One notable issue, was that the feature `Cleaning_Liquid_Usage_liters` exhibited a significant right skew. This skewness led to an imbalance in the dataset, adversely impacting the performance of our models.

Additionally, our analysis revealed that features such as `Cargo_Capacity_kg`, `Route_Optimization_Per_Second`, and `Water_Usage_liters` contained negative values. These are logically implausible, as such variables can't assume negative values in real-world scenarios. To address this inconsistency, we decided to replace these erroneous values with NaN. This approach aligns with our data preprocessing pipeline, which includes systematic handling of missing values later in the workflow.

Lastly, the column `Flight_Duration_Minutes` was found to contain Boolean values, which is clearly incorrect since it should represent numerical data. Given that this column was not providing meaningful insights and was irreparably flawed, we decided to remove it from the dataset entirely.

## Preprocessing 👷‍♀️

**Filling Missing Data**

We then checked for missing values in the dataset and visualized the results to conduct a thorough analysis. Our findings revealed that nearly 10% of all features contained missing values, which could pose significant challenges during model training.

![PHOTO-2025-01-05-17-38-04](https://github.com/user-attachments/assets/640bbe05-0477-4e71-a593-aecb6663e11e)

At the beginning of our preprocessing, we attempted to fill missing values using the K-Nearest Neighbors (KNN) imputation method. While this algorithm is effective in many scenarios, it proved to be too slow for our dataset due to the large volume of missing values and the associated computational complexity.
Faced with this challenge, we sought an alternative solution. After examining the correlation of missing values across features, we concluded that the missing data appeared independently in each column. This observation indicates that the dataset follows a Missing Completely at Random (MCAR) pattern.

![Unknown](https://github.com/user-attachments/assets/1af660c8-258d-4f69-a630-44da6cf77b4f)


Based on this analysis, we decided to handle missing values by imputing numerical columns with their median and categorical columns with their mode. However, for the `Cargo_Capacity_Column`, which serves as the target variable, we opted not to apply imputation, as maintaining accuracy in this column is critical, we've just decided to erase all the missing values of this column.

**Feature Encoding**

To prepare the dataset for analysis, categorical variables were transformed using one-hot encoding. This technique created binary columns for each unique category, ensuring compatibility with machine learning models. The final dataset contained 34 features, ready for further analysis.

**Correlation Analysis**

Two correlation matrices were computed to identify relationships between features and the target variable, `Cargo_Capacity_kg`⁠. 

![Unknown](https://github.com/user-attachments/assets/932d364c-3a13-4b07-b11b-2c92b5f9f3f9)

The second heatmap highlights the futures with the stronger relationships with the target. Features with negligible correlations, such as *`Route_Optimization_Per_Second`* and *`Flight_Zone_North`*, were marked for potential removal but we'll use other techinques to analyze the relationship between the values, and calculate how important each feature is in predicting `Cargo_Capacity_kg`. 


![Unknown](https://github.com/user-attachments/assets/f79afde5-8b0e-45a4-bf24-bb9ac10d991c)

**Feature Selection via Mutual Information**

Therefore, to complement correlation analysis, we applied Mutual Information (MI) to evaluate feature importance. Unlike correlation, MI measures both linear and non-linear dependencies, offering a holistic view of feature relevance.

Features with MI scores of 0 were deemed irrelevant and removed. This step ensured that the dataset was streamlined, retaining only impactful variables. By removing irrelevant features, we reduced noise, risks of overfitting, and allows models to focus on the most relevant variables, enhancing both interpretability and performance.

**Dataset Splitting, Scaling, and Reflection**

After feature selection, the dataset was divided into training and test subsets using an 80/20 split, this is important to avoid overfitting. Standard scaling was than applied to standardize the magnitude of features, ensuring uniformity and compatibility with machine learning algorithms.

##  Experimental Design 🔬

### Main Purpose

The primary goal of this project is to evaluate the performance of various machine learning models in predicting drone cargo capacity. Specific objectives include:
•⁠  ⁠Identifying the best-performing model in terms of accuracy and computational efficiency.
•⁠  ⁠Exploring the impact of preprocessing techniques on model performance.

**Regression Models**

To predict the continuous target variable ⁠`Cargo_Capacity_kg`⁠, regression analysis was employed. Three machine learning algorithms were implemented:

1.⁠ ⁠**Linear Regression**:
   - A baseline model chosen for its simplicity and interpretability.
   - Results showed a Mean Absolute Error (MAE) of 0.7552 on the training set, indicating reasonable predictive capability but limited flexibility in capturing complex patterns.

2.⁠ ⁠**Random Forest**:
   - A robust ensemble model capable of capturing non-linear relationships.
   - Achieved a training MAE of 0.2838 but exhibited slight overfitting, as the test MAE increased to 0.7651.

3.⁠ ⁠**Hist Gradient Boosting**:
   - A gradient boosting algorithm optimized for large datasets.
   - Performed consistently across training and test sets with an MAE of approximately 0.754, demonstrating strong generalization capabilities.

**Hyperparameter Optimization**

Hyperparameter tuning was performed to enhance model performance, focusing on Random Forest and Hist Gradient Boosting. Using ⁠ RandomizedSearchCV ⁠, key hyperparameters such as the number of estimators, maximum depth, and learning rate were optimized:

•⁠  ⁠For *Random Forest*, the best configuration reduced overfitting, aligning test performance closer to training results.
•⁠  ⁠For *Hist Gradient Boosting*, fine-tuning further improved generalization, achieving nearly identical results on both training and test datasets.

## Results 👩‍🏫

**Comparative Analysis of Time Complexity Across the Models**

![Unknown](https://github.com/user-attachments/assets/4c1adfca-a1e2-4fd5-8d6c-93612d68f253)

**Performance Metrics Comparison**

The comparison of regression models reveals the following performance metrics:

| Model                   | MAE (Test) | RMSE (Test) | R² (Test) |
|-------------------------|-------------|--------------|------------|
| Linear Regression       | 0.7535     | 0.9392      | 0.6937    |
| Random Forest           | 0.7651     | 0.9566      | 0.6822    |
| Random Forest Tuned     | 0.7778     | 0.9723      | 0.6717    |
| Hist Gradient Boosting  | 0.7538     | 0.9396      | 0.6934    |
| Hist Gradient Boosting Tuned | 0.7538     | 0.9396      | 0.6934    |


**Main Findings**

1.⁠ ⁠*Linear Regression* achieved a balanced trade-off between simplicity, computational efficiency, and predictive performance. Emerging as the best model.

2.⁠ ⁠While *Gradient Boosting* and *Random Forest* demonstrated slightly higher train-set performance, Linear Regression performed equivalently or better on the test set, ensuring its robustness against overfitting.

## Conclusions 🖋️

### Key Takeaways:

1. **Linear Regression Outperforms Other Models**: Across all metrics (MAE, RMSE, R²), Linear Regression consistently demonstrated the best performance, highlighting its suitability for this dataset.
2. **Hist Gradient Boosting is a Strong Alternative**: With comparable performance to Linear Regression, it showcases robustness and generalization capabilities.
3. **Random Forest Requires Further Tuning**: Its overfitting tendencies limit its generalizability to unseen data.

![Unknown](https://github.com/user-attachments/assets/90a6ce5c-dcf3-44ba-a114-e00d0cd9f93b)
