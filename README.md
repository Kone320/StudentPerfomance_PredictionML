
## Project Overview

This project analyzes student performance data to identify key factors influencing academic success. Using machine learning and statistical techniques, we explore relationships between student characteristics (study habits, extracurricular activities, etc.) and academic outcomes.

**Goal**: Understand what factors impact student academic performance and build predictive models for student success.

**Target Variables**:

-   `math_score`, `physics_score`, `biology_score`, `geography_score`

-   `average_score` (engineered feature)

-   `high_achiever` (binary classification target)

## Key Insights from EDA

### Distribution of Scores

All subject scores show approximately normal distributions with varying means and spreads.

Mathematics and biology scores show the strongest correlation with overall average performance.

Positive correlation is observed between weekly self-study hours and average score. Students without part-time jobs perform better (average score 82 vs 77), and absences negatively correlate with performance. Extracurricular activities have a moderate positive impact.

Mathematics and biology scores are most strongly correlated with average performance (r \> 0.85).

+------------------------------------------------------+------------------------------------------------------------+---------------------------------------------------------------+
| ![Math Score](Graphics/Math_score.png) | ![Biology Score](Graphics/Biology_score.png) | ![English Score](Graphics/English_score.png) \| |
+:====================================================:+:==========================================================:+:=============================================================:+
| Math                                                 | Biology                                                    | English                                                       |
+------------------------------------------------------+------------------------------------------------------------+---------------------------------------------------------------+

#### Correlation plots

#### Corplot

| ![Heatmap](Graphics/Corr.png) \|

#### Corellation with average score

| ![Heatmap](Graphics/corvsavg.png) \|

### Bivariate Analysis

#### Quantitative Variables vs Average Score

+----------------------------------------------------------------------+-----------------------------------------------------------------------+
| ![Study Time vs Average Score](Graphics/studvsavg.png) | ![Absence Days vs Average Score](Graphics/absvsavg.png) |
+:====================================================================:+:=====================================================================:+
| Study Time                                                           | Absence Days                                                          |
+----------------------------------------------------------------------+-----------------------------------------------------------------------+

------------------------------------------------------------------------

#### Categorical Variables vs Average Score

+-----------------------------------------------------------------------+-------------------------------------------------------------------------------------+----------------------+
| ![Part-time Job vs Average Score](Graphics/ptvsavg.png) | ![Extracurricular Activities vs Average Score](Graphics/extvsavg.png) |    ![Career vs Average Score](Graphics/carvsavg.png) |
+:=====================================================================:+:===================================================================================:+:====================:+
| Part-time Job                                                         | Extracurricular Activities                                                          | Career aspiration    |
+-----------------------------------------------------------------------+-------------------------------------------------------------------------------------+----------------------+

## Clustering Analysis

+---------------------------------------------------------------+----------------------------------------------------+
| ![Optimal number of cluster](Graphics/opti.png) | ![cluster plot](Graphics/clplot.png) |
+:=============================================================:+:==================================================:+
+---------------------------------------------------------------+----------------------------------------------------+

K-means clustering identified 2 student profiles:

-High Achievers (high scores, moderate study hours, low absences)

-At-Risk Students (low scores, high absences, variable study patterns).

## Machine learning Models

### Regression Models (Predicting Average Score)

![regression models](Graphics/regr.png) 

**Interpretation**:

-   Tree-based models significantly outperform logistic regression for classification\
-   XGBoost achieves the best overall performance with 96.8% accuracy\
-   High AUC scores (0.99+) indicate excellent class separation capability

### Classification Models (Predicting High Achievers)

![regression models](Graphics/class.png) **Interpretation**:

-   Tree-based models significantly outperform logistic regression for classification\
-   XGBoost achieves the best overall performance with 96.8% accuracy\
-   High AUC scores (0.99+) indicate excellent class separation capability

## Results Interpretation

### Academic Implications

-   Self-study hours show positive correlation with performance\
-   Part-time employment negatively impacts performance (\~5 point difference)\
-   Absences have a negative relationship with performance\
-   Mathematics proficiency is the strongest predictor of overall academic success

### Model Selection Guidance

-   For precise score prediction: Use XGBoost (RMSE = 1.28)\
-   For identifying high achievers: Use XGBoost (Accuracy = 96.8%)\
-   For interpretability: Use Random Forest (good performance with feature importance)
