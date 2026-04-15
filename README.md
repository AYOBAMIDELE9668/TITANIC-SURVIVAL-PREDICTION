# TITANIC-SURVIVAL-PREDICTION
The primary objective of this project is to build a binary classification model that predicts whether a passenger aboard the RMS Titanic survived the tragic sinking on April 15, 1912. Using passenger data such as age, gender, ticket class, fare, and family size, the model learns patterns that distinguish survivors from non-survivors.

The specific goals of this project are:
•	Perform exploratory data analysis (EDA) to understand the dataset and uncover survival trends.
•	Engineer relevant features and handle missing or inconsistent data appropriately.
•	Train a Logistic Regression classifier on the prepared data.
•	Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
•	Interpret the results and draw meaningful conclusions from the model's performance.

2. Dataset Overview
The dataset used is the classic Titanic Dataset obtained from Kaggle. It contains 891 passenger records and 12 original columns. Below are the key features used in model training:

Feature	Description	Notes
Pclass	Passenger class (1, 2, 3)	1 = First, 2 = Second, 3 = Third
Sex	Gender of passenger	Encoded: male = 0, female = 1
Age	Age in years	Missing values filled with median age
Fare	Ticket fare paid	Missing values filled with median fare
FamilySize	Total family members onboard	Engineered: SibSp + Parch + 1
Survived	Survival outcome (target)	0 = Did Not Survive, 1 = Survived

3. Methodology
3.1 Data Preprocessing
Before training the model, the raw dataset required several preprocessing steps to ensure data quality and model compatibility:
•	Missing Age Values: The Age column had missing entries. These were filled using the median age of the dataset to preserve the distribution without introducing bias.
•	Missing Fare Values: Similarly, missing Fare values were imputed with the median fare.
•	Categorical Encoding: The Sex column was label-encoded (male = 0, female = 1) to convert it into a numeric format suitable for the algorithm.
•	Feature Engineering: A new feature FamilySize was created by summing SibSp (siblings/spouses) and Parch (parents/children) and adding 1 for the passenger themselves.

3.2 Train/Test Split
The dataset was split into 80% training data (712 records) and 20% testing data (179 records) using a stratified split to preserve the proportion of survivors across both sets. A random state of 42 was used for reproducibility.

3.3 Feature Scaling
StandardScaler was applied to normalize all feature values. This is critical for Logistic Regression, as the algorithm is sensitive to the scale of input features. The scaler was fit only on the training data and then applied to the test set to prevent data leakage.

3.4 Model Training
A Logistic Regression model from the scikit-learn library was trained using the scaled training data. The model was configured with max_iter=1000 to ensure convergence and random_state=42 for reproducibility. Logistic Regression was chosen for its interpretability, efficiency, and strong baseline performance on binary classification tasks.

4. Exploratory Data Analysis & Visualizations
Three key visualizations were produced to understand the dataset before model training. These plots reveal important patterns and correlations that informed feature selection.

4.1 Survival Count
The bar chart below shows the distribution of passengers who survived versus those who did not. Out of 891 passengers, approximately 549 did not survive (class 0) while 342 survived (class 1). This indicates a class imbalance, with non-survivors making up roughly 62% of the dataset.
 
Figure 1: Survival Count — Distribution of survivors vs. non-survivors in the Titanic dataset

4.2 Age Distribution by Survival
The histogram below compares the age distribution of survivors and non-survivors. Key observations include: young children (ages 0–10) had a notably higher survival rate; the majority of non-survivors were adults aged 20–35, likely because there were more passengers in this age group; and older passengers (60+) had lower survival rates. The KDE (Kernel Density Estimate) curves clearly show the distribution shape for each group.
 
Figure 2: Age Distribution by Survival — KDE histogram showing age vs. survival outcome

4.3 Feature Correlation Heatmap
The correlation heatmap illustrates the pairwise relationships between all features. Notable findings:
•	Sex has the strongest positive correlation with Survived (0.54), confirming the 'women and children first' evacuation policy.
•	Pclass shows a negative correlation with Survived (-0.34), indicating that first-class passengers had higher survival rates.
•	Fare is positively correlated with Survived (0.26), which aligns with higher-class passengers paying more.
•	Age has a weak negative correlation with Survived (-0.06), suggesting age alone is not a strong predictor.
 
Figure 3: Feature Correlation Heatmap — Pairwise correlations between all features and the target variable

5. Model Evaluation & Results
5.1 Accuracy Score
Overall Model Accuracy: 81.01%
The Logistic Regression model achieved an overall accuracy of 81.01% on the 179-record test set. This means the model correctly predicted survival outcomes for approximately 4 in every 5 passengers — a strong result for a baseline model on this dataset.

5.2 Classification Report
The table below presents the full classification report broken down by class:

Class	Precision	Recall	F1-Score	Support
Not Survived (0)	0.83	0.86	0.85	110
Survived (1)	0.77	0.72	0.75	69
Macro Average	0.80	0.79	0.80	179
Weighted Average	0.81	0.81	0.81	179
Table 1: Classification Report — Precision, Recall, F1-Score for each class

Key takeaways from the classification report:
•	The model performs better at identifying non-survivors (Class 0) with a precision of 0.83 and recall of 0.86, yielding an F1-score of 0.85.
•	For survivors (Class 1), precision is 0.77 and recall is 0.72, resulting in an F1-score of 0.75. This slight drop is expected given the class imbalance.
•	The macro and weighted averages both sit at 0.80–0.81, confirming consistent, balanced performance across both classes.

5.3 Confusion Matrix
The confusion matrix below provides a detailed breakdown of correct and incorrect predictions made by the model on the test set:
 
Figure 4: Confusion Matrix — Actual vs. Predicted survival outcomes on the test set (n = 179)

Interpretation of the confusion matrix values:
•	True Negatives (TN) = 95: Correctly predicted passengers who did not survive.
•	False Positives (FP) = 15: Passengers who did not survive but were incorrectly predicted as survivors.
•	False Negatives (FN) = 19: Actual survivors that the model missed and incorrectly classified as non-survivors.
•	True Positives (TP) = 50: Correctly predicted survivors.

The model demonstrates a good balance between sensitivity and specificity. The relatively low false positive (15) and false negative (19) counts indicate that the model avoids extreme errors in either direction.

6. Challenges Faced & Solutions
Several challenges were encountered during the development of this project. The following describes each challenge and how it was addressed:

Challenge 1: Missing Age Values
Problem: The Age column contained a significant number of missing values (approximately 177 out of 891 records), which could not be ignored as Age is an important predictor of survival.
Solution: Missing Age values were imputed with the median age of the dataset. The median was preferred over the mean to avoid distortion from outliers (e.g., very old or very young passengers). This approach preserved the overall age distribution without introducing artificial bias.

Challenge 2: Categorical Encoding of Gender
Problem: The Sex column contained string values ('male', 'female') which are not directly interpretable by machine learning algorithms.
Solution: Label encoding was applied using a direct mapping (male = 0, female = 1). This binary encoding is well-suited for this feature since Sex is naturally binary and avoids the dimensionality increase associated with one-hot encoding.

Challenge 3: Feature Scaling for Logistic Regression
Problem: Logistic Regression is sensitive to the scale of input features. Features like Fare (ranging from 0 to 512) and Age (0 to 80) exist on very different scales, which can cause the model to disproportionately weight larger-valued features during gradient descent.
Solution: StandardScaler was applied to normalize all features to have a mean of 0 and a standard deviation of 1. Importantly, the scaler was fitted exclusively on the training data and then applied (transformed) to the test data to prevent data leakage.

Challenge 4: Class Imbalance
Problem: The dataset is imbalanced, with approximately 62% non-survivors (class 0) and 38% survivors (class 1). This imbalance could cause the model to be biased toward predicting the majority class.
Solution: A stratified train/test split was used (stratify=y) to ensure that both training and test sets maintained the same class distribution as the original dataset. This helps the model learn fairly from both classes.

7. Conclusion
This project successfully demonstrated the application of Logistic Regression to predict Titanic passenger survival. Through careful data preprocessing, feature engineering, and model evaluation, the trained model achieved an accuracy of 81.01% on unseen test data.

The analysis confirmed well-known historical patterns: female passengers and those in higher ticket classes had significantly better survival rates. The feature correlation analysis identified Sex and Pclass as the most influential predictors, while Fare acted as a proxy for passenger class.

The confusion matrix and classification report showed that the model maintains strong performance for both classes, with only a modest decline in recall for the survivor class due to natural class imbalance in the dataset.

Potential future improvements include:
•	Testing ensemble methods such as Random Forest or Gradient Boosting for improved accuracy.
•	Applying SMOTE (Synthetic Minority Oversampling Technique) to address class imbalance more aggressively.
•	Incorporating additional features such as cabin location, passenger title (Mr., Mrs., etc.), and ticket prefix.
•	Performing hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

References
•	Titanic Dataset — Kaggle Machine Learning Competition (https://www.kaggle.com/c/titanic)
•	scikit-learn Documentation — Logistic Regression (https://scikit-learn.org)
•	Python Data Analysis Library — Pandas (https://pandas.pydata.org)
•	Statistical Data Visualization — Seaborn (https://seaborn.pydata.org)

