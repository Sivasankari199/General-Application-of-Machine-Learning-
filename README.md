Machine learning has emerged as a promising approach for network intrusion detection (NID) due to its ability to identify complex patterns in network traffic data. In the exploration of machine learning for NID, emphasises the significance of effective algorithms for realtime applications. Despite being a benchmark dataset,
the dataset has drawbacks that are recognised  By using data pre-processing techniques, this study improves the quality and dependability of this data for real-time NID evaluation,
hence addressing these restrictions.we have used preprocessing techniques and trained various ML models in this dataset. Although machine learning performs well at NID, network features must be taken into account.
In their exploration of deep learning models for NID in 5G networks, highlight the necessity of customised methods for various network scenarios. By concentrating on feature engineering and machine learning models especially suited for real-time NID within telecommunication networks, this study expands on this idea.
By forecasting network failures,show how machine learning can be used for network security in 5G in a more comprehensive way than just intrusion detection.
In order to prevent security breaches and ensure network uptime, this research focuses on real-time detection of anomolies inside network traffic data, which is a complement to their work. we have trained various ML
models for this dataset,by employing various pre-processing techniques as recognised by and employed  models  for labelled dataset.

Our goal is to use machine learning techniques to develop a reliable and deployable solution that can improve network security and protect vital infrastructure. We aim to develop a
system that leverages machine learning algorithms to analyze network traffic data identify anomalies indicative of malicious activity and predict it. By comparing each
model’s performance, we aim to identify the most effective approach for detecting network intrusions while minimizing the false positives and achieving high accuracy. This proactive approach will enable us to mitigate
threats before they can cause significant damage.

Data Preprocessing:
By preprocessing the data, we create a high-quality foundation for building robust NID systems which includes
              Handling missing values
              Data Exploration
              Scaling and Encoding
              Feature Selection

Dimensionality Reduction with PCA:

Reducing the number of features (columns) while maintaining the highest possible level of information is known as dimensionality reduction. It is a technique for displaying a given dataset with fewer features—that
is, dimensions—while retaining the significant characteristics of the original data. A variety of feature selection and data compression techniques used in preprocessing are included in dimensionality reduction.

This is equivalent to eliminating features that are superfluous or redundant, or just noisy data, to build a model with fewer variables.

Principal Component Analysis(PCA) is mainly used for Linear Reduction. The data is linearly mapped to a lower-dimensional space in a way that maximizes the variance of the data in the low-dimensional representation while preserving the maximum amount of
information.PCA is utilised to alter the preprocessed data. a new class of features found by PCA, are responsible for capturing the majority of the variance in the original data. The
122 features in the data are reduced to 20 primary components by our technique.

Data Splitting: 

Splitting the data into training and testing sets is the last step before training the model. Using a random state for reproducibility, the data (features and target variable) is
split into two different training and testing sets namely for our research purpose. One in an 80/20 split for balance and the other in a 50/50 split for data efficiency evaluation. The model is trained using the training set,
which enables it to discover patterns and connections in the data. The model’s performance on fresh data and generalizability is next assessed using the unseen testing set.


Model Selection and Analysis:

These models are divided into two primary groups: Ensemble learning models and classification models.
After that, each model is examined and trained to determine which one works best with our particular dataset.

Classification Models:

Classification models attempt to learn a mapping function that assigns a data point to a specific class (normal or attack).
we explore several traditional classification algorithms:
Logistic Regression, GaussianNB, LinearSVC, DecisionTreeClassifier, RandomForest Classifier, KNeighborsClassifier and Isolation Forest.
Ensemble Learning Methods:

Ensemble learning combines predictions from several weaker models typically decision trees, to improve forecasting accuracy and robustness. It uses the collective intelligence of the ensemble to reduce errors or biases that can be present in individual models. Through
ensemble learning, a single, more powerful predictor is created. Every model gets better overall by learning from the errors of the one before it. 
we have trained four
models: XGBoost Classifier, Gradient Boosting Classifier, CatBoost, and Light GBM.

The Champion Model:

Finding the balance between accuracy, speed, and memory utilization is crucial when choosing a model for real-time network intrusion detection (NID). Although models with good accuracy, detection speed,
and low memory consumption, such as GaussianNB, Decision Tree, Logistic Regression, and Linear SVC, may not be able to meet real-time demands because of their overall performance. Similarly, even with its
advantages, Random Forest might be vulnerable to overfitting, which reduces its usefulness in practical situations. The high detection time and memory utilization of K-Nearest Neighbours (KNN) render it an
obvious outlier and unsuitable for real-time applications.
XGBoost and LightGBM are at the forefront of real-time NID. A compelling combination of excellent accuracy, respectable speed, and manageable memoryutilization is provided by XGBoost. Because of its
adaptability, it can be used in a wide range of real-time traffic load. But LightGBM succeeds in analyzing large amounts of network traffic with good speed and efficiency which is our top priority. Despite having the
largest memory footprint among the top performers, its remarkable speed and efficiency in processing big datasets make it the best option for real-time intrusion detection. With a potential balance between speed,
memory consumption, and accuracy, CatBoost exhibits promise. Though each model has advantages and disadvantages, XGBoost and LightGBM stand out as the best options for real-time NID due to their remarkable
performance and economical usage.
