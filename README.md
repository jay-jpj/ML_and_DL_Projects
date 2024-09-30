# ML_and_DL_Projects

# About Projects

# 1. Diabetes Prediction Using Machine Learning and Deep Learning

## Objective
To predict whether a person has diabetes based on several medical predictor variables using various machine learning algorithms and deep learning models. The goal was to optimize the prediction accuracy through both traditional machine learning techniques and deep learning.

## Steps Involved

1. **Exploratory Data Analysis (EDA)**  
   - Performed detailed exploratory data analysis (EDA) to understand the dataset's features, identify missing values, and check for correlations. 
   - Visualized data distributions, feature importance, and relationships between the features and target variable.

2. **Data Preprocessing**  
   - Cleaned and prepared the data by handling missing values, scaling features, and encoding categorical variables where necessary.
   - The dataset was split into training and test sets to evaluate model performance.

3. **Machine Learning Models Implemented:**
   - **Logistic Regression**
   - **Support Vector Machine (SVM)**
   - **Random Forest**
   - **Gradient Boosting**
   - **AdaBoost**
   
   After training and evaluating these models, the **Random Forest** classifier achieved the highest accuracy of 98.79%.

4. **Deep Learning Model**  
   - Built a deep learning model using a multi-layer perceptron (MLP) architecture.
   - The model consisted of several fully connected layers, using **ReLU** activations for hidden layers and **sigmoid** activation for the output layer.
   - The model was compiled using the **binary crossentropy** loss function and **Adam** optimizer.

5. **Model Evaluation**  
   - The deep learning model was trained, validated, and tested. Performance metrics such as accuracy, precision, recall, and F1-score were used to evaluate both machine learning and deep learning models.
   - Although the deep learning model was promising, the **Random Forest** classifier provided the best accuracy overall.

# 2. Twitter Suicidal Ideation Detection Using Machine Learning and Deep Learning

## Objective
To detect potential suicidal ideation from tweets using both classical machine learning and deep learning techniques, with a goal of optimizing accuracy in classification.

## Steps Involved

1. **Exploratory Data Analysis (EDA)**  
   - Analyzed the text lengths of suicidal vs. non-suicidal tweets to identify any patterns.
   - Compared the most frequent words across the whole dataset, followed by separate comparisons for suicidal and non-suicidal tweets.
   - Created word clouds for these frequent word findings to visually represent the differences in word usage.
   - Constructed a co-occurrence heatmap to examine word correlations and frequent combinations.
   - Generated 3-gram count vectorizations to identify the most common three-word combinations.

2. **Text Preprocessing**  
   - Preprocessed the text data, converting it into vectors using **TF-IDF** (Term Frequency-Inverse Document Frequency) to represent the importance of words in the tweets.

3. **Machine Learning Models Implemented:**
   - **Logistic Regression**
   - **Naive Bayes**
   - **Random Forest**
   - **AdaBoost**
   - **Gradient Boost**
   - **Support Vector Classifier (SVC)**

   The **Logistic Regression** model achieved the highest accuracy of 93%. A **Voting Classifier** was then constructed using Logistic Regression, Random Forest, Gradient Boost, and SVC, which boosted the accuracy to 95%.

4. **Deep Learning Models Implemented:**
   - Explored deep learning models like **GRU**, **LSTM**, **CNN**, and a **Hybrid model**. The highest accuracy obtained from deep learning models was 94%.

5. **Fine-Tuning Transformer Models:**
   - Fine-tuned **BERT** and **XLNet** models to further improve accuracy. The **BERT** model achieved a 98% accuracy, while **XLNet** reached 97%.
   - The **BERT model** was chosen as the final model due to its superior performance.

## Conclusion
The project concluded with the **BERT** model as the most accurate, achieving 98% accuracy in detecting suicidal ideation from tweets.



**Twitter Suicidal Ideation Detection:** Support mental health with Twitter Suicidal Ideation Detection. Use natural language processing techniques to identify signs of potential suicidal ideation, offering early intervention and support.

**Credit Card Fraud Detection:** Safeguard financial transactions with Credit Card Fraud Detection. Employ machine learning and deep learning to detect and prevent fraudulent activities, providing secure payment environments.

