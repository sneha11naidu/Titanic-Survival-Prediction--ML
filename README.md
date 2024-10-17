# Titanic-Survival-Prediction--ML
In this Project we explore the Titanic Survival Dataset to predict passenger survival. 

The process began with data analysis and visualization, which was crucial for a comprehensive understanding
of the factors at play. Data pre-processing followed, streamlining the datasets for the subsequent application of machine
learning models. During this stage, the pivotal task was feature scaling, crucial for maintaining a balanced influence of all
variables in the predictive process. We then separated target outcomes from features, with normalization to ensure data
uniformity. After splitting the data into training and testing sets, the focus shifted to model tuning. In this stage, various
hyperparameters were tuned through different methods, using specific performance metrics to evaluate and enhance the
models’ performance. Let’s delve into the specifics in the report below.

3.1. Pre-processing
Starting with the Titanic dataset, I first loaded it to understand the data. This helped me identify which columns weren’t
going to be useful for predicting who survived. For instance, names and ticket details didn’t seem helpful, so I decided to
drop them to decrease the models load.


Then I moved to Data Analysis and Visualization. I plotted various graphs to understand the patterns in the data and
which factors affected the Survival rate. Like, passengers in first class had a better chance of surviving than others as
in Figure. And, depending on their age and sex, some passengers were more likely to survive than others as shown
in Figure. The graphs also showed that men and women had different survival rate, with women with higher rate of
survival. Figure[6].


The heatmap of all the data pointed out which factors might influence survival. For example, it showed that age and ticket
class were big factors. Additionally, the number of siblings/spouses aboard (SibSp) also hinted at survival probabilities,
indicating that those with fewer family members aboard were more likely to survive. After spotting some missing information
in important areas like age and passenger class, I filled those gaps with median values or placeholders to keep the
data consistent.


Next I turned the ’Sex’ column from words into numbers using Label Encoding so that our models could understand it
better. Then I normalized the data using Min-Max Scaling. After that I split the data into Two sets Target and Feature
further diving them into Training and Testing data 650 datapoints for training and the rest 240 for testing . With the data
now ready and set up nicely, we’re all prepped to start using models.

![image](https://github.com/user-attachments/assets/b74dfcaa-1926-49ad-b98f-58da4ff361d5)

![image](https://github.com/user-attachments/assets/5c839ff4-f4f1-4f83-9794-7138e9e7b948)

3.2. Methodology
We expiremented with various classification models Decision Trees and Logistic Regression and Random Forest Regression.
The results leading to Decision Tree being our Main Model. The Decision Tree Classifier excelled in our analysis
for its clarity and adaptability. Its interpretability is a major advantage, letting us follow the thought process behind each
prediction. This is especially valuable for historical data like ours, where understanding the reasons behind survival is as
important as knowing who survived. Mathematically, the operation of a Decision Tree can be summarized with a formula
representing the decision-making process from the root to the leaves:
Mathematical Definition:

<img width="116" alt="image" src="https://github.com/user-attachments/assets/2e7d20ea-d7c5-478e-a63e-f17b33603f64">

In this formula:
• DT(x) denotes the Decision Tree’s prediction for an input x.
• N is the number of leaf nodes, each corresponding to a decision outcome.
• ci is the outcome predicted within the ith leaf node.
• I is an indicator function that returns 1 if x falls within the region Ri associated with leaf node i, and 0 otherwise.
Through this equation, we can see how the Decision Tree classifies each individual by following a series of decisions
(I(xRi)) until reaching a conclusion (ci) at a leaf node. This clarity and sequential breakdown of decisions underscore why
the Decision Tree is especially suited to our analysis of the Titanic dataset.


3.3. Experiments
3.3.1. EXPERIMENTAL SETTINGS


In our experiment, we explored several models to analyze the Titanic dataset, beginning with Logistic Regression, advancing
to models such as Random Forest and Decision tree. For hyperparameter I used Grid Search CV a powerful tool that
searches through a range of parameter values to find the most effective settings for each model.
RandomForestClassifier: Tuned ’max-depth’ and ’n-estimators’, achieving the best performance with a depth of 5 and 200
trees, which significantly improved accuracy and precision.
DecisionTreeClassifier: Adjusted ’max-depth’ and ’min-samples-split’, finding optimal settings at a depth of 5 and a
minimum of 2 samples to split a node. This improved the recall and F1 Score, indicating a good precision-recall balance.
LogisticRegression: Optimized ’C’ for regularization strength and ’penalty’ type, with the best results at ’C’:
1.623776739188721 and ’penalty’: ’l2’, enhancing balanced accuracy and precision.
Through GridSearchCV, we pinpointed the best settings for each model on the Titanic data, enhancing accuracy and
offering insights into optimal model behavior for survival prediction.


3.3.2. RESULTS


In evaluating the performance of our models on the Titanic dataset, we utilized several classification metrics: accuracy,
precision, recall, and F1 Score. Among these, we chose to focus primarily on the F1 Score for its comprehensive ability
to balance the trade-offs between precision and recall. Precision evaluates how well the model identifies true positives
without inflating survival rates, while recall ensures all actual positives are captured, minimizing overlooked survivals.
F1 Score, combining precision and recall, serves as a unified metric reflecting their equilibrium, crucial for accurately
identifying survivors without overlooking any. This balance is vital in predicting Titanic survival, ensuring precise survivor
identification and minimal misclassification.
The results table includes the F1 Score and other metrics for a holistic performance overview.

<img width="450" alt="image" src="https://github.com/user-attachments/assets/99832445-feab-4e19-836f-a72beb7f3ab3">

![image](https://github.com/user-attachments/assets/afd974a1-6ec4-48c4-8f4b-650a19a6e8be)

3.3.3. DISCUSSION
In the analysis of the Titanic survival predictions three models were used namely - RandomForestClassifier, Decision-
TreeClassifier, and LogisticRegression. When comparing these models, the RandomForestClassifier achieved a good balance
between accuracy (0.8319) and F1 Score (0.8160) for the ’survived’ class but the DecisionTreeClassifier outperformed
it with a slightly higher F1 Score (0.8276) and better precision. The LogisticRegression model lagged slightly behind, with
lower overall accuracy (0.7899) and F1 Score (0.7712).
The DecisionTreeClassifier, with its best F1 Score, was the most successful at navigating the complexities of the Titanic
dataset. F1 metric was particularly appropriate for the dataset since it accounts for the balance between not only correctly
predicting survivors (precision) but also not missing out on actual survivors (recall), which is vital when assessing survival
outcomes.
The DecisionTreeClassifier’s ability to parse through the data, considering the unique interactions between features like
age, class, and familial ties, likely gave it the edge in generating a more nuanced predictive model, as indicated by its
superior F1 Score and balanced accuracy.
This confusion matrix for DecisionTreeClassifier shows that the model correctly predicted ’Not Survived’ 53% of the time
and ’Survived’ 31% of the time, with some misclassifications evident. 








