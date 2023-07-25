"""# **Explainability**"""

#Install non-standard packages
!pip install shap
!pip install lime
!pip install eli5

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

tfidf = TfidfVectorizer(analyzer = 'word', max_features = 100)
X = tfidf.fit_transform(df['tweet'])
X

y = df['intention']
X = X.toarray()
X

X = pd.DataFrame(X)
X

feature = tfidf.vocabulary_
col_names = []

for key, value in feature.items():
    print(key, ' : ', value)
    col_names.append(key)

X.columns = col_names
col_names
X

X.shape, y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train.shape

pip install --user scikit-learn

#eli5 package (https://eli5.readthedocs.io/en/latest)
import eli5
from eli5.sklearn import PermutationImportance

#lime package (https://github.com/marcotcr/lime)
import lime
import lime.lime_tabular

#shap package (https://github.com/slundberg/shap)
import shap

RANDOM_STATE = 123
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import itertools

#Train random forest classification model
model = RandomForestClassifier(max_depth=4, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# Diagnosis prediction
y_predict = model.predict(X_test)

# Probability of malignant tissue produced by the model
y_prob = [probs[1] for probs in model.predict_proba(X_test)]

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=4, random_state=123)
model.fit(X_train, y_train)

# Feature importance dataframe
imp_df = pd.DataFrame({'feature': X_train.columns.values,
                       'importance': model.feature_importances_})

# Reorder by importance
ordered_df = imp_df.sort_values(by='importance')
imp_range=range(1,len(imp_df.index)+1)

## Barplot with confidence intervals
height = ordered_df['importance']
bars = ordered_df['feature']
y_pos = np.arange(len(bars))

# Create horizontal bars
plt.barh(y_pos, height)

# Create names on the y-axis
plt.yticks(y_pos, bars)

plt.xlabel("Mean reduction in tree impurity in random forest")

plt.tight_layout()
# Show graphic
plt.show()


"""# **SHAP**"""

# explain the model's predictions on test set using SHAP values
# same syntax works for xgboost, LightGBM, CatBoost, and some scikit-learn models
explainer = shap.TreeExplainer(model)

# shap_values consists of a list of two matrices of dimension samplesize x #features
# The first matrix uses average nr of benign samples as base value
# The second matrix which is used below uses average nr of malignant samples as base value
shap_values = explainer.shap_values(X_explain)


# Interactive visualization of the explanation of the first subject
# in the test set (X_explain).
# It shows the relative contribution of features to get from the base value of malignant
# samples(average value)
# to the output value (1 in case of malignant sample)
# the numbers at the bottom show the actual values for this sample.
shap.initjs() #initialize javascript in cell
shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_explain.iloc[0,:])

# Import the SHAP library
import shap
import matplotlib.pyplot as plt

# load JS visualization code to notebook
shap.initjs()

# Create the explainer
explainer = shap.TreeExplainer(model)

"""
Compute shap_values for all of X_test rather instead of
a single row, to have more data for plot.
"""
shap_values = explainer.shap_values(X_test)

print("Variable Importance Plot - Suicide Ideation Detection")
figure = plt.figure()
shap.summary_plot(shap_values, X_test)

#Interactive visualization of all sample/feature Shapley values
#It is possible to show the relative contribution of individual features for all
# samples on the y-axis as well.
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], X_explain)

# Dependence Plot on Age feature
shap.dependence_plot('my', shap_values[1], X_test, interaction_index="my")

#A summary plot with the shapley value (feature importance)
shap.summary_plot(shap_values[1], X_explain)

#Same as above, but with violin plots to better see the distribution of shapley values
shap.summary_plot(shap_values[1], X_explain, plot_type="violin")