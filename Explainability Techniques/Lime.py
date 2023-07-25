
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

"""# **LIME**"""

#Explain samples in test set
X_explain = X_test
explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train.values,
                                                   feature_names=X_train.columns.values,
                                                   discretize_continuous=True,
                                                   class_names=["Suicidal", "Non Suicidal"],
                                                   mode="classification",
                                                   verbose=True,
                                                   random_state=RANDOM_STATE)

#Explaining first subject in test set using all 10 features
exp = explainer.explain_instance(X_explain.values[5,:],model.predict_proba,
                                 num_features=10)
#Plot local explanation
plt = exp.as_pyplot_figure()
plt.tight_layout()
exp.show_in_notebook(show_table=True)