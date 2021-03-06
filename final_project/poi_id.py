from copy import copy
import sklearn

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit

import enron_functions
import evaluate
from sklearn.grid_search import GridSearchCV
# features_list is a list of strings, each of which is a feature name
# first feature must be "poi", as this will be singled out as the label
target_label = 'poi'
email_features_list = [
    # 'email feature
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages',
    ]
financial_features_list = [
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'total_payments',
    'total_stock_value',
]
features_list = [target_label] + financial_features_list + email_features_list

# load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

# remove outliers
outlier_keys = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
enron_functions.remove_outliers(data_dict, outlier_keys)
print len(data_dict)
# instantiate copies of dataset and features for grading purposes
my_dataset = copy(data_dict)
my_feature_list = copy(features_list)


# get K-best features
num_features = 4 #4 for KN
best_features = enron_functions.get_k_best(my_dataset, my_feature_list, num_features)
my_feature_list = [target_label] + best_features.keys() 


'''# Convert NAN to 0 in selected features
for name, item in my_dataset.items():
    for key in my_feature_list:
        if item[key] == 'NaN':
            my_dataset[name][key] = 0
            #print my_dataset[name]
 '''           


# add two new features
#enron_functions.add_financial_sum(my_dataset, my_feature_list)
enron_functions.add_poi_interaction_fraction(my_dataset, my_feature_list) # Adding only this feature
#enron_functions.visualize(data_dict, 'total_stock_value', 'poi_interaction')
# print features
print "{0} selected features: {1}\n".format(len(my_feature_list) - 1, my_feature_list[1:])

# extract the features specified in features_list
data = featureFormat(my_dataset, my_feature_list)

# split into labels and features (this line assumes that the first
# feature in the array is the label, which is why "poi" must always
# be first in the features list
labels, features = targetFeatureSplit(data)

# scale features via min-max
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)



### Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression

l_clf = LogisticRegression(C=10**18, tol=10**-21)






### K-N
from sklearn.neighbors import KNeighborsClassifier
kn_clf = KNeighborsClassifier(algorithm='auto', leaf_size=10, metric='minkowski',
           metric_params=None, n_neighbors=3, p=3, weights='distance')          
 

knn = KNeighborsClassifier()
'''
#TRAIN FOR KN
from sklearn import cross_validation
num_iters=1100
test_size=0.3
for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test =\
            cross_validation.train_test_split(features, labels, test_size=test_size)

#TUNE KNN

scores = ['precision', 'recall']
for score in scores:
    tune_kn_clf_recall = GridSearchCV(KNeighborsClassifier(),
                           { "n_neighbors" : [1, 2, 3, 4,5], 
                           "algorithm":('auto','ball_tree','kd_tree','brute'),
                            "p":[1,2,3,4,5],
                            "leaf_size":[10,20,30,40,50],
                            "weights":('distance','uniform')},
                            scoring = score)
    tune_kn_clf_recall.fit(features_train,labels_train)
    print tune_kn_clf_recall.best_params_
    
 '''


           
### Decision Tree
from sklearn import tree
dt_clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', 
                                  max_depth=None, max_features=None, 
                                  max_leaf_nodes=None,
                                  min_samples_leaf=1, min_samples_split=2, 
                                  min_weight_fraction_leaf=0.0,
                                  random_state=None, splitter='best')
                                  
                                  

### Selected Classifiers Evaluation
#evaluate.evaluate_clf(l_clf, features, labels)
#evaluate.evaluate_clf(kn_clf, features, labels)
#evaluate.evaluate_clf(dt_clf, features, labels)
### Final Machine Algorithm Selection
clf = kn_clf

# dump your classifier, dataset and features_list so
# anyone can run/check your results
pickle.dump(clf, open("my_classifier.pkl", "w"))
pickle.dump(my_dataset, open("my_dataset.pkl", "w"))
pickle.dump(my_feature_list, open("my_feature_list.pkl", "w"))