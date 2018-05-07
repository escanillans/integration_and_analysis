n # Import py_entitymatching package
import py_entitymatching as em
import os
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# Load the pre-labeled data
metacriticData = em.read_csv_metadata("data/metacritic.csv")
wikiData = em.read_csv_metadata("data/wikiData.csv")

# add ID column to each dataset
metacriticID = ["a" + str(num) for num in np.arange(1, len(metacriticData.index)+1)]
wikiID = ["b" + str(num) for num in np.arange(1, len(wikiData.index)+1)]

col_idx = 0
metacriticData.insert(loc = col_idx, column = 'ID', value = metacriticID)
wikiData.insert(loc = col_idx, column = 'ID', value = wikiID)
em.set_key(wikiData, 'ID')
em.set_key(metacriticData, 'ID')

#read in labeled samples
S = em.read_csv_metadata("candidates_sample.csv", 
                         key='_id',
                         ltable=metacriticData, rtable=wikiData, 
                         fk_ltable='ltable_ID', fk_rtable='rtable_ID')

# Split S into I an J
i_file = "I.csv"
j_file = "J.csv"
if not os.path.isfile(i_file): #so you don't delete your labels on accident
    IJ = em.split_train_test(S, train_proportion=0.5, random_state=0)
    I = IJ['train']
    J = IJ['test']
    I.to_csv(i_file,sep=",")
    J.to_csv(j_file,sep=",")
    print("Split samples into I and J")
else:
    I = em.read_csv_metadata(i_file,key="_id",ltable=metacriticData,rtable=wikiData,fk_ltable="ltable_ID",fk_rtable="rtable_ID")
    J = em.read_csv_metadata(j_file,key="_id",ltable=metacriticData,rtable=wikiData,fk_ltable="ltable_ID",fk_rtable="rtable_ID")
    print("Reading I and J from files")
print(len(I))
print(len(J))

# Generate a set of features
F = em.get_features_for_matching(metacriticData, wikiData, validate_inferred_attr_types=False)

# Convert the I into a set of feature vectors using F
H = em.extract_feature_vecs(I, 
                            feature_table=F, 
                            attrs_after='label',
                            show_progress=False)

# create learners
random_state = 0 

dt = em.DTMatcher(name='DecisionTree', random_state=random_state)
rf = em.RFMatcher(name='RF', random_state=random_state)
svm = em.SVMMatcher(name='SVM', random_state=random_state)
ln = em.LinRegMatcher(name='LinReg')
lg = em.LogRegMatcher(name='LogReg', random_state=random_state)
nb = em.NBMatcher(name = 'NaiveBayes')

# Impute feature vectors with the mean of the column values.
H = em.impute_table(H, 
                exclude_attrs=['_id', 'ltable_ID', 'rtable_ID', 'label'],
                strategy='mean')

#initial results
result = em.select_matcher([dt, rf, svm, ln, lg, nb], table=H, 
        exclude_attrs=['_id', 'ltable_ID', 'rtable_ID', 'label'],
        k=5,
        target_attr='label', metric_to_select_matcher='f1', random_state=0)
result['cv_stats']

classifiers = np.array([dt, rf, svm, ln, lg, nb])

# Convert J into a set of feature vectors using F
L = em.extract_feature_vecs(J, feature_table=F,
                            attrs_after='label', show_progress=False)

# Impute feature vectors with the mean of the column values
L = em.impute_table(L, 
                exclude_attrs=['_id', 'ltable_ID', 'rtable_ID', 'label'],
                strategy='mean')

for c in classifiers:
    # Train using feature vectors from I 
    c.fit(table=H, exclude_attrs=['_id', 'ltable_ID', 'rtable_ID', 'label'], target_attr='label')
    
    # Predict on L 
    predictions = c.predict(table=L, exclude_attrs=['_id', 'ltable_ID', 'rtable_ID', 'label'], 
                             append=True, target_attr='predicted', inplace=False)
    
    predictions[['_id', 'ltable_ID', 'rtable_ID', 'predicted','label']].head()

    # Evaluate the predictions
    print(c.name)
    eval_result = em.eval_matches(predictions, 'label', 'predicted')
    em.print_eval_summary(eval_result)
    print()
    
### STAGE 4 ###
### FUNCTIONS###
    
# create function to remove char a/b in list of matches
def removeChar(string):
    return string[1:] 
    
# use this function to edit metacritic genre column since Pop/Rock should be pop rock
def replaceBackslash(string):
    # make sure string is string
    string = str(string)
    
    # if empty, return the string
    if not string:
        return string
    
    # otherwise, replace backslash with a space
    return string.replace("/",' ')

def merge_genre_or_label_or_producer(v1, v2):
    # make sure string is string then set to lower case
    v1 = str.lower(str(v1))
    v2 = str.lower(str(v2))
    
    # remove brackets and single quotes for now
    v1 = v1[v1.find("[")+1:v1.find("]")].replace("'",'').replace("-",' ')
    v2 = v2[v2.find("[")+1:v2.find("]")].replace("'",'').replace("-",' ')

    # in case of multiple values for v1 or v2, we need to split by comma
    v1 = [x.strip() for x in v1.split(',')]
    v2 = [x.strip() for x in v2.split(',')]

    # combine v1 and v2
    v1.extend(v2)
    
    # remove anything in common
    v1 = list(set(v1))
    
    # remove any empty string
    v1 = [i for i in v1 if i]
    
    return v1

def merge_releaseDate(v1, v2):
    # always take wiki release date
    return v2

def merge_metaScore(v1, v2):
    # wiki has no meta score, so return metacritic score (v1)
    return v1

# this function takes two tuples and merges them
def mergeTuple(t1, t2):
    # create empty combined tuple (to add to table E)
    t = []
    for i in np.arange(1,len(t1)):
        if wikiData.columns[i] in ['Album', 'Artist']:
            if len(t1[i]) >= len(t2[i]):
                currT = t1[i]
                t.append(currT)
            else:
                currT = t2[i]
                t.append(currT)
        elif wikiData.columns[i] in ['Genre', 'Label', 'Producer']:
            currT = merge_genre_or_label_or_producer(t1[i], t2[i])
            t.append(currT)
        elif wikiData.columns[i] in ['Release Date']:
            currT = merge_releaseDate(t1[i], t2[i])
            t.append(currT)
        else:
            currT = merge_metaScore(t1[i], t2[i])
            t.append(currT)
    return t

### CODE ###
# read in complete candidate list
C = em.read_csv_metadata("candidates.csv", 
                         key='_id',
                         ltable=metacriticData, rtable=wikiData, 
                         fk_ltable='ltable_ID', fk_rtable='rtable_ID') 
    
# Convert candidate C into a set of feature vectors using F
K = em.extract_feature_vecs(C, 
                            feature_table=F,
                            show_progress=False)
# Impute feature vectors with the mean of the column values
K = em.impute_table(K, 
                exclude_attrs=['_id', 'ltable_ID', 'rtable_ID'],
                strategy='mean')
    
# take best classifier (linear regression) and output predictions (i.e. matches)
predictions = ln.predict(table=K, exclude_attrs=['_id', 'ltable_ID', 'rtable_ID'], 
                        append=True, target_attr='predicted', inplace=False)

# save set of matches between two tables
matches = predictions.loc[predictions['predicted'] == 1]
matches = C.loc[matches.loc[:,'_id'],:]
matches.to_csv("matches.csv", index = False)

# get a list of the indices to match
metacriticIndexMatches = [int(removeChar(s))-1 for s in list(predictions.loc[predictions['predicted'] == 1,'ltable_ID'])]
wikiIndexMatches = [int(removeChar(s))-1 for s in list(predictions.loc[predictions['predicted'] == 1,'rtable_ID'])]
    
# remove "/" in Genre column of metacriticData 
metacriticData['Genre'] = metacriticData['Genre'].apply(lambda x: replaceBackslash(x))

# replace all NaNs with an empty string since that'll be easier for me to work with
metacriticData = metacriticData.replace(np.nan, '', regex = True)
wikiData = wikiData.replace(np.nan, '', regex = True)

# initialize table E that merges metacriticData and wikiData
E = pd.DataFrame(columns = metacriticData.columns[1:])

# add all matches
for i in np.arange(0,len(wikiIndexMatches)):
    # merge tuples in wikiIndexMatches and metacriticIndexMatches (exclude ID column)
    metaTuple = metacriticData.loc[metacriticIndexMatches[i]][0:]
    wikiTuple = wikiData.loc[wikiIndexMatches[i]][0:]
    mergedT = mergeTuple(metaTuple, wikiTuple)
    
    # add to E
    E.loc[i] = mergedT

# add everything NOT in metacriticIndexMatches and wikiIndexMatches  
metaIndices = list(range(np.shape(metacriticData)[0]))
wikiIndices = list(range(np.shape(wikiData)[0]))
leftoverMetaIndices = [i for i in metaIndices if i not in metacriticIndexMatches]
leftoverWikiIndices = [i for i in wikiIndices if i not in wikiIndexMatches]

# subset data based on leftover indices (exclude ID column)
subsetMetaData = metacriticData.iloc[leftoverMetaIndices, 1:]
subsetWikiData = wikiData.iloc[leftoverWikiIndices, 1:]

# lower case genre, label, and producer cell values
toLowerCase = ['Genre', 'Label', 'Producer']
for col in toLowerCase:
    subsetMetaData[col] = subsetMetaData[col].str.lower().str.replace("-",' ')
    subsetWikiData[col] = subsetWikiData[col].str.lower().str.replace("-",' ')


# combine with E
E = pd.concat([E, subsetMetaData], axis = 0)
E = pd.concat([E, subsetWikiData], axis = 0)

# change index of E
E.index = range(np.shape(E)[0])

# replace empty strings and lists with NaN
E.replace(r'^\s*$', np.nan, regex=True, inplace = True)

#convert all elements to string first, and then compare with '[]'. Finally use mask function to mark '[]' as na
# then save to "E.csv"
E.mask(E.applymap(str).eq('[]')).to_csv("E.csv", index = False)

















    
    