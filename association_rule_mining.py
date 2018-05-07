import pandas as pd
import numpy as np
import sys
from itertools import combinations, groupby
from collections import Counter
from IPython.display import display

### FUNCTIONS ###
# Returns frequency counts for items and item pairs
def freq(iterable):
    if type(iterable) == pd.core.series.Series:
        return iterable.value_counts().rename("freq")
    else: 
        return pd.Series(Counter(iterable)).rename("freq")

    
# Returns number of unique orders
def order_count(order_item):
    return len(set(order_item.index))


# Returns generator that yields item pairs, one at a time
def get_item_pairs(order_item):
    order_item = order_item.reset_index().as_matrix()
    for order_id, order_object in groupby(order_item, lambda x: x[0]):
        item_list = [item[1] for item in order_object]
              
        for item_pair in combinations(item_list, 2):
            yield item_pair
            

# Returns frequency and support associated with item
def merge_item_stats(item_pairs, item_stats):
    return (item_pairs
                .merge(item_stats.rename(columns={'freq': 'freqA', 'support': 'supportA'}), left_on='item_A', right_index=True)
                .merge(item_stats.rename(columns={'freq': 'freqB', 'support': 'supportB'}), left_on='item_B', right_index=True))




def association_rules(order_item, min_support, min_frequency):

    print("Starting order_item: {:22d}".format(len(order_item)))


    # Calculate item frequency and support
    item_stats             = freq(order_item).to_frame("freq")
    item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100

    # filter by only looking at genres that appear 4 times or more
    item_stats = item_stats[item_stats['freq'] >= min_frequency]

    # Filter from order_item items below min support 
    qualifying_items       = item_stats[item_stats['support'] >= min_support].index
    order_item             = order_item[order_item.isin(qualifying_items)]

    print("Items with support >= {}: {:15d}".format(min_support, len(qualifying_items)))
    print("Remaining order_item: {:21d}".format(len(order_item)))


    # Filter from order_item orders with less than 2 items
    order_size             = freq(order_item.index)
    qualifying_orders      = order_size[order_size >= 2].index
    order_item             = order_item[order_item.index.isin(qualifying_orders)]

    print("Remaining orders with 2+ items: {:11d}".format(len(qualifying_orders)))
    print("Remaining order_item: {:21d}".format(len(order_item)))


    # Recalculate item frequency and support
    item_stats             = freq(order_item).to_frame("freq")
    item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100


    # Get item pairs generator
    item_pair_gen          = get_item_pairs(order_item)


    # Calculate item pair frequency and support
    item_pairs              = freq(item_pair_gen).to_frame("freqAB")
    item_pairs['supportAB'] = item_pairs['freqAB'] / len(qualifying_orders) * 100

    print("Item pairs: {:31d}".format(len(item_pairs)))


    # Filter from item_pairs those below min support
    item_pairs              = item_pairs[item_pairs['supportAB'] >= min_support]

    print("Item pairs with support >= {}: {:10d}\n".format(min_support, len(item_pairs)))


    # Create table of association rules and compute relevant metrics
    item_pairs = item_pairs.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})
    item_pairs = merge_item_stats(item_pairs, item_stats)
    
    item_pairs['confidenceAtoB'] = item_pairs['supportAB'] / item_pairs['supportA']
    item_pairs['confidenceBtoA'] = item_pairs['supportAB'] / item_pairs['supportB']
    item_pairs['lift']           = item_pairs['supportAB'] / (item_pairs['supportA'] * item_pairs['supportB'])
    
    
    # Return association rules sorted by lift in descending order
    return item_pairs.sort_values('lift', ascending=False)


# Load table E
E = pd.read_csv("E.csv")
E.loc[5180,'Genre']

# convert data to extract each genre for each tuple and make it its own row
Genres = pd.DataFrame(columns = ['tuple_id', 'genre'])

# populate Genres dataframe
E_genre = E.loc[:,'Genre'].dropna(axis = 0)
id_ = 0
count = 0
for tup in E_genre:
    # make genres a list
    tup = tup[tup.find("[")+1:tup.find("]")].replace("'",'').split(",")
    tup = [s.lstrip() for s in tup]
    
    t = [id_]
    for genre in tup:
        # add genre to tuple
        t.append(genre)
        
        # add tuple to Genres
        Genres.loc[count] = t
        count = count + 1
        t = [id_]
    
    id_ = id_+1

# save Genres to csv
Genres.to_csv("GenreData.csv", index = False)
    
### debug ###
Genres = pd.read_csv("GenreData.csv")
#############

# convert from dataframe to Series, with tuple_id as index and genre as value
Genres = Genres.set_index('tuple_id')['genre'].rename('genre')

# display summary statistis for order data
print('dimensions: {0};  unique_tuples: {1};   unique_genres: {2}'
      .format(Genres.shape, len(Genres.index.unique()), len(Genres.value_counts())))

# time
support_pct = 0.01
min_frequency = 4
rules = association_rules(Genres, support_pct, min_frequency)  
rules = rules.sort_values('lift', ascending=False)

# only show results for lift > 1 => there is a positive relationship between A and B 
# i.e. A and B ocur together more often than random
rules_final = rules[rules.loc[:,'lift'] > 1]

rules_final.to_csv("rules.csv", index = False)

display(rules_final)






