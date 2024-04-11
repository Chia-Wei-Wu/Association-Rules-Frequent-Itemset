#import package
import csv
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#clean CSV
def data1():
    document1 = [] #input product_id
    document1_1 = [] #input
    x='0'
    y='0'
    with open('sales_fact_1998.csv', newline='') as dataset1: #read data
        row1 = csv.reader(dataset1)
        for row in row1:
            if row[1] == x and row[2] == y:
                document1.append(row[0])
            else:
                x = row[1]
                y = row[2]
                document1_1.append(document1)
                document1=[]
                document1.append(row[0])
        document1_1.append(document1)
        del document1_1[0]
        del document1_1[0]
        #print(document1_1)
        return document1_1

def data2():
    document2 = [] #input product_id
    document2_1 = [] #input
    x='0'
    y='0'
    with open('sales_fact_dec_1998.csv', newline='') as dataset2: #read data
        row2 = csv.reader(dataset2)
        for row in row2:
            if row[1] == x and row[2] == y:
                document2.append(row[0])
            else:
                x = row[1]
                y = row[2]
                document2_1.append(document2)
                document2=[]
                document2.append(row[0])
        document2_1.append(document2)
        del document2_1[0]
        del document2_1[0]
        #print(document2_1)
        return document2_1

def merge():
    x = data1()
    y = data2()
    total_data = np.sum([x,y])
    #print(total_data)
    return total_data


def apr():
    x =  merge()
    ts = TransactionEncoder()
    ts_data = ts.fit(x).transform(x)
    df = pd.DataFrame(ts_data, columns=ts.columns_)
    frequent_itemsets = apriori(df,min_support=0.0001,use_colnames=True)
    Association_Rules= association_rules( frequent_itemsets ,metric='confidence',min_threshold=0.9)
    Association_Rules.to_csv("Association_Rules.csv")
    return Association_Rules

# find top10 confidence
def conf():
    Association_Rules = apr()
    result_confidence = Association_Rules.sort_values(['confidence'], ascending=False)
    result_confidence_sort = result_confidence.head(10)
    result_confidence_sort.to_csv("ap_confidence_sort.csv")

# find top10 lift
def lift():
    Association_Rules = apr()
    result_lift = Association_Rules.sort_values(['lift'], ascending=False)
    result_lift_sort = result_lift.head(10)
    result_lift_sort.to_csv("ap_lift_sort.csv")

#main
print(conf())
print(lift())
