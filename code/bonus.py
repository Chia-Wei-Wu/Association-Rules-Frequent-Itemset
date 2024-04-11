#import package
import csv
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

#clean CSV
def data1():
    document1 = [] #input product_id
    document1_1 = [] #output
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
    document2_1 = [] #output
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


def fpgrow1():
    final_data = data1()
    ts = TransactionEncoder()
    data_ary = ts.fit(final_data).transform(final_data)
    frequent_items = pd.DataFrame(data_ary, columns=ts.columns_)
    frequent_items = fpgrowth(frequent_items, min_support=0.0001, use_colnames=True)
    Association_Rules = association_rules(frequent_items, metric='confidence', min_threshold=0.9)
    Association_Rules.to_csv("fp_growth_association_Rules1.csv")
    return Association_Rules

#find top10 confidence
def conf1():
    Association_Rules = fpgrow1()
    result_confidence = Association_Rules.sort_values(['confidence'], ascending=False)
    result_confidence_sort = result_confidence.head(10)
    result_confidence_sort.to_csv("fp_growth_confidence_sort1.csv")

#find top10 lift
def lift1():
    Association_Rules = fpgrow1()
    result_lift = Association_Rules.sort_values(['lift'], ascending=False)
    result_lift_sort = result_lift.head(10)
    result_lift_sort.to_csv("fp_growth_lift_sort1.csv")

def fpgrow2():
    final_data2 = data2()
    ts2 = TransactionEncoder()
    data_ary2 = ts2.fit(final_data2).transform(final_data2)
    frequent_items2 = pd.DataFrame(data_ary2, columns=ts2.columns_)
    frequent_items2 = fpgrowth(frequent_items2, min_support=0.0001, use_colnames=True)
    Association_Rules2 = association_rules(frequent_items2, metric='confidence', min_threshold=0.9)
    Association_Rules2.to_csv("fp_growth_association_Rules2.csv")
    return Association_Rules2

#find top10 confidence
def conf2():
    Association_Rules2 = fpgrow2()
    result_confidence2 = Association_Rules2.sort_values(['confidence'], ascending=False)
    result_confidence_sort2 = result_confidence2.head(10)
    result_confidence_sort2.to_csv("fp_growth_confidence_sort2.csv")

#find top10 lift
def lift2():
    Association_Rules2 = fpgrow2()
    result_lift2 = Association_Rules2.sort_values(['lift'], ascending=False)
    result_lift_sort2 = result_lift2.head(10)
    result_lift_sort2.to_csv("fp_growth_lift_sort2.csv")


print(conf1())
print(lift1())
print(conf2())
print(lift2())