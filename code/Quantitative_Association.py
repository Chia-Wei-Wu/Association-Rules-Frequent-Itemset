#import package
import csv
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

#collect data
def col_data():
    data1 = pd.read_csv('sales_fact_1998.csv')
    data1 = pd.DataFrame(data1)
    #print(data1)
    data2 = pd.read_csv('sales_fact_dec_1998.csv')
    data2 = pd.DataFrame(data2)
    #print(data2)
    product = pd.read_csv('product.csv')
    product = pd.DataFrame(product)
    #print(product)
    product_merge1 = pd.merge(product,data1,on=['product_id'])
    product_merge2 = pd.merge(product,data2,on=['product_id'])
    #print(product_merge1)
    #print(product_merge2)
    product_merge = pd.concat([product_merge1,product_merge2])
    #print(product_merge)
    #product_merge.to_csv("product_merge.csv")
    customer_back = pd.read_csv('customer.csv')
    customer_back = pd.DataFrame(customer_back)
    total_data = pd.merge(product_merge,customer_back,on=['customer_id'])
    #total_data.to_csv("total_data.csv")
    feature = ['product_class_id','state_province','yearly_income','gender','total_children','education','occupation','houseowner','num_cars_owned']
    dataset = total_data[feature]
    #print(dataset)
    return dataset

def fpgrow():
    final_data = col_data()
    frequent_items = pd.get_dummies(final_data,columns=final_data.columns)
    frequent_items = fpgrowth(frequent_items, min_support=0.01, use_colnames=True)
    Association_Rules = association_rules(frequent_items, metric='confidence', min_threshold=0.3)
    Association_Rules.to_csv("fp_growth_association_Rules.csv")
    return Association_Rules

#find top10 confidence
def conf():
    Association_Rules = fpgrow()
    result_confidence = Association_Rules.sort_values(['confidence'], ascending=False)
    result_confidence_sort = result_confidence.head(10)
    result_confidence_sort.to_csv("fp_growth_confidence_sort.csv")

#find top10 lift
def lift():
    Association_Rules = fpgrow()
    result_lift = Association_Rules.sort_values(['lift'], ascending=False)
    result_lift_sort = result_lift.head(10)
    result_lift_sort.to_csv("fp_growth_lift_sort.csv")


#main
#col_data()
fpgrow()
#conf()
#lift()