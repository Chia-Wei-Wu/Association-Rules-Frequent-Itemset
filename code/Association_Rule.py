#import package
import csv
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

#collect data
def data():
    document = []
    custom_back = []
    with open('customer.csv', newline='') as dataset:  #read data
        row1 = csv.reader(dataset)
        for row in row1:
            document.append('stat_province:'+row[10]) #input stat_province
            document.append('yearly_income:'+row[18]) #input yearly_income
            document.append('gen:'+row[19]) #input gender
            document.append('totoal_children:'+row[20]) #input total_childen
            document.append('num_childen_at_home:'+row[21]) #input num_childen_at_home
            document.append('eduation:'+row[22]) #input eduation
            document.append('occupation:'+row[25]) #input occupation
            document.append('houseowner:'+row[26]) #input houseowner
            document.append('num_car_owner:'+row[27]) #input num_car_owner
            custom_back.append(document)
            document = []
            #print(custom_back)
        del custom_back[0]
        #print(custom_back)
        return custom_back

def fpgrow():
    final_data = data()
    ts = TransactionEncoder()
    data_ary = ts.fit(final_data).transform(final_data)
    frequent_items = pd.DataFrame(data_ary, columns=ts.columns_)
    frequent_items = fpgrowth(frequent_items, min_support=0.05, use_colnames=True)
    Association_Rules = association_rules(frequent_items, metric='confidence', min_threshold=0.9)
    Association_Rules = Association_Rules.head(10)
    Association_Rules.to_csv("fp_growth_custom_back_association_Rules.csv")

print(fpgrow())