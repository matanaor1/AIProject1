import json
import ast
import re
import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot



def json_to_csv ():
    df = pd.read_json('business.json', lines=True)
    df.to_csv('business.csv', encoding='utf-8', index=False)
    df = pd.read_json('checkin.json', lines=True)
    df.to_csv('checkin.csv', encoding='utf-8', index=False)


def leave_only_restaurants_in_vegas ():
    df= pd.read_csv('business.csv', sep=',', header=0)
    df_vegas=df.loc[df['city'] == 'Las Vegas'] #leave only vegas
    df_vegas=df_vegas.dropna() #removes rows with missing values
    df_vegas_rest= df_vegas.loc[df_vegas['categories'].str.contains('Restaurants')]#leave only restaurants
    df_vegas_rest.to_csv('restaurants_in_vegas.csv', encoding='utf-8', index=False)


def remove_columns():
    df = pd.read_csv('restaurants_in_vegas.csv', sep=',', header=0)
    df= df.drop(['city', 'business_id', 'name', 'state', 'postal_code'], axis=1)
    df.to_csv('restaurants_in_vegas.csv', encoding='utf-8', index=False)

def categories_to_one_hot():
    df = pd.read_csv('restaurants_in_vegas.csv', sep=',', header=0)
    """ finding most popular caterogies
    all_categories={'Mexican':0}
    for index, row in df.iterrows():
        for c in row['categories'].replace(",","").split():
            if c in all_categories:
                all_categories[c]+=1
            else:
                all_categories[c]=0
    print(sorted(all_categories.items(), key=lambda x: x[1], reverse=True))
    """
    #conversion to one hot:
    temp1 = preprocessing.label_binarize(df['categories'], classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    categories = ['Bars', 'American', 'Nightlife', 'Fast', 'Mexican', 'Sandwiches', 'Pizza', 'Breakfast',
                  'Burgers', 'Chinese', 'Italian', 'Seafood', 'Japanese', 'Asian','Coffee']
    for i in range(len(categories)):
        col = temp1[:, i]
        df['category: ' + categories[i]] = col

    for index, row in df.iterrows():
        row_categories= row['categories'].replace(",","").split()
        for c in row_categories:
            if c in categories:
                df.at[index,'category: '+c] = 1

    #df.drop(['categories'], axis=1)
    df.to_csv('restaurants_in_vegas_after_categories.csv', encoding='utf-8', index=False)

def fix_attributes():
    df = pd.read_csv('restaurants_in_vegas_after_categories.csv', sep=',', header=0)

    """ finding most popular attribultes 
    att = {'WiFi': 0}
    for index, row in df.iterrows():
        for a in ast.literal_eval(row['attributes']).keys():
            if a in att:
                att[a] += 1
            else:
                att[a] = 0
    print(sorted(att.items(), key=lambda x: x[1], reverse=True))
    """

    #attributes to one hot:
    attributes = ['BusinessAcceptsCreditCards', 'RestaurantsPriceRange2','RestaurantsTakeOut','RestaurantsReservations',
                  'RestaurantsGoodForGroups', 'GoodForKids','RestaurantsDelivery','BusinessParking','OutdoorSeating','RestaurantsAttire',
                  'Ambience','HasTV','Alcohol','WiFi','BikeParking','NoiseLevel']
    for i in range(len(attributes)):
        df['attribute: ' + attributes[i]] = ""

    for index, row in df.iterrows():
        for a in ast.literal_eval(row['attributes']).keys():
            if a in attributes:
                df.at[index,'attribute: '+a] = ast.literal_eval(row['attributes'])[a]
    # df.drop(['attributes'], axis=1)

    #binarize business parking:
    for index, row in df.iterrows():
       if row['attribute: BusinessParking'] == 'None':
           df.at[index,'attribute: BusinessParking'] = 0
       elif row['attribute: BusinessParking'] != "":
           df.at[index, 'attribute: BusinessParking'] = 1

    #ambiences to one hot:
    ambiences= ['romantic', 'intimate', 'classy',  'hipster', 'divey', 'touristy', 'trendy', 'upscale', 'casual']
    for i in range(len(ambiences)):
        df['ambience: ' + ambiences[i]] = ""

    for index, row in df.iterrows():
        if row['attribute: Ambience'] == "":
            continue
        elif row['attribute: Ambience'] == "None":
            for a in ambiences:
                df.at[index, 'ambience: ' + a] = 'False'
        else:
            for a in ast.literal_eval(row['attribute: Ambience']).keys():
                if a in ambiences:
                    df.at[index,'ambience: '+a] = ast.literal_eval(row['attribute: Ambience'])[a]


    #fixing noise: removing "u'"
    for index, row in df.iterrows():
        df.at[index, 'attribute: RestaurantsAttire']= row['attribute: RestaurantsAttire'].replace("u\'","\'")
        df.at[index, 'attribute: WiFi'] = row['attribute: WiFi'].replace("u\'", "\'")
        df.at[index, 'attribute: Alcohol'] = row['attribute: Alcohol'].replace("u\'", "\'")
        df.at[index, 'attribute: NoiseLevel'] = row['attribute: NoiseLevel'].replace("u\'", "\'")

    df.to_csv('restaurants_in_vegas_after_attributes.csv', encoding='utf-8', index=False)

def add_opening_hours_sum():
    df = pd.read_csv('restaurants_in_vegas_after_attributes.csv', sep=',', header=0)
    df['Opening Hours'] = df['hours']
    opening_hours_arr_int = []
    for line in df['hours']:
        current_restaurant_hours_only_digits = re.findall(r'[0-9]+', line)
        sum_of_restaurant_opening_hours_weekly = 0
        i=0
        while i < current_restaurant_hours_only_digits.__len__():
            start_hour = int(current_restaurant_hours_only_digits[i])
            start_minute = int(current_restaurant_hours_only_digits[i+1])
            end_hour = int(current_restaurant_hours_only_digits[i+2])
            end_minute = int(current_restaurant_hours_only_digits[i+3])
            if start_hour == end_hour and start_minute == end_minute:
                sum_of_restaurant_opening_hours_weekly += 24
            else:
                current_day_hours = (end_hour - start_hour) + (end_minute - start_minute)/60
                if -12 < current_day_hours < 0:
                    current_day_hours += 12
                if current_day_hours <= -12:
                    current_day_hours += 24
                sum_of_restaurant_opening_hours_weekly += current_day_hours
            i += 4
        if sum_of_restaurant_opening_hours_weekly <= 0: print("error")
        opening_hours_arr_int.append(sum_of_restaurant_opening_hours_weekly)
    df['Opening Hours'] = opening_hours_arr_int

    #removing columns:
    df = df.drop(['address', 'attributes', 'categories', 'hours', 'latitude', 'longitude', 'attribute: Ambience'], axis=1)

    df.to_csv('restaurants_in_vegas_after_hours.csv', encoding='utf-8', index=False)


def convert_non_numeric_date():
    df = pd.read_csv('restaurants_in_vegas_after_hours.csv', sep=',', skipinitialspace=True, header=0)
    zero_or_one=['attribute: RestaurantsGoodForGroups','attribute: BusinessAcceptsCreditCards','attribute: RestaurantsTakeOut',
                 'attribute: RestaurantsReservations', 'attribute: GoodForKids', 'attribute: RestaurantsDelivery',
                 'attribute: OutdoorSeating', 'attribute: HasTV', 'attribute: BikeParking', 'ambience: romantic',
                 'ambience: intimate', 'ambience: classy', 'ambience: hipster', 'ambience: divey', 'ambience: touristy',
                 'ambience: trendy', 'ambience: upscale', 'ambience: casual']
    for col in zero_or_one:
        df[col]=df[col].map({'TRUE': 1, 'FALSE': 0, 'True': 1, 'False': 0, True: 1, False: 0}).astype(float)

    df['attribute: RestaurantsAttire'] = df['attribute: RestaurantsAttire'].map({'\'formal\'': 2, '\'casual\'': 1, '\'dressy\'': 0}).astype(float)
    df['attribute: Alcohol'] = df['attribute: Alcohol'].map({'\'full_bar\'': 2, '\'beer_and_wine\'': 1, '\'none\'': 0}).astype(float)
    df['attribute: WiFi'] = df['attribute: WiFi'].map({'\'free\'': 2, '\'paid\'': 1, '\'no\'': 0}).astype(float)
    df['attribute: NoiseLevel'] = df['attribute: NoiseLevel'].map({'\'very_loud\'':3, '\'loud\'': 2, '\'average\'': 1, '\'quiet\'': 0}).astype(float)
    df['attribute: RestaurantsPriceRange2']=df['attribute: RestaurantsPriceRange2'].map({'1': 1, '2': 2, '3': 3, '4': 4}).astype(float)
    df.to_csv('all_numeric.csv', encoding='utf-8', index=False)


def split_train_test_val():
# Split data to test, train and val
    df = pd.read_csv('all_numeric.csv', sep=',', skipinitialspace=True, header=0)
    temp_train, test = train_test_split(df, test_size=0.2, random_state=1)
    train, val = train_test_split(temp_train, test_size=0.2, random_state=1)
    train.to_csv("train.csv", sep=',', header=True, index=False)
    val.to_csv("val.csv", sep=',', header=True, index=False)
    test.to_csv("test.csv", sep=',', header=True, index=False)

def print_columns_histograms():
    train = pd.read_csv('train_scaled.csv', sep=',', skipinitialspace=True, header=0)
    val = pd.read_csv('val_scaled.csv', sep=',', skipinitialspace=True, header=0)
    test = pd.read_csv('test_scaled.csv', sep=',', skipinitialspace=True, header=0)
    for column in train:
        plot.hist(train[column])
        plot.xlabel(column)
        plot.show()
        plot.hist(val[column])
        plot.xlabel(column)
        plot.show()
        plot.hist(test[column])
        plot.xlabel(column)
        plot.show()

def fill_missing_values():
    train = pd.read_csv('train.csv', sep=',', skipinitialspace=True, header=0)
    val= pd.read_csv('val.csv', sep=',', skipinitialspace=True, header=0)
    test= pd.read_csv('test.csv', sep=',', skipinitialspace=True, header=0)
    medians=train.median()
    #print(medians)
    train=train.fillna(medians)
    val=val.fillna(medians)
    test=test.fillna(medians)
    train.to_csv("train_full.csv", sep=',', header=True, index=False)
    val.to_csv("val_full.csv", sep=',', header=True, index=False)
    test.to_csv("test_full.csv", sep=',', header=True, index=False)


def scale():
    train = pd.read_csv('train_full.csv', sep=',', skipinitialspace=True, header=0)
    val = pd.read_csv('val_full.csv', sep=',', skipinitialspace=True, header=0)
    test = pd.read_csv('test_full.csv', sep=',', skipinitialspace=True, header=0)
    to_scale=['review_count', 'attribute: RestaurantsPriceRange2', 'attribute: Alcohol', 'attribute: WiFi', 'attribute: NoiseLevel',
              'attribute: RestaurantsAttire', 'Opening Hours']
    min_max_scalar = preprocessing.MinMaxScaler()
    train[to_scale] = min_max_scalar.fit_transform(train[to_scale])
    val[to_scale] = min_max_scalar.transform(val[to_scale])
    test[to_scale] = min_max_scalar.transform(test[to_scale])
    train.to_csv("train_scaled.csv", sep=',', header=True, index=False)
    val.to_csv("val_scaled.csv", sep=',', header=True, index=False)
    test.to_csv("test_scaled.csv", sep=',', header=True, index=False)

def split_x_y():
    train = pd.read_csv('train_scaled.csv', sep=',', header=0)
    val = pd.read_csv('val_scaled.csv', sep=',', header=0)
    test = pd.read_csv('test_scaled.csv', sep=',', header=0)

    train_data_x = train.drop(['stars'], axis=1)
    train_data_y = train.stars

    val_data_x = val.drop(['stars'], axis=1)
    val_data_y = val.stars

    test_data_x = test.drop(['stars'], axis=1)
    test_data_y = test.stars

    train_data_x.to_csv("train_x.csv", sep=',', header=True, index=False)
    train_data_y.to_csv("train_y.csv", sep=',', header=True, index=False)
    val_data_x.to_csv("val_x.csv", sep=',', header=True, index=False)
    val_data_y.to_csv("val_y.csv", sep=',', header=True, index=False)
    test_data_x.to_csv("test_x.csv", sep=',', header=True, index=False)
    test_data_y.to_csv("test_y.csv", sep=',', header=True, index=False)

if __name__ == '__main__':
    #json_to_csv()
    #leave_only_restaurants_in_vegas()
    #remove_columns()
    #categories_to_one_hot()
    #fix_attributes()
    #add_opening_hours_sum()
    #convert_non_numeric_date()
    #split_train_test_val()
    #fill_missing_values()
    #scale()
    #print_columns_histograms()
    split_x_y()