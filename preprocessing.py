import json
import pandas as pd
import re

def json_to_csv ():
    df = pd.read_json('business.json', lines=True)
    df.to_csv('business.csv', encoding='utf-8', index=False)
    df = pd.read_json('checkin.json', lines=True)
    df.to_csv('checkin.csv', encoding='utf-8', index=False)
###C########

def leave_only_restaurants_in_vegas ():
    df= pd.read_csv('business.csv', sep=',', header=0)
    df_vegas=df.loc[df['city'] == 'Las Vegas'] #leave only vegas
    df_vegas=df_vegas.dropna() #removes rows with missing values
    df_vegas_rest= df_vegas.loc[df_vegas['categories'].str.contains('Restaurants')]
    df_vegas_rest.to_csv('restaurants_in_vegas.csv', encoding='utf-8', index=False)


def remove_columns():
    df = pd.read_csv('restaurants_in_vegas.csv', sep=',', header=0)
    df= df.drop(['city', 'business_id', 'name', 'state', 'postal_code'], axis=1)
    df.to_csv('restaurants_in_vegas.csv', encoding='utf-8', index=False)

def add_opening_hours_sum():
    df = pd.read_csv('restaurants_in_vegas.csv', sep=',', header=0)
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
    df.to_csv('restaurants_in_vegas.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    #json_to_csv()
    #leave_only_restaurants_in_vegas()
    #remove_columns()
    add_opening_hours_sum()