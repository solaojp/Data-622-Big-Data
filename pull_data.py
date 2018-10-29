#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 17:59:51 2018

@author: GD
"""

# fetching data directly from kaggle portal

import csv
import urllib.request
import requests
import urllib.parse, urllib.error


# The direct link to the Kaggle data set
data_url = 'https://www.kaggle.com/c/titanic/download/train.csv'

# The local path where the data set is saved.
local_filename = "traintitanic.csv"

# Kaggle Username and Password
kaggle_info = {'UserName': "*******", 'Password': "******"}

# Attempts to download the CSV file. Gets rejected because we are not logged in.
r = requests.get('https://www.kaggle.com/c/titanic/download/train.csv')

# Login to Kaggle and retrieve the data.
r = requests.post(r.url, data = kaggle_info)


# Writes the data to a local file one chunk at a time.
f = open(local_filename, 'w+') 


#similarly for test file

data_url1 = 'https://www.kaggle.com/c/titanic/download/test.csv'

# The local path where the data set is saved.
local_filename1 = "testitanic.csv"

# Kaggle Username and Password
kaggle_info = {'UserName': "*******", 'Password': "******"}

# Attempts to download the CSV file. Gets rejected because we are not logged in.
r1 = requests.get('https://www.kaggle.com/c/titanic/download/test.csv')

# Login to Kaggle and retrieve the data.
r1 = requests.post(r1.url, data = kaggle_info)


# Writes the data to a local file one chunk at a time.
f1 = open(local_filename1, 'w+') 



























    


