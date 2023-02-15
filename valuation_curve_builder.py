# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 21:19:51 2023

@author: Renea
"""


import pandas as pd
import sqlite3
from tkinter import Toplevel, ttk, messagebox
from dateutil.relativedelta import relativedelta
from itertools import repeat

def calculate_date_delta_by_decimal(date, decimal):
    ''' helper function to create a list of dates from a starting date and a list of months'''
    newdate = date + relativedelta(months=decimal)
    return newdate

def timestamp_to_date(time):
    ''' helper function to convert timestamp to simple date string'''
    item = pd.Timestamp(time)
    return item.date()

class interpolate_curve:
    def __init__(self):
        
        # read the dataframe of spot rates and drop the first 359 rows so only data from year 2000 onwards is in
        self.valuation_curve = pd.read_excel('.\Economic Data\Curves\Spot_curve_pandas_friendly.xlsx')
        self.valuation_curve = self.valuation_curve.tail(-359).reset_index(drop=True)    
        self.db = r'.\valuation_curve.db'
        self.amount = 426
        self.progress()
        self.run()

    def build_database(self, curvedata):
        ''' method for building the database of data '''
        
        conn = sqlite3.Connection(self.db)
        curvedata.to_sql('valcurve',conn, if_exists='replace', index=True) 
        self.progress_jump()
        cur = conn.cursor()
        cur.execute("CREATE INDEX valindex ON valcurve (valuation_date)")
        self.progress_jump()
        conn.close()

    def interpolate_spot_curve(self):
        ''' this takes the spot curve data and interpolates all the daily data using a cubic spline forward and 
            a simple backfill backwards (since the earliest dates are a short time frame)
        '''
        
        # capture curve date / valuation date as a list
        valuation_dates = self.valuation_curve['Unnamed: 0'].tolist()
        
        # capture a list of monthly periods from column header eg [1,2,3...460]
        valuation_periods = self.valuation_curve.columns.tolist()
        
        # remove column name from list
        valuation_periods = valuation_periods[1:len(valuation_periods)] 
        
        # transpose data
        self.valuation_curve  = self.valuation_curve.transpose()
        
        # now we iterate through the dataframe columns and capture the yields, combine it with the corresponding periods
        # and interpolate the days in between
        dflist=[]
        valuation_dates = list(map(timestamp_to_date, valuation_dates))
        counter=0
        for date in valuation_dates:
            # convert the list of months into dates by mapping it to the helper function defined at top of code
            periods = list(map(calculate_date_delta_by_decimal,repeat(date),valuation_periods))
            
            # convert to datetime 
            periods = list(map(pd.to_datetime, periods))
            
            # capture yield data from dataframe columns
            col = self.valuation_curve.iloc[:,[counter]]
            col = col[counter].to_list()
            col = col[1:len(col)]
            
            # recombine data into a dataframe with yields corresponding to their actual dates
            valuation_df = pd.DataFrame(index=periods, data={"yield":col})   
            
            # upsample to daily data using a cubic spline interpolation
            upsampler = valuation_df.resample('D')
            valuation_df = upsampler.interpolate(method='cubic')
            valuation_df = valuation_df.bfill()
            
            # add valuation date to data and tick counter
            valuation_df['valuation_date'] = date
            dflist.append(valuation_df)
            
            self.progress_step()
            counter+=1
        return pd.concat(dflist)

    def progress(self):
        ''' displays a progress bar as data is parsed from excel files. The two following methods are helper 
            methods for this progress bar method
        '''
        
        self.pb_popup = Toplevel()
        self.pb_popup.title('Progress...')
        self.pb_popup.geometry("300x100")
        
        self.pb = ttk.Progressbar(
        self.pb_popup,
        orient='horizontal',
        mode='determinate',
        length=300)

        self.value_label = ttk.Label(self.pb_popup, text=self.update_progress_label())
        self.value_label.grid(column=0, row=1, columnspan=2)
        self.pb.grid(column=0, row=2, columnspan=2)
               
    def update_progress_label(self):
        return f"Current Progress: {self.pb['value']:.1f}%"

    def progress_step(self):
        if (self.pb['value'] < 100):
            self.pb['value'] += (100 / self.amount) # Note : updated after condition
            self.value_label['text'] = self.update_progress_label()       
        else:
            messagebox.showinfo(message='Database Build Complete')
            self.pb_popup.destroy()

    def progress_jump(self):
        ''' make the progress bar leap'''
        for _ in range(50):
            self.progress_step()

    def run(self):
        ''' runs the entire process of interpolating spot curve and injecting into database '''

        # gets data        
        data = self.interpolate_spot_curve()
 
        # build database
        self.build_database(data)       
        
        # final step to push progress bar past 100% to trigger the 'complete' condition
        self.progress_jump()
