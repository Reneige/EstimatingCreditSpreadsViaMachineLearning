# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 21:19:51 2023

@author: Rene Alby
"""
import pandas as pd
import sqlite3
from Progress_Bar import Progress_Bar
from dateutil.relativedelta import relativedelta
from itertools import repeat


def calculate_date_delta_by_decimal(date, decimal):
    ''' helper function to create a list of dates from a starting date and a list of months'''
    newdate = date + relativedelta(months=decimal)
    return newdate


def timestamp_to_date(time):
    ''' helper function to convert timestamp to simple datetime.date '''
    item = pd.Timestamp(time)
    return item.date()


class interpolate_curve:
    def __init__(self):
        ''' read the dataframe of spot rates and drop the first 359 rows so only data from year 2000 onwards 
            is in Databaase. 427 represents the number of steps to use on the progress bar - steps for the 276 
            days of data + 3x50 leaps and 1 extra step. 
        '''

        self.valuation_curve = pd.read_excel(
            '.\Economic Data\Curves\Spot_curve_pandas_friendly.xlsx')
        self.valuation_curve = self.valuation_curve.tail(
            -359).reset_index(drop=True)
        self.db = r'.\valuation_curve.db'
        self.progress(427)
        self.run()

    def build_database(self, curvedata):
        ''' method for building the database of yield curve data '''

        # change message and add step so it displays
        self.progressbar.change_message(
            "Injecting data into Database - Please be patient")
        self.progressbar.progress_step()

        # add data to db
        conn = sqlite3.Connection(self.db)
        curvedata.to_sql('valcurve', conn, if_exists='replace', index=False)
        # change message before step so it displays again
        self.progressbar.change_message(
            "Creating Database Search Index - Please be patient")
        self.progressbar.progress_jump()

        # create search index for fast querying of a large dataset
        cur = conn.cursor()
        cur.execute("CREATE INDEX valindex ON valcurve (valuation_date)")
        self.progressbar.progress_jump()
        conn.close()

    def interpolate_spot_curve(self):
        ''' this takes the spot curve data and interpolates all the daily data using a cubic spline forward and 
            a simple backfill backwards (since the earliest dates are a very short time frame from first observation)
        '''

        # capture curve date / valuation date as a list
        valuation_dates = self.valuation_curve['Unnamed: 0'].tolist()

        # capture a list of monthly periods from column header eg [1,2,3...460]
        valuation_periods = self.valuation_curve.columns.tolist()

        # remove column name from list
        valuation_periods = valuation_periods[1:len(valuation_periods)]

        # transpose data
        self.valuation_curve = self.valuation_curve.transpose()

        # now we iterate through the dataframe columns and capture the yields, 
        # combine it with the corresponding periods
        # and interpolate the days in between
        dflist = []
        valuation_dates = list(map(timestamp_to_date, valuation_dates))
        counter = 0
        for date in valuation_dates:
            # convert the list of months into dates by mapping it to the 
            # helper function defined at top of code
            periods = list(map(calculate_date_delta_by_decimal,
                           repeat(date), valuation_periods))

            # add the valuation date to periods - background: previously the 
            # curve would begin 1 month after valuation date
            # this makes it satrt from day 0 which is required for 
            #coupons expiring less than a month away
            
            periods.insert(0, date)

            # convert to datetime
            periods = list(map(pd.to_datetime, periods))

            # capture yield data from dataframe columns
            col = self.valuation_curve.iloc[:, [counter]]
            col = col[counter].to_list()
            col = col[1:len(col)]

            # insert the first element of list into index 0 (i.e. duplicate it). 
            # This aligns the data
            # to the periods where we added an extra date above
            col.insert(0, col[1])

            # Now we need to interpolate the last point of the curve, so start by removing missing
            # elements. Then grab the last value and add the average difference in yield for last
            # element (I calculated it manually as -0.21)
            col_no_na = [x for x in col if str(x) != 'nan']
            datum = col_no_na[-1]
            datum = datum - 0.21
            col[-1] = datum

            # recombine data into a dataframe with yields corresponding to their actual dates
            valuation_df = pd.DataFrame(index=periods, data={"yield": col})

            # upsample to daily data using a cubic spline interpolation
            upsampler = valuation_df.resample('D')
            valuation_df = upsampler.interpolate(method='cubic')
            valuation_df = valuation_df.bfill()

            # add valuation date to data and tick counter
            valuation_df['valuation_date'] = date
            dflist.append(valuation_df)

            self.progressbar.progress_step()
            counter += 1

        # collapse list of dataframe into final large dataframe, 
        # rename index, convert from timestamp to datetime.date
        finaldata = pd.concat(dflist)
        finaldata = finaldata.reset_index().rename(
            columns={'index': 'spot_date'})
        finaldata['spot_date'] = list(
            map(timestamp_to_date, (finaldata['spot_date'])))
        return finaldata

    def progress(self, amount):
        ''' displays a progress bar as data is parsed and calcualted '''
        self.progressbar = Progress_Bar(amount)
        self.progressbar.progress()

    def run(self):
        ''' runs the entire process of interpolating spot curve and injecting into database '''

        # gets data
        self.progressbar.change_message("Interpolating Spot Curve Data ")
        data = self.interpolate_spot_curve()

        # build database
        self.build_database(data)

        # final step to push progress bar past 100% to trigger the 'complete' condition
        self.progressbar.progress_jump()
