# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 20:58:52 2023

@author: Renea
"""

import pandas as pd
import sqlite3
from dateutil.relativedelta import relativedelta
from itertools import repeat
import datetime
import math
from Progress_Bar import Progress_Bar

def timeshift(string):
    return (str(int(string[0:4])+1)+"-"+string[5:7]+"-"+str(int(string[8:10])))

def string_to_date_yyyy_mm_dd(string):
    return datetime.date(int(string[0:4]), int(string[5:7]), int(string[8:10]))

def string_to_date_dd_mm_yyyy(string):
    return datetime.date(int(string[6:10]), int(string[3:5]), int(string[0:2]))

def calc_days_between_dates(start,end):
    return datetime.date.toordinal(end) - datetime.date.toordinal(start)

def construct_payment_timeseries(maturity_date,months_between_coupons, number_of_coupons, coupon):
    ''' counts downward from maturity date to issue date creating time series of bond payments
        note, this may add an additional coupon on issue date, but it doesn't matter because valuations 
        will occur after that date.
    '''
    coupon_date = maturity_date
    coupon_dates_list = [maturity_date]
    payment = [coupon+100]
    
    for x in range(int(round(number_of_coupons,0))):
        coupon_date = coupon_date - relativedelta(months=months_between_coupons)
        coupon_dates_list.append(coupon_date)
        payment.append(coupon)
    #payment[-1] = payment[-1]+100
    coupon_dates_list.reverse()
    payment.reverse()
    return [coupon_dates_list, payment]


def construct_valuation_timeseries(payment_series_list, valuation_date, val_curve):
    ''' takes the dates/coupons list-of-lists and cuts off data before valuation dates. Then adds 
        the number of days between valuation date and coupon dates as a 3rd list. Then adds yields from
        spot curve as a fourth list
    '''
        
    # remove dates and payments that precede the valuation date. If bond is matured -> break -> empty list
    while payment_series_list[0][0]< valuation_date:    
        payment_series_list[0].pop(0)
        payment_series_list[1].pop(0)
        if len(payment_series_list[0]) == 0:
            break
    
    # append the day-count between coupon dates and valuation dates
    days = []
    for date in payment_series_list[0]:    
        days.append(calc_days_between_dates(valuation_date,date))
    payment_series_list.append(days)
    
    # extracts corresponding yields from valuation curve, convet to decimal, and add to list
    valcurve_on_coupon_dates = val_curve[val_curve['spot_date'].isin(payment_series_list[0])]
    yields = valcurve_on_coupon_dates['yield'].tolist()
    yields = [x/100 for x in yields]
    payment_series_list.append(yields)
    return payment_series_list


def discounted_cf(cf, days, spot_yield):
    ''' calculates the discounted cash flow using the spot yield and days to maturity'''
    return cf/(math.pow((1+spot_yield),(days/365.25)))

def discounted_cf_w_spread(cf, days, spot_yield, spread):
    ''' calculates the discounted cash flow using the spot yield and days to maturity whilst applying a spread'''
    #print(f"cf: {cf} spot_yield: {spot_yield} spread: {spread} days: {days}")
    return cf/(math.pow((1+spot_yield+spread),(days/365.25)))


tst = """SELECT DISTINCT Prices.*, Master.Issuer, Master.Coupon, strftime('%d-%m-%Y', Master.Maturity) AS Maturity, Master.IssueDate, Master.FirstCouponDate, Master.CouponFrequency
                   FROM Master 
                   INNER JOIN Prices
                   ON Master.ISIN = Prices.ISIN
                   ORDER BY Prices.Date"""


class ZSpread_Calculator:
    def __init__(self):
        self.database = "./database.db"
        self.val_curve_db = "./valuation_curve.db"
        # get bonds with prices from db
        self.allbonds = self.query(tst, self.database)

    def run(self):
        calcs = []
        # capture number of dates to calculate
        dates = self.get_dates()
        
        # instantiate a progress bar
        self.progressbar = Progress_Bar(len(dates)-1,"Z-Spread Calculation Complete")
        self.progressbar.progress()
        
        for date in dates:
            
            # capture bond data
            self.capture_bond_data(date)
            
            # convert string to datetime
            date = string_to_date_yyyy_mm_dd(date)

            # display date in progress bar
            self.progressbar.change_message(f"Calculating Spreads for {date} ")
            
            # creates a list of lists containing 1: coupon dates, 2: coupon amounts including 
            # any principal payment
            self.list_of_timeseries = list(map(construct_payment_timeseries,
                                               self.maturity,
                                               self.months_between_coupons, 
                                               self.number_of_coupons,
                                               self.coupons))
            
            # extract the spot curve on valuation date
            val_curve = self.query(f"SELECT * FROM valcurve WHERE valuation_date = '{date}'", self.val_curve_db)

            # because the datetime.date gets converted to string in SQLite, we need to convert back to date here
            val_curve['spot_date'] = list(map(string_to_date_yyyy_mm_dd,(val_curve['spot_date'])))
            
            # adds two more lists to list-of-list : 3. days between coupon and valuation date. 4. spot curve
            # rate that corresponds to the coupon/cf date. Note this is applied to list in-place.
            for x in self.list_of_timeseries:
                x = construct_valuation_timeseries(x, date, val_curve)
            
            # calculate z-spreads into lists and merge back into bond data (self.bonds)
            self.calc_zspreads()
            calcs.append(self.bonds)
            self.progressbar.progress_step()
    
        data = pd.concat(calcs)
        return data[['Date','Bid','Ask','BidYld','AskYld','BidYChg','Calculated_DirtyPrice','ZSpread','Calculated_ZSpread','RedemptionDate','ID','ISIN']]

    def query(self, query, db):
        conn = sqlite3.connect(db)
        data = pd.read_sql(query, conn)
        conn.close()
        return data

    def get_dates(self):        
        dates = self.query("SELECT DISTINCT(Date) FROM Prices", self.database)
        dates = dates['Date'].tolist()
        return dates

    def capture_bond_data(self, valu_date):
        
        # Take a subset of the bond data - slice it out by date.
        # Ugly! timestamp stored as string, so convert datetime.date to pd.timestamp then to string
        self.bonds = self.allbonds.loc[self.allbonds['Date'] == str(pd.Timestamp(valu_date))]

        # drop rows with missing price data
        self.bonds = self.bonds.dropna(subset=['Bid'])
        
        #extract the coupons and convert to float
        self.coupons = self.bonds['Coupon'].to_list()
        self.coupons = list(map(float, self.coupons))
        
        #extract issue datetime and remove timestamp portion, shift ahead 1 year
        self.issue = self.bonds['IssueDate'].to_list()
        self.issue = list(map(string_to_date_yyyy_mm_dd,self.issue))
        
        # capture frequency, swap nan's with 1 and convert to integers
        self.frequency = self.bonds['CouponFrequency'].to_list()
        self.frequency = [int(x) if not math.isnan(x) else 1 for x in self.frequency]
        
        # capture maturity and convert to dates
        self.maturity = self.bonds['Maturity'].to_list()
        self.maturity = list(map(string_to_date_dd_mm_yyyy,self.maturity))
        
        # adjust coupons for frequency
        self.coupons = [x/y for x,y in zip(self.coupons,self.frequency)]
        
        # extract prices
        self.cleanbidprice = self.bonds['Bid'].tolist()
            
        self.num_days_in_bond = list(map(calc_days_between_dates,self.issue, self.maturity))
        self.num_years_in_bond = [x/365.25 for x in self.num_days_in_bond]
        self.number_of_coupons = [x*y for x,y in zip(self.num_years_in_bond,self.frequency)]
        self.months_between_coupons = [int(12 / x) for x in self.frequency]
        
    
    def calc_zspreads(self):  
        '''
        the below is an algorithm for calculating Z-Spread. It begins by finding the market dirty price. 
        Then it calculates the discounted cashflow price, applying a spread downward or upward depending on if 
        the calculated price is below or above the market dirty price. For efficiency, we iterate backward and 
        forward applying more and more precision each time the calculated price rises above or falls below 
        target price 
        '''
        spreads=[]
        dirtyprices=[]
        
        for clean_market_price, valuation_series, freq in zip(self.cleanbidprice,self.list_of_timeseries,self.frequency):
                       
            # handling for matured bonds. check if length is zero and skip them
            if len(valuation_series[0]) == 0:
                spreads.append(None)
                dirtyprices.append(None)
                continue
            
            # handling for matured bonds. check if length is 1 and but days to next payment is zero - skip 
            if len(valuation_series[0]) == 1 and valuation_series[2][0] == 0:
                spreads.append(None)
                dirtyprices.append(None)
                continue      
            
            
            days_to_first_pmt = valuation_series[2][0]
            coupon = valuation_series[1][0]
            
            # handling for case where coupon includes principal payment
            if coupon > 50:
                coupon = coupon - 100
            
            # calculation for bond accrual and dirty price
            accrual = coupon*(1-(days_to_first_pmt/(365.25/freq)))
            dirty_market_price = clean_market_price+accrual
            calculated_price = sum(list(map(discounted_cf,valuation_series[1],valuation_series[2],valuation_series[3])))
            
            decimal = [0.001,0.0001,0.00001,0.000001]
            spread=0
            for precision in decimal:
                if calculated_price > dirty_market_price:
                    while calculated_price > dirty_market_price:
                        spread += precision
                        calculated_price = sum(list(map(discounted_cf_w_spread, valuation_series[1], valuation_series[2], valuation_series[3], repeat(spread))))
                       # print(f"spread : {spread}   calc price {calculated_price}   target   {dirty_market_price}")
                if calculated_price < dirty_market_price:
                    while calculated_price < dirty_market_price:
                        spread -= precision
                        calculated_price = sum(list(map(discounted_cf_w_spread, valuation_series[1], valuation_series[2], valuation_series[3], repeat(spread))))
                      #  print(f"spread : {spread}   calc price {calculated_price}   target   {dirty_market_price}")
            spreads.append(spread*10000)
            dirtyprices.append(dirty_market_price)
        self.bonds['Calculated_DirtyPrice'] = dirtyprices
        self.bonds['Calculated_ZSpread'] = spreads