# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 13:40:56 2023

@author: Rene Alby

To rebuild the database from the raw data files downloaded in to the data folder. This tool has a GUI progress
bar component as well.

"""

import pandas as pd
import glob2
import sqlite3
import os
from tkinter import Toplevel, ttk, messagebox

class database_builder:
    def __init__(self):
        
        self.price_files = glob2.glob('.\**\Price Data\*.xlsx')
        self.financial_files = glob2.glob('.\**\Financials\*.xlsx')
        self.bond_universe = glob2.glob('.\**\Master List of Bonds\*.xlsx')
        self.inflation_curve = r'.\Economic Data\Curves\Spot Implied Inflation Curve.xlsx'
        self.nominal_curve = r'.\Economic Data\Curves\Spot Nominal Curve.xlsx'
        self.gdp = r'.\Economic Data\ONS UK GDP Estimate Monthly.xlsx'
        self.vix = r'.\Economic Data\VIX_History_cboe.xlsx'
        self.ftse100 = r'.\Economic Data\Price History_20230208_FTSE100_refinitiv.xlsx'
        self.total_files = len(self.price_files) + len(self.financial_files) + len(self.bond_universe) + 5
        self.db = r'.\database.db'
        self.progress()
        self.run()

    def forwardfill_data(self, df):
        ''' Market data like ftse returns are only on weekends. This is an issue because compiling the final
            data relies on merging data tables on dates. Missing weekend data results in gaps using this method.
            therefore this function forward-fills market data by filling weekends with the last observed price.
            It relies DataFrame.Resample(), a convenice method that returns a resampling object. The new DataFrame
            is extracted from this object using resampling_object.ffill() and it is reset to descending order
        
        '''
        
        df = df.set_index('Date')
        upsampled = df.resample('D')
        return upsampled.ffill().sort_values(by='Date', ascending=False).reset_index()       
        
    def read_price(self,file):
        ''' method for extracting price data from excel files '''
        
        pricedata = pd.read_excel(file,skiprows=16)
        
        # ugly way to strip out identifier and add it to dataframe
        identifier = pd.read_excel(file, index_col=None, usecols = "A", header = 3, nrows=0)
        identifier = identifier.columns.values[0]
        pricedata[['ID']] = identifier[:-5]
        
        # extract isin from file name
        isin = os.path.basename(file)
        pricedata[['ISIN']] = isin[:-5]
        
        # tick counter up and return data
        self.progress_step()
        return pricedata
    
    def read_financials(self,file):
        ''' method for extracting financial data from excel files '''
        
        agg = []
        
        # extract all statements from each excel tab into a dictionary of dataframes
        findata_dict = pd.read_excel(file,sheet_name=None,skiprows=10, index_col=0)
        
        # extract company name from data and add as dataframe column
        company = pd.read_excel(file, index_col=None, usecols = "B", header = 1, nrows=0)
          
        # cycle through, capture each statement, add as a column and concat into large dataframe
        for key, value in findata_dict.items():
            financial_statement = key
            df = value
            df = df.transpose()
            
            # add company name and statement type to data
            df[['Statement']] = financial_statement.replace(" ", "-")
            df[['Company']] = company.columns.values[0]
            
            # remove duplicated columns eg 'total' lines on financial statements
            df = df.loc[:,~df.columns.duplicated()]
            
            # append to list
            agg.append(df)

        # tick counter up and return data
        self.progress_step()
        return agg
    
    def read_bond_universe_master(self, file):
        ''' method for extracting bond reference data from excel files '''
        
        uni = pd.read_excel(file)
        self.progress_step()       
        return uni

    def build_database(self, pricedata, security_master, findata, nomcurve, inflcurve, gdp, vix, ftse100):
        ''' method for building the database of data '''
        
        conn = sqlite3.Connection(self.db)
        pricedata.to_sql('Prices',conn, if_exists='replace', index=False)   
        security_master.to_sql('Master',conn, if_exists='replace', index=False)   
        findata.to_sql('Financials',conn, if_exists='replace', index=False) 
        nomcurve.to_sql('Nominal_Curve',conn, if_exists='replace', index=False)  
        inflcurve.to_sql('Inflation_Curve',conn, if_exists='replace', index=False) 
        gdp.to_sql('GDP',conn, if_exists='replace', index=False)
        vix.to_sql('VIX',conn, if_exists='replace', index=False)
        ftse100.to_sql('FTSE100',conn, if_exists='replace', index=False)
        conn.close()

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
            self.pb['value'] += (100 / self.total_files) # Note : updated after condition
            self.value_label['text'] = self.update_progress_label()       
        else:
            messagebox.showinfo(message='Database Build Complete')
            self.pb_popup.destroy()


    def run(self):
        ''' runs the entire process of parsing data and injecting into database '''
        
        # aggregate price data into dataframes and removes spaces and dashes from column names
        list_of_dfs = list(map(self.read_price,self.price_files))
        pricedata = pd.concat(list_of_dfs)
        pricedata.columns = pricedata.columns.str.replace(' ', '')
        pricedata.columns = pricedata.columns.str.replace('-', '')        
        
        # aggregate financial data
        # the handling here is different because it is a list of lists, so need to flatten the list first
        list_of_dfs2 = list(map(self.read_financials,self.financial_files))
        list_of_dfs2 = [item for sublist in list_of_dfs2 for item in sublist]
        findata = pd.concat(list_of_dfs2)
              
        # convert columns to lower case and drop duplicates and remove spaces and dashes from column names
        findata = findata.loc[:,~findata.columns.str.lower().duplicated()]
        findata.columns = findata.columns.str.replace(' ', '')
        findata.columns = findata.columns.str.replace('-', '')
      
        # aggregate bond universe / security master data and remove spaces and dashes from column names
        list_of_dfs3 = list(map(self.read_bond_universe_master,self.bond_universe))
        security_master = pd.concat(list_of_dfs3)
        security_master.columns = security_master.columns.str.replace(' ', '')
        security_master.columns = security_master.columns.str.replace('-', '')
        
        # econ data (less complex since data is preformatted)
        nomcurve = pd.read_excel(self.nominal_curve)
        self.progress_step()
        
        inflcurve = pd.read_excel(self.inflation_curve)
        self.progress_step()
        
        gdp = pd.read_excel(self.gdp)
        self.progress_step()
        
        # forward fill ftse data to fill weekends
        vix = pd.read_excel(self.vix)
        vix = self.forwardfill_data(vix)
        self.progress_step()
        
        # forward fill ftse data to fill weekends
        ftse100 = pd.read_excel(self.ftse100)
        ftse100 = self.forwardfill_data(ftse100)
        self.progress_step()
        
        # build database
        self.build_database(pricedata, security_master, findata, nomcurve, inflcurve, gdp, vix, ftse100)
        
        # final step to push progress bar past 100% to trigger the 'complete' condition
        self.progress_step()
