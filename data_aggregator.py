# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 13:40:56 2023

@author: Rene Alby
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
        self.total_files = len(self.price_files) + len(self.financial_files) + len(self.bond_universe)
        self.progress()
        self.run()
        
        
    def read_price(self,file):
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
        uni = pd.read_excel(file)
        self.progress_step()       
        return uni

    def build_database(self, pricedata, security_master, findata):
        # Build the database
        conn = sqlite3.Connection(r'.\database.db')
        pricedata.to_sql('Prices',conn, if_exists='replace', index=False)   
        security_master.to_sql('Master',conn, if_exists='replace', index=False)   
        findata.to_sql('Financials',conn, if_exists='replace', index=False)   
        conn.close()

    def progress(self):
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
        if self.pb['value'] < 100:
            self.pb['value'] += (100 / self.total_files)
            self.value_label['text'] = self.update_progress_label()
        else:
            messagebox.showinfo(message='Database Build Complete')
            self.pb_popup.destroy()


    def run(self):
        # aggregate price data into dataframes
        list_of_dfs = list(map(self.read_price,self.price_files))
        pricedata = pd.concat(list_of_dfs)
        
        # aggregate financial data
        # the handling here is different because it is a list of lists, so need to flatten the list first 
        list_of_dfs2 = list(map(self.read_financials,self.financial_files))
        list_of_dfs2 = [item for sublist in list_of_dfs2 for item in sublist]
        findata = pd.concat(list_of_dfs2)
       
        # convert columns to lower case and drop duplicates
        findata = findata.loc[:,~findata.columns.str.lower().duplicated()]
      
        # aggregate bond universe / security master data
        list_of_dfs3 = list(map(self.read_bond_universe_master,self.bond_universe))
        security_master = pd.concat(list_of_dfs3)
       
        # build database
        self.build_database(pricedata, security_master, findata)
        self.progress_step()
