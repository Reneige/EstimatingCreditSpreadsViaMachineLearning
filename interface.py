# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 20:55:44 2023

@author: Rene Alby
"""

from tkinter import Tk, ttk, LabelFrame, Button, Scrollbar, Toplevel, Menu
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from threading import Thread
import xlwings as xw
import numpy as np

class myTool:
    def __init__(self, root):
        self.root = root
        self.database = r".\database.db"
        self.generate_view()
        self.generate_panel()
        self.generate_menu()
        self.generate_drop_down_menu()
        self.load_data(self.run_query(QUERY["Main"]))
        
    ''' ------------------------------ GUI Methods  ----------------------------------'''
    
    def generate_view(self):
        
        # create frame with title
        self.frame1 = LabelFrame(root, text="Database Viewer")
        self.frame1.place(height=250, width=550)
        
        # place treeview within frame
        self.tree_view = ttk.Treeview(self.frame1)
        self.tree_view.place(height=220, width=530, x=0, y=0)
        
        #add scrollbars to treeview
        self.treescrolly = Scrollbar(self.frame1, orient="vertical", command=self.tree_view.yview)
        self.treescrollx = Scrollbar(self.frame1, orient="horizontal", command=self.tree_view.xview)
        self.tree_view.configure(xscrollcommand=self.treescrollx.set, yscrollcommand=self.treescrolly.set)
        self.treescrollx.pack(side="bottom", fill="x")
        self.treescrolly.pack(side="right", fill="y")             
   
    def generate_panel(self):
        
        # Create frame with title
        self.label_file = LabelFrame(root, text="Control Panel")
        self.label_file.place(height=250, width=550, y=250)

        self.button3 = Button(self.label_file, text="Generate Research data Table (30 sec)", 
                              command=lambda: new_excel(self.execute()))
        
        self.button3.grid(row=0, column=4)


    def generate_menu(self):
        
        # Create top level menu at root
        self.topmenu = Menu(self.root, tearoff=0)
        
        # Add submenus 
        self.filemenu = Menu(self.topmenu, tearoff=0)
        self.datamenu = Menu(self.topmenu, tearoff=0)
        self.dbmenu = Menu(self.topmenu, tearoff=0)
        self.schema = Menu(self.dbmenu, tearoff=0)
        
        # Add submenu cascade elements
        self.topmenu.add_cascade(label="File", menu=self.filemenu)
        self.topmenu.add_cascade(label="Data", menu=self.datamenu)
        self.topmenu.add_cascade(label="Database", menu=self.dbmenu)
        
        # Add menu commands
        self.filemenu.add_command(label="Export Bond List", command=lambda: new_excel(self.run_query(QUERY["Main"])))
        
        self.datamenu.add_command(label="Browse Price Data", command=lambda: self.popup_tree(self.run_query('SELECT * FROM Prices')))
        self.datamenu.add_command(label="Browse Nominal Curve Data", command=lambda: self.popup_tree(self.run_query('SELECT * FROM Nominal_Curve')))
        self.datamenu.add_command(label="Browse Inflation Curve Data", command=lambda: self.popup_tree(self.run_query('SELECT * FROM Inflation_Curve')))
        self.datamenu.add_command(label="Browse ONS GDP Estimate Data", command=lambda: self.popup_tree(self.run_query('SELECT * FROM GDP')))
        self.datamenu.add_command(label="Browse VIX Data", command=lambda: self.popup_tree(self.run_query('SELECT * FROM VIX')))
        self.datamenu.add_command(label="Browse FTSE100 Data", command=lambda: self.popup_tree(self.run_query('SELECT * FROM FTSE100')))
        self.datamenu.add_separator()
        self.datamenu.add_command(label="Browse Financial Summary Data", command=lambda: self.popup_tree(self.run_query("SELECT * FROM Financials WHERE statement='Financial-Summary'"))) 
        self.datamenu.add_command(label="Browse Income Statement Data", command=lambda: self.popup_tree(self.run_query("SELECT * FROM Financials WHERE statement='Income-Statement'")))
        self.datamenu.add_command(label="Browse Balance Sheet Data", command=lambda: self.popup_tree(self.run_query("SELECT * FROM Financials WHERE statement='Balance-Sheet'")))
        self.datamenu.add_command(label="Browse Cash Flow Statement Data", command=lambda: self.popup_tree(self.run_query("SELECT * FROM Financials WHERE statement='Cash-Flow'")))
        self.datamenu.add_separator()
        self.datamenu.add_command(label="Browse Bond Master Data", command=lambda: self.popup_tree(self.run_query('SELECT * FROM Master ORDER BY Issuer')))
        
        # add schema menu 
        self.dbmenu.add_cascade(label="Schema", menu=self.schema)
        
        # loop through database tables and build GUI elements and queries based on that
        for table in self.get_list_of_database_tables():
            self.schema.add_command(label=f"Check {table} Schema", command=lambda: self.popup_tree(self.run_query(f"PRAGMA table_info({table})")))
                 
        self.dbmenu.add_command(label="Rebuild Database", command=lambda: threadit(self.rebuild_database))
    
        # Add menu to root window        
        self.root.config(menu=self.topmenu)
        
    def generate_drop_down_menu(self):       
        ''' creates a drop-down context menu and binds it to the right click on the treeview '''
        
        self.popup_menu = Menu(self.tree_view, tearoff=0)
        self.popup_menu.add_command(label="Chart Prices", command=lambda: self.plot_query("Date","Bid","Prices","ISIN",self.ISIN()))
        self.popup_menu.add_command(label="Chart Yield", command=lambda: self.plot_query("Date","BidYld","Prices","ISIN",self.ISIN()))
        self.popup_menu.add_command(label="Chart Z-Spread", command=lambda: self.plot_query("Date","Z Spread","Prices","ISIN",self.ISIN()))
        self.popup_menu.add_separator()
        self.popup_menu.add_command(label="View Bond Info", command=lambda: self.popup_tree(self.transpose(self.run_query(f"SELECT * FROM Master WHERE ISIN = '{self.ISIN()}'"))))
        self.popup_menu.add_separator()
        self.popup_menu.add_command(label="Browse Price Data", command=lambda: self.popup_tree(self.run_query(f"SELECT * FROM Prices WHERE ISIN = '{self.ISIN()}' ORDER BY Date ASC")))
        self.popup_menu.add_separator()
        
        # create cascading menu - financials
        self.fins_menu = Menu(self.popup_menu, tearoff=0)
        self.popup_menu.add_cascade(label="Financials", menu=self.fins_menu)
        
        # add financials to cascade
        self.fins_menu.add_command(label="Browse Financial Summary Data", command=lambda: self.popup_tree(self.query_financials_by_isin("*",self.ISIN(),"Financial-Summary")))
        self.fins_menu.add_command(label="Browse Income Statement Data", command=lambda: self.popup_tree(self.query_financials_by_isin("*",self.ISIN(),"Income-Statement")))
        self.fins_menu.add_command(label="Browse Balance-Sheet Data", command=lambda: self.popup_tree(self.query_financials_by_isin("*",self.ISIN(),"Balance-Sheet")))
        self.fins_menu.add_command(label="Browse Cash Flow Data", command=lambda: self.popup_tree(self.query_financials_by_isin("*",self.ISIN(),"Cash-Flow")))
        self.fins_menu.add_separator()
        self.fins_menu.add_command(label="Total Assets", command=lambda: self.popup_tree(self.query_item_by_isin(assets, self.ISIN())))
        self.fins_menu.add_command(label="Total Liabilities", command=lambda: self.popup_tree(self.query_item_by_isin(liabilities, self.ISIN())))
        self.fins_menu.add_separator()
        self.fins_menu.add_command(label="Cash and Debt", command=lambda: self.popup_tree(self.merge_item_by_isin(cashflow, debt, self.ISIN())))
        self.fins_menu.add_separator()
        self.fins_menu.add_command(label="Cash, Debt, Income, Asset", command=lambda: self.popup_tree(self.recursive_merge([cashflow, debt, assets, income], self.ISIN())))
        self.fins_menu.add_command(label="Key Financials", command=lambda: self.popup_tree(self.recursive_merge([revenue, income, cashflow, assets, liabilities, debt, current_assets, current_liabilities], self.ISIN())))
        self.fins_menu.add_command(label="Prices Merged", command=lambda: self.popup_tree(self.merge_prices(self.recursive_merge([revenue, income, cashflow, assets, liabilities, debt, current_assets, current_liabilities], self.ISIN()),self.ISIN())))
        self.fins_menu.add_command(label="Static Bond Data Added", command=lambda: self.popup_tree(self.add_static_bond_data(self.recursive_merge([revenue, income, cashflow, assets, liabilities, debt, current_assets, current_liabilities], self.ISIN()),'SeniorityType',self.ISIN())))

        # bind right-click to the context menu method
        self.tree_view.bind("<Button-3>", self.context_menu)

    def context_menu(self, event):       
        ''' captures the right-click event and sets the row clicked to item instance variable '''
        
        self.item =self.tree_view.identify_row(event.y)       
        try:
            self.tree_view.selection_set(self.item)
            self.popup_menu.tk_popup(event.x_root, event.y_root, 0)
        finally:
            self.popup_menu.grab_release()

    def load_data(self, df):
        ''' method used to populate data into the main data viewer / TreeView '''
        
        # clear data
        self.clear_data()
        
        # initialise treeview columns to dastaframe columns
        self.tree_view["column"] = list(df.columns)
        self.tree_view["show"] = "headings"
        
        # set columns names
        for column in self.tree_view["columns"]:
            self.tree_view.heading(column, text=column)
        
        # push data into columns
        df_rows = df.to_numpy().tolist()
        for row in df_rows:
            self.tree_view.insert("","end", values=row)
        
    def clear_data(self):
        ''' method used to clear data from the main data viewer / TreeView '''
        self.tree_view.delete(*self.tree_view.get_children())

    def popup_scatter_plot(self, df,x,y,title):
        ''' creates a pop-up window displaying a scatter plot of a dataframe '''
        pop_up = Toplevel()
        pop_up.title('plot viewer')
        figure = plt.Figure(figsize=(10,5), dpi=100)
        ax = figure.add_subplot(111)
        chart_type = FigureCanvasTkAgg(figure, pop_up)
        chart_type.get_tk_widget().pack()
        df.plot(x=x, y=y, style='.', legend=True, ax=ax)
        ax.set_title(title)

    def popup_tree(self, df):        
        ''' creates a pop-up window displaying the data contents of a dataframe with export to excel functionality '''
        
        # create popup window 
        pop_up = Toplevel()
        pop_up.title('data viewer')
        pop_up.geometry("700x300")
        tree_view = ttk.Treeview(pop_up)
        
        #The below commands makes the Treeview data resize upon window resizing
        pop_up.grid_rowconfigure(0, weight=1)
        pop_up.grid_columnconfigure(0, weight=1)
        tree_view.grid(column=0, row=0, sticky="nsew")
        
        # initialise treeview columns to dastaframe columns
        tree_view["column"] = list(df.columns)
        tree_view["show"] = "headings"

        # set columns names
        for column in tree_view["columns"]:
            tree_view.heading(column, text=column)
        
        # push data into columns
        df_rows = df.to_numpy().tolist()
        for row in df_rows:
            tree_view.insert("","end", values=row)

        #add scrollbars to freeview on pop-up
        treescrolly = Scrollbar(pop_up, orient="vertical", command=tree_view.yview)
        treescrollx = Scrollbar(pop_up, orient="horizontal", command=tree_view.xview)
        tree_view.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
        treescrollx.grid(column=0, row=1, sticky="nsew")
        treescrolly.grid(column=1, row=0, sticky="nsew")

        # Create Export Menu
        popup_menu = Menu(pop_up, tearoff=0)
        popup_submenu = Menu(popup_menu, tearoff=0)
        popup_menu.add_cascade(label="Export Data", menu=popup_submenu)
        popup_submenu.add_command(label="Export to Excel", command=lambda: new_excel(df))
        pop_up.config(menu=popup_menu)

    def ISIN(self):
        ''' returns the ISIN of a bond clicked in the main data viewer'''
        return self.tree_view.item(self.item)['values'][0]


    ''' ------------------------------ Data Methods  ----------------------------------'''
    
    def transpose(self, df):
        df = df.transpose()
        df = df.reset_index()
        return df

    def clean(self, df):
        df = df.dropna()
        df = df.reset_index()
        return df

    def get_list_of_database_tables(self):
        df = self.run_query("SELECT tbl_name AS 'Table' FROM sqlite_schema")
        return df['Table'].tolist()
        
    
    def query_financials_by_isin(self, columns, isin, statement):
        query = f"""SELECT {columns} 
                     FROM Financials
                     WHERE Company = (SELECT Issuer 
                                      FROM Master 
                                      WHERE ISIN='{isin}') 
                     AND Statement='{statement}'"""
                     
        return self.run_query(query)

    def query_item_by_isin(self, item, isin):
        query = f"""SELECT DISTINCT PeriodEndDate AS Date,
                    {item} 
                    FROM Financials
                    WHERE Company = (SELECT Issuer 
                                     FROM Master 
                                     WHERE ISIN='{isin}')"""
                     
        return self.run_clean_query(query)

    def merge_item_by_isin(self, item, item2, isin):
        ''' Note: The raw research data is parsed in an unstructured manner and so does not have dates as a 
            primary key which results in duplication of dates when multiple items are queried. 
            To get around this, I have merged the dataframes by date
        '''
        df1 = self.query_item_by_isin(item, isin)
        df2 = self.query_item_by_isin(item2, isin)       
        return df1.merge(df2, on='Date')

    def recursive_merge(self, list_of_items, isin):
        ''' recursively merges data through queries stored in a list '''
        
        df = self.query_item_by_isin(list_of_items.pop(), isin)
        if len(list_of_items) > 0:
            return df.merge(self.recursive_merge(list_of_items, isin), on='Date')
        else:
            return df     

    def merge_prices(self, df, isin):
        ''' This method brings all the price data into the dataframe by merging the dataframes on the date columns
        '''
        df1 = self.run_query(f"SELECT * FROM Prices WHERE ISIN = '{isin}' ORDER BY Date ASC")
        return df1.merge(df, on='Date', how='left').sort_values(by='Date',ascending=False)

    def add_static_bond_data(self,df,item,isin):
        ''' queries the Master bond list of static data by isin and inserts as a column into a dataframe'''        
        df1 = self.run_query(f"SELECT {item} FROM Master WHERE ISIN = '{isin}'")
        try: 
            df[[item]] = df1[item].iloc[0]
        except:
            df[[item]] = 0
        return df
        
        
    def plot_query(self, xaxis,yaxis,table,target_col, target_id):                
        query = f"""SELECT {xaxis}, `{yaxis}` 
                    FROM {table} 
                    WHERE {target_col} = '{target_id}' 
                    ORDER BY {xaxis} ASC"""               
        return self.popup_scatter_plot(self.run_query(query),xaxis,yaxis,target_id)

    def run_query(self, query):
        ''' runs an SQLite query via Pandas and returns the dataframe '''
        print(query)
        conn = sqlite3.Connection(self.database)
        df = pd.read_sql(f"{query}", conn)
        conn.close()
        return df
    
    def run_clean_query(self, query):
        ''' runs an SQLite query via Pandas, replaces string nan with NaNs, drops NaN returns the dataframe '''
        print(query)
        conn = sqlite3.Connection(self.database)        
        df = pd.read_sql(f"{query}", conn)
        df = df.replace('nan', np.nan)
        df = df.dropna()
        conn.close()
        return df
    
    def rebuild_database(self):
        ''' parses all the raw research data and feeds into an SQLite database that use used to drive all 
            functionality in this tool. Note this completely deltes the database and then re-creates it.
        '''
        # all the work here is done in the imported database_builder class
        from data_aggregator import database_builder
        database_builder()
        
        # refresh view when database is rebuilt
        self.load_data(self.run_query(QUERY["Main"]))
  
    def execute(self):
        ''' Produces main research data table'''
        df = self.run_query("SELECT DISTINCT ISIN FROM Prices")
        isins = df['ISIN'].tolist()
        agg =[]
        for isin in isins:
            data = self.recursive_merge([revenue, income, cashflow, assets, liabilities,
                                         debt, current_assets, current_liabilities],isin)
            data = self.merge_prices(data, isin)
            data = self.add_static_bond_data(data, 'SeniorityType', isin)
            agg.append(data)
        return pd.concat(agg)
            
    
def threadit(targ):
    tr = Thread(target=targ)
    tr.start()

def new_excel(df):
    wb = xw.Book()
    wb.app.activate(steal_focus=True)
    sht=wb.sheets[0]
    sht.range('A1').options(index=False, header=True).value=df

    
if __name__=='__main__':
    
    # WORK IN PROGRESS BELOW
    QUERY = {"Main2":"SELECT DISTINCT ISIN, Issuer, Coupon, Maturity FROM Master",
             "Main":"""SELECT DISTINCT Prices.ISIN, Master.Issuer, Master.Coupon, strftime('%d-%m-%Y', Master.Maturity) AS Maturity
                       FROM Master 
                       INNER JOIN Prices
                       ON Master.'PreferredRIC' = Prices.ID"""}

   

    
    cashflow = "CAST(NetCashFlowfromOperatingActivities AS Decimal) AS NetCashFlowfromOperatingActivities"
    debt = "CAST(DebtLongTermTotal AS Decimal) AS DebtLongTermTotal"
    assets = "CAST(TotalAssets AS Decimal) AS TotalAssets"  
    liabilities = "CAST(TotalCurrentLiabilities AS Decimal) + CAST(TotalNonCurrentLiabilities AS Decimal) AS TotalLiabilities"
    current_liabilities = "CAST(TotalCurrentLiabilities AS Decimal) AS TotalCurrentLiabilities"
    current_assets = "CAST(TotalCurrentAssets AS Decimal) AS TotalCurrentAssets"
    revenue = "CAST(RevenuefromBusinessActivitiesTotal AS Decimal) AS RevenuefromBusinessActivitiesTotal"
    income = "CAST(IncomebeforeTaxes AS Decimal) AS PretaxIncome"


    
    # TotalLiabilities is a dupe column and the blank is being retained. Same may be happening elsewhere with total assets                          
                       

    root = Tk()
    root.title("Research Data Tool")
    root.geometry("550x550")
    application = myTool(root)
    root.lift()
    root.mainloop()
    
