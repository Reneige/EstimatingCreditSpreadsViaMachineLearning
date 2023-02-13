# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 20:55:44 2023

@author: Rene Alby

The purpose of this tool is to aggregate data for research in machine learning applications in corporate 
bond markets. The tool is designed so that research data can be dropped into the './data/GBP Bonds' subfolder 
and be immediately integrated into the dataset by clicking the 'Rebuild Database' menu from the Database 
drop-down menu. Then the data can be browsed to ensure it is being captured correctly in the data model. 
Finally, clicking the 'Generate Research Data Table' button will run all the required queries to build the 
data set

"""

from tkinter import Tk, ttk, LabelFrame, Button, Scrollbar, Toplevel, Menu
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from threading import Thread
import xlwings as xw
import numpy as np

class ResearchQueryTool:
    def __init__(self, root):
        self.root = root
        self.database = r".\database.db"
        self.generate_view()
        self.generate_panel()
        self.generate_menu()
        self.generate_drop_down_menu()
        self.load_data(self.run_query(sql.mainview))
        
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

        self.button3 = Button(self.label_file, text="Generate Research Data Table (30 sec)", 
                              command=lambda: new_excel(self.build_research_dataset()))
        
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
        self.filemenu.add_command(label="Export Bond List", command=lambda: new_excel(self.run_query(sql.mainview)))
        
        self.datamenu.add_command(label="Browse Price Data", command=lambda: self.display_table_data('Prices'))
        self.datamenu.add_command(label="Browse Nominal Curve Data", command=lambda: self.display_table_data('Nominal_Curve'))
        self.datamenu.add_command(label="Browse Inflation Curve Data", command=lambda: self.display_table_data('Inflation_Curve'))
        self.datamenu.add_command(label="Browse ONS GDP Estimate Data", command=lambda: self.display_table_data('GDP'))
        self.datamenu.add_command(label="Browse VIX Data", command=lambda: self.display_table_data('VIX'))
        self.datamenu.add_command(label="Browse FTSE100 Data", command=lambda: self.display_table_data('FTSE100'))
        self.datamenu.add_separator()
        self.datamenu.add_command(label="Browse Financial Summary Data", command=lambda: self.display_table_data_subset('Financials','statement',"'Financial-Summary'"))
        self.datamenu.add_command(label="Browse Income Statement Data", command=lambda: self.display_table_data_subset('Financials','statement',"'Income-Statement'"))
        self.datamenu.add_command(label="Browse Balance Sheet Data", command=lambda: self.display_table_data_subset('Financials','statement',"'Balance-Sheet'"))
        self.datamenu.add_command(label="Browse Cash Flow Statement Data", command=lambda: self.display_table_data_subset('Financials','statement',"'Cash-Flow'"))
        self.datamenu.add_separator()
        self.datamenu.add_command(label="Browse Bond Master Data", command=lambda: self.popup_tree(self.run_query('SELECT * FROM Master ORDER BY Issuer')))
        
        # add schema menu 
        self.dbmenu.add_cascade(label="Schema", menu=self.schema)
        self.schema.add_command(label="Display Schema", command=lambda: self.popup_tree(self.run_query("SELECT * FROM sqlite_schema")))
        self.schema.add_separator()
        self.schema.add_command(label="Check Master Schema", command=lambda: self.inspect_table('Master'))
        self.schema.add_command(label="Check Prices Schema", command=lambda: self.inspect_table('Prices'))
        self.schema.add_command(label="Check Financials Schema", command=lambda: self.inspect_table('Financials'))
        self.schema.add_command(label="Check Nominal_Curve Schema", command=lambda: self.inspect_table('Nominal_Curve'))
        self.schema.add_command(label="Check Inflation_Curve Schema", command=lambda: self.inspect_table('Inflation_Curve'))
        self.schema.add_command(label="Check GDP Schema", command=lambda: self.inspect_table('GDP'))
        self.schema.add_command(label="Check VIX Schema", command=lambda: self.inspect_table('VIX'))
        self.schema.add_command(label="Check FTSE100 Schema", command=lambda: self.inspect_table('FTSE100'))
        self.schema.add_separator()
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
        self.popup_menu.add_command(label="TEST", command=lambda: self.popup_tree(self.run_query('SELECT Date, "1.5yr" FROM Nominal_Curve')))

       
        # create cascading menu - financials
        self.fins_menu = Menu(self.popup_menu, tearoff=0)
        self.popup_menu.add_cascade(label="Financials", menu=self.fins_menu)
        
        # A note on the below - these are mostly for testing whether the queries I need to build my research
        # Data set are working. They are not designed for readability. Sorry! 
        
        # add financials to cascade
        self.fins_menu.add_command(label="Browse Financial Summary Data", command=lambda: self.popup_tree(self.query_financials_by_isin("*",self.ISIN(),"Financial-Summary")))
        self.fins_menu.add_command(label="Browse Income Statement Data", command=lambda: self.popup_tree(self.query_financials_by_isin("*",self.ISIN(),"Income-Statement")))
        self.fins_menu.add_command(label="Browse Balance-Sheet Data", command=lambda: self.popup_tree(self.query_financials_by_isin("*",self.ISIN(),"Balance-Sheet")))
        self.fins_menu.add_command(label="Browse Cash Flow Data", command=lambda: self.popup_tree(self.query_financials_by_isin("*",self.ISIN(),"Cash-Flow")))
        self.fins_menu.add_separator()
        self.fins_menu.add_command(label="Total Assets", command=lambda: self.popup_tree(self.query_item_by_isin(sql.assets, self.ISIN(),0)))
        self.fins_menu.add_command(label="Total Liabilities", command=lambda: self.popup_tree(self.query_item_by_isin(sql.liabilities, self.ISIN(),0)))
        self.fins_menu.add_separator()
        self.fins_menu.add_command(label="Cash and Debt", command=lambda: self.popup_tree(self.merge_financial_item_by_isin(sql.cashflow, sql.debt, self.ISIN())))
        self.fins_menu.add_separator()
        self.fins_menu.add_command(label="Cash, Debt, Income, Asset", command=lambda: self.popup_tree(self.recursive_merge_financials_by_isin([sql.cashflow, sql.debt, sql.assets, sql.income], self.ISIN(),0)))
        self.fins_menu.add_command(label="Key Financials", command=lambda: self.popup_tree(self.recursive_merge_financials_by_isin(sql.key_fins_queries(), self.ISIN(),0)))
        self.fins_menu.add_command(label="Prices Merged", command=lambda: self.popup_tree(self.merge_prices(self.recursive_merge_financials_by_isin(sql.key_fins_queries(), self.ISIN(),0),self.ISIN())))
        self.fins_menu.add_command(label="Static Bond Data Added", command=lambda: self.popup_tree(self.add_static_bond_data(self.recursive_merge_financials_by_isin(sql.key_fins_queries(), self.ISIN(),0),'SeniorityType',self.ISIN())))

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
        ''' returns a transposed dataframe with reset index'''
        df = df.transpose()
        df = df.reset_index()
        return df

    def clean(self, df):
        ''' drops NaNs from dataframe and resets the index'''
        df = df.dropna()
        df = df.reset_index()
        return df        

    def query_financials_by_isin(self, columns, isin, statement):
        ''' Queries financial statement data by ISIN code by matching the ISIN to the Issuer Name via a 
            Sub Query on the 'Master' table, i.e. the security master or bond-lookup table '''
        
        query = f"""SELECT {columns} 
                     FROM Financials
                     WHERE Company = (SELECT Issuer 
                                      FROM Master 
                                      WHERE ISIN='{isin}') 
                     AND Statement='{statement}'"""
                     
        return self.run_query(query)

    def query_item_by_isin(self, item, isin, month_shift):
        ''' Here you can shift the dates forward using the months_shift parameter
            This is useful for lagging data / adjusting for look-ahead bias.
        '''
        
        # this requires ugly handling because just using '+3 month' can result in date falling on the
        # first day of the following month if following month has fewer days. (weak implementation by 
        # SQLite. So here I have set to first day of month, shifted by x months + 1, then gone back a day.
        
        month_shift += 1
        query = f"""SELECT DISTINCT DATETIME(PeriodEndDate, 'start of month', '+{month_shift} month', '-1 day') AS Date,
                    {item} 
                    FROM Financials
                    WHERE Company = (SELECT Issuer 
                                     FROM Master 
                                     WHERE ISIN='{isin}')"""
                     
        return self.run_clean_query(query)

    def merge_financial_item_by_isin(self, item, item2, isin):
        ''' Note: The raw research data is parsed in an unstructured manner and so does not have dates as a 
            primary key which results in duplication of dates when multiple items are queried. 
            To get around this, I have merged the dataframes by date
        '''
        df1 = self.query_item_by_isin(item, isin,0)
        df2 = self.query_item_by_isin(item2, isin,0)       
        return df1.merge(df2, on='Date')

    def recursive_merge_financials_by_isin(self, list_of_items, isin, month_shift):
        ''' recursively merges data through queries stored in a list '''
        
        df = self.query_item_by_isin(list_of_items.pop(), isin, month_shift)
        if len(list_of_items) > 0:
            return df.merge(self.recursive_merge_financials_by_isin(list_of_items, isin, month_shift), on='Date')
        else:
            return df     

    def merge_prices(self, df, isin):
        ''' This method brings all the price data into the dataframe by merging the dataframes on the date columns
        '''
        df1 = self.run_query(f"SELECT * FROM Prices WHERE ISIN = '{isin}' ORDER BY Date ASC")
        return df1.merge(df, on='Date', how='left').sort_values(by='Date',ascending=False)

    def merge_econ_data_by_date(self, df, query):
        ''' This queries economic data and merges it to a dataframe using date as a key
        '''
        df1 = self.run_query(query)
        return df.merge(df1, on='Date', how='left').sort_values(by='Date',ascending=False)

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
        self.load_data(self.run_query(sql.mainview))

    def inspect_table(self, db_table):
        ''' returns df table describing the columns and data types of a database table and 
            displays the information in a pop-up window containing a TreeView             '''
        df = self.run_query(f"PRAGMA table_info({db_table})")
        return self.popup_tree(df)

    def display_table_data(self, db_table):
        ''' queries an entire database table and displays the data in a pop-up window containing a TreeView '''
        df = self.run_query(f"SELECT * FROM {db_table}")
        return self.popup_tree(df)

    def display_table_data_subset(self, db_table, column, subset):
        ''' queries an entire database table and displays the data in a pop-up window containing a TreeView '''
        df = self.run_query(f"SELECT * FROM {db_table} WHERE {column} = {subset}")
        return self.popup_tree(df)

    def display_fin_table_by_isin(self, db_table, column, subset):
        ''' queries an entire database table and displays the data in a pop-up window containing a TreeView '''
        df = self.run_query(f"SELECT * FROM {db_table} WHERE {column} = {subset}")
        return self.popup_tree(df)


    def build_research_dataset(self):
        ''' Produces main research data table. Note :            
            When building the research data set, financials are extracted first with their dates
            pushed forward 3 months with month_shift parameter. Then the price data is JOINED on that 
            future date. This ensures prices are 3 months after financial reporting dates. 
        '''
        df = self.run_query("SELECT DISTINCT ISIN FROM Prices")
        isins = df['ISIN'].tolist()
        agg =[]
        for isin in isins:
            data = self.recursive_merge_financials_by_isin([sql.revenue, sql.income, sql.cashflow, sql.assets, sql.liabilities,
                                         sql.debt, sql.current_assets, sql.current_liabilities], isin, month_shift=3)
            data = self.merge_prices(data, isin)
            
            # confusingly, the below forward fills the financial data, so, for example, the september 
            # 'total assets' also becomes the october and november, until there is a new data point.
            # it uses 'bfill' since it is not looking at the datetime info, unlike the df.resample() method)
            data = data.fillna(method='bfill')
            
            data = self.add_static_bond_data(data, 'SeniorityType', isin)
            data = self.add_static_bond_data(data, 'Coupon', isin)
            data = self.add_static_bond_data(data, 'CouponFrequency', isin)
            agg.append(data)
        data = pd.concat(agg)
        
        # Note - sqlite requires double quotes "" for columns that begin with number - this means I need single quotes for string to be python compatible
        data = self.merge_econ_data_by_date(data, sql.nominal_yield_1yr)
        data = self.merge_econ_data_by_date(data, sql.nominal_yield_3yr)      
        data = self.merge_econ_data_by_date(data, sql.nominal_yield_5yr)      
        data = self.merge_econ_data_by_date(data, sql.nominal_yield_10yr)      
        data = self.merge_econ_data_by_date(data, sql.be_inflation_5yr)      
        data = self.merge_econ_data_by_date(data, sql.gdp_gr_estimate)      
        data = self.merge_econ_data_by_date(data, sql.ftse_risk_return)
        data = self.merge_econ_data_by_date(data, sql.vix_usd)
        return data
         
    
def threadit(targ):
    tr = Thread(target=targ)
    tr.start()

def new_excel(df):
    wb = xw.Book()
    wb.app.activate(steal_focus=True)
    sht=wb.sheets[0]
    sht.range('A1').options(index=False, header=True).value=df

class sql:
    ''' A class for storing useful SQL queries'''
    
    # mainview generates the list of bonds to view in the tool
    mainview = """SELECT DISTINCT Prices.ISIN, Master.Issuer, Master.Coupon, strftime('%d-%m-%Y', Master.Maturity) AS Maturity
                       FROM Master 
                       INNER JOIN Prices
                       ON Master.'PreferredRIC' = Prices.ID"""
    
    # The below extract financial statement items and cast them to float/decimal
    cashflow = "CAST(NetCashFlowfromOperatingActivities AS Decimal) AS NetCashFlowfromOperatingActivities"
    debt = "CAST(DebtLongTermTotal AS Decimal) AS DebtLongTermTotal"
    assets = "CAST(TotalAssets AS Decimal) AS TotalAssets"  
    liabilities = "CAST(TotalCurrentLiabilities AS Decimal) + CAST(TotalNonCurrentLiabilities AS Decimal) AS TotalLiabilities"
    current_liabilities = "CAST(TotalCurrentLiabilities AS Decimal) AS TotalCurrentLiabilities"
    current_assets = "CAST(TotalCurrentAssets AS Decimal) AS TotalCurrentAssets"
    revenue = "CAST(RevenuefromBusinessActivitiesTotal AS Decimal) AS RevenuefromBusinessActivitiesTotal"
    income = "CAST(IncomebeforeTaxes AS Decimal) AS PretaxIncome"
    
    # The ResearchQueryTool.RecursiveMergeByIsin function takes a list of the above financial queries and pops them until the recursive 
    # function terminates. Therefore, I need to define a new list each time it is run. Using the function below solves this.
    # A single implementation of this list would only function once, because after the recursive function pops all elements, the list would be empty.
    
    def key_fins_queries():
        return [sql.revenue, sql.income, sql.cashflow, sql.assets, sql.liabilities, sql.debt, sql.current_assets, sql.current_liabilities]

    # All the below are economic data items: yields, inflation, gdp, ftse stdev/returns and VIX returns in usd.    
    nominal_yield_1yr = 'SELECT Date, "1yr" AS "1yr_nominal_gov_yield" FROM Nominal_Curve'
    nominal_yield_3yr = 'SELECT Date, "3yr" AS "3yr_nominal_gov_yield" FROM Nominal_Curve'      
    nominal_yield_5yr = 'SELECT Date, "5yr" AS "5yr_nominal_gov_yield" FROM Nominal_Curve'      
    nominal_yield_10yr = 'SELECT Date, "10yr" AS "10yr_nominal_gov_yield" FROM Nominal_Curve'      
    be_inflation_5yr = 'SELECT Date, "5yr" AS "5yr_breakeven_inflation" FROM Inflation_Curve'      
    gdp_gr_estimate = 'SELECT "2M Lagged Date" AS Date, "Gross Value Added Growth" AS "GDP_Growth_estimate" FROM GDP'
    ftse_risk_return = 'SELECT Date, "Daily Rolling 22 Day Sample StDev" AS FTSE_22_day_rolling_stdev,\
                        "Daily Rolling 22 Day Geometric Return" AS FTSE_22_day_rolling_return FROM FTSE100'
    vix_usd = 'SELECT Date, Close AS "VIX_Close" FROM VIX'

   
''' The below instantiates the main GUI and triggers the main loop ''' 
if __name__=='__main__':
    root = Tk()
    root.title("Research Data Tool")
    root.geometry("550x550")
    application = ResearchQueryTool(root)
    root.lift()
    root.mainloop()
    