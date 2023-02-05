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


class myTool:
    def __init__(self, root):
        self.root = root
        self.generate_view()
        self.generate_panel()
        self.generate_menu()
        self.generate_drop_down_menu()
        self.load_data(self.run_query(QUERY["Main"]))
    
    def generate_view(self):
        
        # create frame with title
        self.frame1 = LabelFrame(root, text="Database Viewer")
        self.frame1.place(height=250, width=520)
        
        # place treeview within frame
        self.tree_view = ttk.Treeview(self.frame1)
        self.tree_view.place(height=200, width=500, x=0, y=0)
        
        #add scrollbars to treeview
        self.treescrolly = Scrollbar(self.frame1, orient="vertical", command=self.tree_view.yview)
        self.treescrollx = Scrollbar(self.frame1, orient="horizontal", command=self.tree_view.xview)
        self.tree_view.configure(xscrollcommand=self.treescrollx.set, yscrollcommand=self.treescrolly.set)
        self.treescrollx.pack(side="bottom", fill="x")
        self.treescrolly.pack(side="right", fill="y")             
   
    def generate_panel(self):
        
        # Create frame with title
        self.label_file = LabelFrame(root, text="Control Panel")
        self.label_file.place(height=250, width=500, y=300)

        self.button3 = Button(self.label_file, text="Refresh View", command=lambda: new_excel(self.run_query(QUERY["Main"])))
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
        self.datamenu.add_separator()
        self.datamenu.add_command(label="Browse Financial Summary", command=lambda: self.popup_tree(self.run_query("SELECT * FROM Financials WHERE statement='Financial-Summary'"))) # WHERE Statement='Financial-Summary'")))
        self.datamenu.add_command(label="Browse Income Statement", command=lambda: self.popup_tree(self.run_query("SELECT * FROM Financials WHERE statement='Income-Statement'")))
        self.datamenu.add_command(label="Browse Balance Sheet", command=lambda: self.popup_tree(self.run_query("SELECT * FROM Financials WHERE statement='Balance-Sheet'")))
        self.datamenu.add_command(label="Browse Cash Flow Statement", command=lambda: self.popup_tree(self.run_query("SELECT * FROM Financials WHERE statement='Cash-Flow'")))
        self.datamenu.add_separator()
        self.datamenu.add_command(label="Browse Bond Master Data", command=lambda: self.popup_tree(self.run_query('SELECT * FROM Master ORDER BY Issuer')))
        
        # add schema menu 
        self.dbmenu.add_cascade(label="Schema", menu=self.schema)
        self.schema.add_command(label="List all Database Tables", command=lambda: self.popup_tree(self.run_query("SELECT tbl_name AS 'Table Names:' FROM sqlite_schema")))
        self.schema.add_command(label="Check Security Master Schema", command=lambda: self.popup_tree(self.run_query("PRAGMA table_info(Master)")))
        self.schema.add_command(label="Check Prices Schema", command=lambda: self.popup_tree(self.run_query("PRAGMA table_info(Prices)")))
        self.schema.add_command(label="Check Financials Schema", command=lambda: self.popup_tree(self.run_query("PRAGMA table_info(Financials)")))
        self.dbmenu.add_command(label="Rebuild Database", command=lambda: threadit(self.rebuild_database))

    
        # Add menu to root window        
        self.root.config(menu=self.topmenu)
        
    def generate_drop_down_menu(self):
        
        # creates a drop-down menu and binds it to the right click on the treeview
        self.popup_menu = Menu(self.tree_view, tearoff=0)
        self.popup_menu.add_command(label="Chart Prices", command=lambda: self.plot_query("Date","Bid","Prices","ISIN",self.ITEM()))
        self.popup_menu.add_command(label="Chart Yield", command=lambda: self.plot_query("Date","BidYld","Prices","ISIN",self.ITEM()))
        self.popup_menu.add_command(label="Chart Z-Spread", command=lambda: self.plot_query("Date","Z Spread","Prices","ISIN",self.ITEM()))
        self.popup_menu.add_separator()
        self.popup_menu.add_command(label="View Bond Info", command=lambda: self.popup_tree(self.transpose(self.run_query(f"SELECT * FROM Master WHERE ISIN = '{self.ITEM()}'"))))
        self.popup_menu.add_separator()
        self.popup_menu.add_command(label="Browse Price Data", command=lambda: self.popup_tree(self.run_query(f"SELECT * FROM Prices WHERE ISIN = '{self.ITEM()}' ORDER BY Date ASC")))
        self.popup_menu.add_command(label="Browse Financial Summary Data", command=lambda: self.popup_tree(self.query_financials_by_isin("*",self.ITEM(),"Financial-Summary")))
        self.popup_menu.add_command(label="Browse Income Statement Data", command=lambda: self.popup_tree(self.query_financials_by_isin("*",self.ITEM(),"Income-Statement")))
        self.popup_menu.add_command(label="Browse Balance-Sheet Data", command=lambda: self.popup_tree(self.query_financials_by_isin("*",self.ITEM(),"Balance-Sheet")))
        self.popup_menu.add_command(label="Browse Cash Flow Data", command=lambda: self.popup_tree(self.query_financials_by_isin("*",self.ITEM(),"Cash-Flow")))
        self.popup_menu.add_command(label="TEST")

        self.tree_view.bind("<Button-3>", self.context_menu)

    def context_menu(self, event):
        
        # captures the right-click event and sets the row to item
        self.item =self.tree_view.identify_row(event.y)       
        try:
            self.tree_view.selection_set(self.item)
            self.popup_menu.tk_popup(event.x_root, event.y_root, 0)
        finally:
            self.popup_menu.grab_release()
    
    def transpose(self, df):
        df = df.transpose()
        df = df.reset_index()
        return df
    
    def query_financials_by_isin(self, columns, isin, statement):
        query = f"""SELECT {columns} 
                     FROM Financials
                     WHERE Company = (SELECT Issuer 
                                      FROM Master 
                                      WHERE ISIN='{isin}') 
                     AND Statement='{statement}'"""
                     
        return self.run_query(query)
        
    def plot_query(self, xaxis,yaxis,table,target_col, target_id):                
        query = f"""SELECT {xaxis}, `{yaxis}` 
                    FROM {table} 
                    WHERE {target_col} = '{target_id}' 
                    ORDER BY {xaxis} ASC"""               
        return self.popup_scatter_plot(self.run_query(query),xaxis,yaxis,target_id)

    def run_query(self, query):
        print(query)
        conn = sqlite3.Connection(r".\database.db")
        df = pd.read_sql(f"{query}", conn)
        conn.close()
        df.dropna(how='all', axis=1, inplace=True)
        return df

    def load_data(self, df):
        
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
        self.tree_view.delete(*self.tree_view.get_children())

    def popup_scatter_plot(self, df,x,y,title):
        pop_up = Toplevel()
        pop_up.title('plot viewer')
        figure = plt.Figure(figsize=(10,5), dpi=100)
        ax = figure.add_subplot(111)
        chart_type = FigureCanvasTkAgg(figure, pop_up)
        chart_type.get_tk_widget().pack()
        df.plot(x=x, y=y, style='.', legend=True, ax=ax)
        ax.set_title(title)

    def popup_tree(self, df):        
        # create popup window with tree view and data export functionality
        pop_up = Toplevel()
        pop_up.title('data viewer')
        pop_up.geometry("700x300")
        tree_view = ttk.Treeview(pop_up)
        tree_view.place(width=690, height=290, x=0, y=0)  
               
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
        treescrollx.pack(side="bottom", fill="x")
        treescrolly.pack(side="right", fill="y")   

        # Create Export Menu
        popup_menu = Menu(pop_up, tearoff=0)
        popup_submenu = Menu(popup_menu, tearoff=0)
        popup_menu.add_cascade(label="Export Data", menu=popup_submenu)
        popup_submenu.add_command(label="Export to Excel", command=lambda: new_excel(df))
        pop_up.config(menu=popup_menu)

    def rebuild_database(self):
        # all the work here is done in the imported database_builder class
        from data_aggregator import database_builder
        db = database_builder()
        
        # refresh view when database is rebuilt
        self.load_data(self.run_query(QUERY["Main"]))
  
    def ITEM(self):
        return self.tree_view.item(self.item)['values'][0]
    
def threadit(targ):
    tr = Thread(target=targ)
    tr.start()

def new_excel(df):
    wb = xw.Book()
    wb.app.activate(steal_focus=True)
    sht=wb.sheets[0]
    sht.range('A1').options(index=False, header=True).value=df

    
if __name__=='__main__':

    root = Tk()
    root.title("Database Viwer")
    root.geometry("550x550")
    application = myTool(root)
    root.mainloop()
    
QUERY = {"Main2":"SELECT DISTINCT ISIN, Issuer, Coupon, strftime('%d-%m-%Y', Maturity) AS Maturity FROM Master",
         "Main":"""SELECT DISTINCT Prices.ISIN, Master.Issuer, Master.Coupon, strftime('%d-%m-%Y', Master.Maturity) AS Maturity
                   FROM Master 
                   INNER JOIN Prices
                   ON Master.'Preferred RIC' = Prices.ID"""}
