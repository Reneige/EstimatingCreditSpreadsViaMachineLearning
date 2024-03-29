# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 21:24:30 2023

@author: Rene Alby

Note all SQL is all hosted locally via SQLite so there is no risk of injection attack. 

The SQL Database is produced by parsing and formatting doanloaded data files in the data_aggregator.py code

"""


class sql:
    ''' A class for storing useful SQL queries '''

    # simple function to enrich a financial query by casting data stored as text to decimal form, and then renaming the column
    def tonumeric(fin_item):
        return f"CAST({fin_item} AS Decimal) AS {fin_item}"

    # mainview generates the list of bonds to view in the tool
    mainview = """SELECT DISTINCT Prices.ISIN, Master.Issuer, Master.Coupon, strftime('%d-%m-%Y', Master.Maturity) AS Maturity, Master.IssueDate, Master.FirstCouponDate, Master.CouponFrequency
                       FROM Master 
                       INNER JOIN Prices
                       ON Master.ISIN = Prices.ISIN
                       ORDER BY Master.Issuer"""
    
    # used to generate zspreads off all the available bond prices
    all_bond_data = """SELECT DISTINCT Prices.*, Master.Issuer, Master.Coupon, strftime('%d-%m-%Y', Master.Maturity) AS Maturity, Master.IssueDate, Master.FirstCouponDate, Master.CouponFrequency
                            FROM Master 
                            INNER JOIN Prices
                            ON Master.ISIN = Prices.ISIN
                            ORDER BY Prices.Date"""
    
    # schema
    schema = "SELECT * FROM sqlite_schema"
    schema_tables = "SELECT name FROM sqlite_schema"

    # The below extract financial statement items and cast them to float/decimal using the above function
    # many of these do not retrieve much data and cause issues, so in the end I used much fewer in the final list
    revenue = tonumeric("RevenuefromBusinessActivitiesTotal")
    income = tonumeric("IncomebeforeTaxes")
    cashflow = tonumeric("NetCashFlowfromOperatingActivities")
    assets = tonumeric("TotalAssets")
    liabilities = "CAST(TotalCurrentLiabilities AS Decimal) + CAST(TotalNonCurrentLiabilities AS Decimal) AS TotalLiabilities"
    debt = tonumeric("DebtTotal")
    current_liabilities = tonumeric("TotalCurrentLiabilities")
    current_assets = tonumeric("TotalCurrentAssets")
    debtlongterm = tonumeric("DebtLongTermTotal")
    intcover = tonumeric("InterestCoverageRatio")
    debtequit = tonumeric("TotalDebtPercentageofTotalEquity")
    debtcapital = tonumeric("TotalDebtPercentageofTotalCapital")
    current = tonumeric("CurrentRatio")
    quick = tonumeric("QuickRatio")
    wcta = tonumeric("WorkingCapitaltoTotalAssets")
    debtassets = tonumeric("TotalDebtPercentageofTotalAssets")
    freecashflow = tonumeric("FreeCashFlow")
    current_assets_2 = tonumeric("CurrentAssets")
    current_liabilities_2 = tonumeric("CurrentLiabilities")
    assets2 = tonumeric("Assets")
    netdebt = tonumeric("NetDebt")
    operating_profit = tonumeric("OperatingProfit")
    roa = tonumeric("PretaxROA")
    roe = tonumeric("PretaxROE")
    debtcaptial = tonumeric("TotalDebtPercentageofTotalCapital")
    debtequity = tonumeric("TotalDebtPercentageofTotalEquity")
    cash = tonumeric("CashCashEquivalents")

    '''    
    The ResearchQueryTool.RecursiveMergeByIsin() function takes a list of the above financial queries and pops them until the recursive
    function terminates. Therefore, I need to define a new list each time it is run. Using the function below solves this.
    A single implementation of this list would only function once, because after the recursive function pops all elements, the list would be empty.
    '''

    def key_fins_queries():
        return [sql.quick,
                sql.current,
                sql.intcover,
                sql.current_liabilities,
                sql.current_assets,
                sql.debt,
                sql.liabilities,
                sql.assets,
                sql.cashflow,
                sql.income,
                sql.revenue,
                sql.wcta,
                sql.debtassets,
                sql.freecashflow,
                sql.current_assets_2,
                sql.current_liabilities_2,
                sql.assets2,
                sql.netdebt,
                sql.operating_profit,
                sql.roa,
                sql.roe,
                sql.debtcaptial,
                sql.debtequity,
                sql.cash]

    def select_financials():
        return [sql.cashflow, 
                sql.debt, 
                sql.assets, 
                sql.income]
    
    def select_ratos():
        return [sql.intcover, 
                sql.debtequity, 
                sql.debtcapital, 
                sql.debtassets, 
                sql.wcta, 
                sql.current]
    
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
    results = 'SELECT * FROM Results'
