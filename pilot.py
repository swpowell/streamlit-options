import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import pandas_ta as pta
import pandas as pd

# """

# LEAP finder.
# Version: 1.0
# Date: 7 April 2021
# Updated: 2 December 2023
# Author: Scott W. Powell

# Purpose: Find long-dated call options with high realistic potential return on investment based 
# on growth assumptions consistent with growth over specified historical time period.
# By default, the historical time period is 5 years. Users can change this by modifying the "years"
# variable in the main() function located at the bottom of this code.
# "Long-dated" is considered any call option with an expiration date at least 1 year from current date.

# Data access based on yfinance module by Ran Aroussi (pypi.org/project/yfinance)

# """

#******************************************************

def findLEAPs(stock):

	from datetime import datetime as dt
	from numpy import array

	# Get tuple of options expiration dates.
	dates = array(stock.options)

	# Get number of days out for each option.
	deltas = [(dt.strptime(i,'%Y-%m-%d').date()-dt.today().date()).days for i in dates]
	 
	return dates[array(deltas)>365]	

#******************************************************

def getoptionsprices(stock,dates):

	from pandas import DataFrame
	from datetime import datetime as dt, timedelta as td
	import numpy as np

	# Initialize a dictionary that will hold one pandas data frame per expiration date.
	options = {}

	# Get the last stock bid price (includes AH trading).
	# price = stock.info['regularMarketPrice']	
	price = stock.history(period='1d',interval='1m')['Close'].iloc[-1]
	# print('Current price: $' + str(price))
	
	for i in dates:
	
		# Initialize a pandas data frame.
		data = DataFrame()

		# Load the option chain.
		chain = stock.option_chain(i)		

		# Get the last purchase dates for each strike (calls only).
		purchase = chain.calls.lastTradeDate
	
		# Get the last purchase dates. We'll exclude anything over 60 days old because it is probably too far OTM or ITM to bother with anyway.
		dts = [dt.strptime(str(i),'%Y-%m-%d %H:%M:%S%z') for i in purchase]
		deltas = np.array([(dt.today().date()-i.date()).days for i in dts])

		# Exclude all ITM calls as well since we generally won't use them for investing in LEAPs.
		# Could also use inTheMoney column.
		strikes = chain.calls.strike		

		# Include data with strike above price and last purchase date within last 60 days.
		cond = (strikes > price) & (deltas <= 60)		

		# Next, get information.
		# Strike price.
		data['Strike Price'] = chain.calls.strike[cond]		
		# Bid price.
		data['Bid Price'] = chain.calls.bid[cond]
		# Ask price.
		data['Ask Price'] = chain.calls.ask[cond]
		# Volume.
		data['Daily Volume'] = chain.calls.volume[cond]

		# Add data frame to the parent dictionary. Reset the indexing.
		options[i] = data.reset_index(drop=True)

	return options

#******************************************************

def simulateMC(stock,years):

	# Conduct 1000 simulations of potential future price of stock using 
	# last 5 years of growth as a predictor for future growth.
	# Predictions will extend to two years beyond the current date.

	import numpy as np
	from pandas import DataFrame
	from datetime import datetime as dt, timedelta as td
	# from termcolor import colored

	numsims = 1000	# Controls number of simulations. This can be changed by advanced users.

	# Get weekly history for last 5 years. Exclude this week since it may be unfinished.
	period = {1:"1y",2:"2y",5:"5y"}
	history = stock.history(period=period[years],interval="1wk")[:-1]	
	# if len(history) < np.ceil(years*52.17):
	# 	print( colored('Warning','red') + ': Less than ' + str(years)  + ' year(s) of data exists for this stock!') 

	# Remove dividend rows.
	history = history[history.Dividends==0]

	# Compute percent change per week.
	#pct = (history['Close']-history['Open'])/history['Open']
	pct = [(history['Close'][i]-history['Close'][i-1]) / \
		history['Close'][i-1] \
		for i in np.arange(1,len(history))]
	pct = np.array(pct)
	pct = pct[~np.isnan(pct)]

	# For now, assume that gains and losses are normally distributed about the mean.
	mean, stdev = 1+np.mean(pct), np.std(pct)

	# Get the last stock bid price (includes AH trading).	
	# price = stock.info['regularMarketPrice']
	price = stock.history(period='1d',interval='1m')['Close'].iloc[-1]

	# Embedded Monte Carlo function.
	def montecarlo(numsims,weeks):
		from numpy.random import default_rng
		rng = default_rng()
		dummy = np.zeros([numsims,weeks])
		for i in range(numsims):
			dummy[i,0] = price
			for j in np.arange(1,weeks):
				dummy[i,j] = dummy[i,j-1]*(rng.normal(loc=mean,scale=stdev,size=1))
				# dummy[i,j] = dummy[i,j-1]*(np.random.normal(loc=mean,scale=stdev))
		return dummy

	# Let's name the columns based on date instead of number of weeks. This will help later when 
	# we are looking up appropriate column for determining expected values of various dated options.
	columnnames = [dt.strftime(dt.today().date()+td(i*7),"%Y-%m-%d") for i in range(125)]	
	
	# Make a prediction data frame. Simulate out 125 weeks (2 years)..
	return DataFrame(montecarlo(numsims,125),columns=columnnames)

#******************************************************

def multiplier(options,prediction):

	import numpy as np
	from datetime import datetime as dt

	# Loop through each option.
	for i in list(options.keys()):
		optiondate = dt.strptime(i,'%Y-%m-%d').date()		
	
		# Find the date in the prediction data frame that most closely matches optiondate.
		predictiondates = (np.array([dt.strptime(i,'%Y-%m-%d').date() for i in prediction.columns]))
		I = np.argmin(np.abs(optiondate-predictiondates))

		# Convert prediction date back to string and get various simulated outcomes.
		outcomes = prediction[dt.strftime(predictiondates[I],"%Y-%m-%d")]

		# Compute expected value.
		ev = np.round(np.median(outcomes),2)

		# Add column to dataframe for each option.
		options[i]['Exp. Value'] = np.maximum(0,ev-options[i]['Strike Price'])
		options[i]['Exp. Fraction'] = np.round(options[i]['Exp. Value']/options[i]['Ask Price'],2)

	return options

#******************************************************

def getRSI(ticker,period='1y',interval='1d'):

    # Relative Strength Index (Uses Wilders averaging by default; I'm used to exponential)
	stock = yf.Ticker(ticker)
	prices = stock.history(period=period,interval=interval)['Close']
	RSI = round(pta.rsi(prices, length=14)[-1],2)

	return RSI

#******************************************************

@st.cache_resource
def liststock(ticker,years):

	output = {}

	try:
		# Create object for ticker.
		output['stock'] = yf.Ticker(ticker)

		# Get the dates of LEAPs for the stock ticker.
		dates = findLEAPs(output['stock'])
		
		# Get the latest bid/ask/strike prices and recent volume on each LEAP.
		# Will only show strike prices with purchase in last 60 days.
		# (This helps avoid analysis of illiquid options.))
		output['options'] = getoptionsprices(output['stock'],dates)

		# Next, simulate the future price of each stock using Monte Carlo approach based on historical data.
		output['prediction'] = simulateMC(output['stock'],years)	
	
		# Finally, compute expected value for the various options and add to options tables.	
		output['options'] = multiplier(output['options'],output['prediction'])		

		# return stock, prediction, options
		return output
	
	except:
		#sys.exit('Invalid ticker name entered.')
		return None	
	
#******************************************************

def run(ticker,years):

	# Get pandas tables of LEAP information, including potential growth.
	# print('\nGetting options information and making predictions...')
	output = liststock(ticker,years)

	if output is None:
		pass
		# print('\nSomething went wrong. Perhaps an invalid ticker was entered. \
		# 	If the error repeatedly occurs, there may be a problem with the Yahoo \
		# 	Finance API and you may need to wait and try again later.')
	else:
		return output

#******************************************************

def display_dataframes(options):
    # Define the tab names and dataframes to display

    # # Create the tabs and display the dataframes
    # selected_tab = st.sidebar.selectbox('Select a tab', list(options.keys()))
    # selected_df = options[selected_tab]
    # st.write('LEAPS info for ' )
    # st.write(selected_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    # Define the tab names and dataframes to display
    tabs = list(options.keys())
    num_tabs = len(tabs)
    tab_selected = st.session_state.get("tab_selected", tabs[0])

    # Create a container for the tab bar and data
    container = st.container()

    # Create the tab bar
    tab_col, data_col = container.columns((1, num_tabs))

    # Create the tabs and display the dataframes
    with data_col:
        tab_selected = st.selectbox("LEAPS info for", tabs, index=tabs.index(tab_selected), key="tab_radio")
        selected_df = options[tab_selected]
        # st.write(selected_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.dataframe(selected_df.set_index('Strike Price'))
	
    # Update the selected tab
    st.session_state["tab_selected"] = tab_selected

#******************************************************

def plotmontecarlo(data):
	
    median = data.median()
    data = data[::20].transpose()

    # Create a list of traces for each stock
    traces = []
    for column in data.columns:
        trace = go.Scatter(x=data.index, y=data[column], name='MC run: ' + str(column), line=dict(color='white'),opacity=0.5)
        traces.append(trace)
    medtrace = go.Scatter(x=median.index,y=median.values, name='Median', line=dict(color='red',width=5))
    traces.append(medtrace)

    # Create the plotly figure
    fig = go.Figure(data=traces)

    # Add layout details
    fig.update_layout(title='Monte Carlo Outcomes',
                    xaxis_title='Date',
                    yaxis_title='Price',
		            showlegend=False)

    # Show the plot in streamlit
    st.plotly_chart(fig, use_container_width=True)

#******************************************************

def prerun():

	tickers = ['AAPL', 'ABT', 'ADBE', 'AMC', 'AMD', 'AMZN', 'BA', 'BABA', 'BAC', 'BBY', 'BX', 'C', 'CAT', 'CCL', \
	'CHWY', 'CI', 'CLX', 'CMCSA', 'CMG', 'COST', 'CRM', 'CRWD', 'CSX', 'CVX', 'DAL', 'DIS', 'EBAY', 'F', 'FDX', \
	'GME', 'GOOGL', 'GPS', 'GS', 'HAL', 'HD', 'HLT', 'HUM', 'JNJ', 'JPM', 'K', 'KO', 'KR', 'KSS', 'LMT', 'LOW', 'LUV', 'M', 'MA', \
	'MCD', 'META', 'MRNA', 'MS', 'MSFT', 'MU', 'NFLX', 'NKE', 'NSC', 'NVDA', 'NVAX', 'OXY', 'PANW', 'PEP', 'PFE', 'PG', 'PLTR', \
	'PYPL', 'QCOM', 'ROKU', 'SWBI', 'SBUX', 'T', 'TGT', 'TSCO', 'TSLA', 'UAL', \
	'UNH', 'V', 'VZ', 'WFC', 'WMT', 'XOM', 'SPY', 'QQQ']

	RSIdata = pd.DataFrame(columns = ['RSI1wk','RSI1dy','RSI1hr'],index=tickers)

	for ticker in tickers:
		try: RSIdata['RSI1wk'][ticker] = getRSI(ticker,period='2y',interval='1wk')
		except: continue
		try: RSIdata['RSI1dy'][ticker] = getRSI(ticker)
		except: continue
		try: RSIdata['RSI1hr'][ticker] = getRSI(ticker,period='1wk',interval='60m')
		except: continue

	RSIdata.to_csv('data/RSIdata.csv')

	# Ishares Sector ETFs
	# IYE: Energy
	# IYF: Financials
	# IYW: Technology
	# IYR: Real Estate
	# IYH: Healthcare
	# IYK: Consumer Staples
	# ITA: Aerospace and Defense
	# ITB: Construction
	# IDU: Utilities
	# IHE: Pharma

	ishares = ['IYE','IYF','IYW','IYR','IYH','IYK','ITA','ITB','IDU','IHE']
	labels = ['Energy (IYE)','Financials (IYF)','Technology (IYW)','Real Estate (IYR)','Healthcare (IYH)',\
	    'Consumer Staples (IYK)',' Aerospace/Defense (ITA)','Construction (ITB)','Utilities (IDU)',' Pharmaceuticals (IHE)']
	sectorRSI = pd.DataFrame(columns=ishares,index=['Daily RSI'])
	for ticker in ishares:
		try: sectorRSI[ticker]['Daily RSI'] = round(getRSI(ticker),2)
		except: continue
	sectorRSI.columns = labels

	sectorRSI.to_csv('data/sectorRSI.csv')

#******************************************************

def RSIcolorandtext(val):
	if float(val) < 30:
		color = 'red'
		text = 'oversold'
	elif float(val) >= 30 and float(val) <= 40:
		color = 'orange'
		text = 'near oversold'
	elif float(val) >= 60 and float(val) <= 70:
		color = 'lightblue'
		text = 'near overbought'
	elif float(val) > 70:
		color = 'green'
		text = 'overbought'
	else:
		color = 'white'
		text = 'neutral'
	return color, text

def highlight_cells(val):
	if float(val) < 30:
		color = 'red'
	elif float(val) >= 30 and float(val) <= 40:
		color = 'orange'
	elif float(val) >= 60 and float(val) <= 70:
		color = 'lightblue'
	elif float(val) > 70:
		color = 'green'
	else:
		color = 'white'
	return 'color: %s' % color

#******************************************************

# Get some information about several stocks beforehand to make recommendations
# about what is overbought and oversold on daily timescales.

# RSIdata = prerun()

# Get pre-written data
RSIdata = pd.read_csv('data/RSIdata.csv',index_col=0).transpose()
sectorRSI = pd.read_csv('data/sectorRSI.csv',index_col=0)

overbought = [key for key in RSIdata if RSIdata[key]['RSI1dy'] > 70]
oversold = [key for key in RSIdata if RSIdata[key]['RSI1dy'] < 30]

# Drop down for overbought and oversold stocks in sidebar.
if len(overbought) > 0:
	selected = st.sidebar.selectbox('Overbought Stocks', overbought)
if len(oversold) > 0:
	selected = st.sidebar.selectbox('Oversold Stocks', oversold)

# List of sectors with daily RSI in certain color.
# Under 30: red (oversold)
# 30–40: orange (nearing oversold)
# 40-60: neutral (white)
# 60-70: blue (nearing overbought)
# Over 70: green (overbought)

st.sidebar.write('')
st.sidebar.write('Sector analysis')
# st.sidebar.write('Energy: ' + str(sectorRSI['IYE']['RSIdaily']))
sectorRSI = sectorRSI.transpose().applymap('{:.2f}'.format)

# st.sidebar.table(sectorRSI.transpose().style.format("{:.2f}"))
st.sidebar.table(sectorRSI.style.applymap(highlight_cells))

ticker = st.text_input('Enter a stock ticker (e.g. AAPL) for more information',selected)

if ticker:

	st.subheader('Basic information')

	# Run the code to do the MC simulations and get LEAPS info.
	output = run(ticker,5)
	stock, prediction, options = output['stock'], output['prediction'], output['options']

	RSI1hr = getRSI(ticker,period='1wk',interval='1h')
	RSI1dy = getRSI(ticker)
	RSI1wk = getRSI(ticker,period='2y',interval='1wk')
	cRSI1hr, tRSI1hr = RSIcolorandtext(RSI1hr)
	cRSI1dy, tRSI1dy = RSIcolorandtext(RSI1dy)
	cRSI1wk, tRSI1wk = RSIcolorandtext(RSI1wk)

	# NOTE: Bug: There seems to be a bug with showing options tables for stocks
	# that do not have a strike date available as late as the latest date available
	# for the first stock analyzed when running the code. Probably need to clear the
	# cache every time a new ticker is entered.

	# Get price info.
	previous = stock.history(period='1wk',interval='1d').sort_values('Date')['Close'].iloc[-2]
	price = stock.history(period='1d',interval='1m')['Close'].iloc[-1]
	previous = round(previous,2)
	price = round(price,2)

	if ~np.isnan(previous) and ~np.isnan(price):
		dprice = round((price-previous),2)

	# Set up columns for metrics.
	col1, col2 = st.columns(2)

	# Display the current price and change.
	col1.metric(label="Current Price", value='$'+str(price), delta=dprice,
	delta_color="normal")

	# col2.metric(label='RSI (weekly)',value=RSI1hr)
	col2.write('Hourly: ' + str(RSI1hr) + ' ('+tRSI1hr+')')
	col2.write('Daily: ' + str(RSI1dy) + ' ('+tRSI1dy+')')
	col2.write('Weekly: ' + str(RSI1wk) + ' ('+tRSI1wk+')')

	# Get stock price history and RSI history for 1-year chart.
	history = stock.history(period='5y',interval='1d')
	RSIhistory = round(pta.rsi(history['Close'], length=14),2)

	# Make a plot of the stock prices and RSI.
	# Create a Plotly figure with two scatter traces
	fig2 = go.Figure()
	# fig2.add_trace(go.Scatter(x=history.index, y=history,name='Price'),row=1,col=1)
	# Add a Candlestick trace for the first subplot
	fig2.add_trace(go.Candlestick(x=history.index, name='Price', open=history['Open'], high=history['High'], low=history['Low'], close=history['Close'],yaxis='y'))
	fig2.add_trace(go.Scatter(x=RSIhistory.index, y=RSIhistory,name='RSI',yaxis='y2',opacity=0.5))

	# Add the horizontal lines
	fig2.add_shape(type='line', xref='paper', yref='y2', opacity = 0.5, x0=0, y0=30, x1=1, y1=30, line=dict(color='magenta', width=2, dash='dash'))
	fig2.add_shape(type='line', xref='paper', yref='y2', opacity = 0.5, x0=0, y0=70, x1=1, y1=70, line=dict(color='magenta', width=2, dash='dash'))

	# Add a hover text that shows the values of both traces at the same x-axis value
	# Set the layout to include the second y-axis
	fig2.update_layout(
		yaxis=dict(title='Price', showgrid=False),
		yaxis2=dict(title='RSI', showgrid=False, overlaying='y', side='right'),
		hovermode='x unified', hoverlabel=dict(bgcolor="gray", font_size=16))

	# Display the Plotly figure in Streamlit
	st.plotly_chart(fig2)


	st.subheader('LEAPS information')

	showmc = st.checkbox('Show Monte Carlo Outcomes')
    # Add a button to the Streamlit app
	if showmc:
		# If the button is pressed, display the graph
		plotmontecarlo(prediction)

	    
	# Show the dataframes with a drop down menu to select the date of expiry for each LEAPS contract.	
	if options:
		display_dataframes(options)