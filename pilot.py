import yfinance as yf
import streamlit as st
import plotly.graph_objs as go

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
		options[i]['Implied Expiration Value'] = np.maximum(0,ev-options[i]['Strike Price'])
		options[i]['Implied Factoral Gain'] = np.round(options[i]['Implied Expiration Value']/options[i]['Ask Price'],2)

	return options

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

st.title('LEAPS options information')

ticker = st.text_input('Enter a stock ticker (e.g. AAPL)')

if ticker:
	# Run the code to do the MC simulations and get LEAPS info.
	output = run(ticker,5)
	stock, prediction, options = output['stock'], output['prediction'], output['options']

	# Get price info.
	previous = stock.history()['Close'].iloc[-1]
	price = stock.history(period='1d',interval='1m')['Close'].iloc[-1]
	previous = round(previous,2)
	price = round(price,2)
	dprice = round(price-previous)

	# Display the current price and change.
	st.metric(label="Current Price", value='$'+str(price), delta=dprice,
	delta_color="inverse")

    # Add a button to the Streamlit app
	if st.button("Show Monte Carlo outcomes"):
		# If the button is pressed, display the graph
		plotmontecarlo(prediction)
		if st.button('Close Graph'):
			# hide the plot by setting the plotly_chart to None
			st.plotly_chart(None)
	    
	# Show the dataframes with a drop down menu to select the date of expiry for each LEAPS contract.	
	if options:
		display_dataframes(options)