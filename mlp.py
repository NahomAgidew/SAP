import numpy as np 
import pandas as pd 
import time, datetime
import matplotlib.pyplot as mpl 

def ParseData(path):
	df = pd.read_csv(path)
	dateStr = df['Date'].values
	D = np.zeros(dateStr.shape)

	for i, j in enumerate(dateStr):
		D[i] = time.mktime(datetime.datetime.strptime(j, '%Y-%m-%d').timetuple())
		#D[i] = dt.strptime(j, '%Y-%m-%d').timestamp()
		df['Timestamp'] = D
		return df.drop('Date', axis=1)

def PlotData(df, p=None):
	if(p is None):
		p = np.array([])

	c = np.array([i for i in range(df.shape[0]) if i not in p])
	ts = df.Timestamp.values
	nTicks = 10
	s = np.min(ts)
	e = np.max(ts)
	r = e - s
	s -= r / 5
	e += r / 5
	tickMarks = np.arange(s, e, (e-s)/nTicks)
	strTs = [datetime.datetime.fromtimestamp(i).strftime('%m-%d-%y') for i in tickMarks]
	mpl.figure()
	mpl.plot(ts, df.High.values, color = '#7092A8', linewidth = 1.618, label = 'Actual')

	if(len(p) > 0):
		mpl.plot(ts[p], df.High.values[p], color='#6F6F6F', linewidth='1.618', label='Predicted')

	mpl.xticks(tickMarks, strTs, rotation='vertical')
	mpl.legend(loc='upper left')
	mpl.show()

PlotData(ParseData('testge.csv'))