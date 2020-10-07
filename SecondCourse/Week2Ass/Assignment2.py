import matplotlib.pyplot as plt
import mplleaflet
import pandas as pd
import matplotlib.dates as mdates
import datetime as dt
import numpy as np

def leaflet_plot_stations(binsize, hashid):

    df = pd.read_csv('data/BinSize_d{}.csv'.format(binsize))

    station_locations_by_hash = df[df['hash'] == hashid]

    lons = station_locations_by_hash['LONGITUDE'].tolist()
    lats = station_locations_by_hash['LATITUDE'].tolist()

    plt.figure(figsize=(8,8))
    plt.scatter(lons, lats, c='r', alpha=0.7, s=200)
    return mplleaflet.show()

# leaflet_plot_stations(400,'fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89')



def main():
    df = pd.read_csv('fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv')
    df.sort_values('Date', inplace=True)


    df_tmp = df.replace({'Date': r'\d{4}-'}, {'Date' : ''}, regex=True)


    df_2015 = df.copy()

    df_2015['Date'] = pd.to_datetime(df_2015['Date'])
    df_2015 = df_2015[df_2015['Date'].dt.year == 2015]
    low_2015 = df_2015.groupby('Date').min('Data_Value')
    high_2015 = df_2015.groupby('Date').max('Data_Value')

    record_low_2015 = []
    record_high_2015 = []

    # print(high_2015['Data_Value'])

    # Getting the Temperatures, i.e. Y-axis
    record_low = df_tmp.groupby('Date').min('Data_Value')
    record_high = df_tmp.groupby('Date').max('Data_Value')

    record_high.drop('02-29', axis=0, inplace=True)
    record_low.drop('02-29', axis=0, inplace=True)

    low_2015.index = record_low.index
    high_2015.index = record_high.index

    low_2015 = low_2015[record_low['Data_Value'] == low_2015['Data_Value']]
    high_2015 = high_2015[record_high['Data_Value'] == high_2015['Data_Value']]

    # print(low_2015)
    print(high_2015)



    # for i in range(len(low_2015)):
    #     if(low_2015['Date'][i] == record_low['Date'][i]):
    #        record_low_2015.append(low_2015[i])
    #     if (high_2015['Date'][i] == record_high['Date'][i]):
    #         record_high_2015.append(high_2015['Date'][i])



    # Getting the dates right, i.e. X-axis
    observation_dates = np.arange('2017-01-01', '2018-01-01', dtype='datetime64[D]')
    observation_dates = list(map(pd.to_datetime, observation_dates))

    plt.plot(record_low/10, 'k', record_high/10, 'k')


    # To fill the space between the lines
    plt.gca().fill_between(range(len(record_low)),
                           record_low['Data_Value']/10, record_high['Data_Value']/10,
                           facecolor='blue',
                           alpha=0.25)


    colors_high = ['red'] * (len(high_2015))
    colors_low = ['blue'] * (len(low_2015))

    plt.scatter(high_2015.index, high_2015 / 10, s = 50, c = colors_high)
    plt.scatter(low_2015.index, low_2015/10, s = 50, c = colors_low)

    ax = plt.gca()
    x = ax.xaxis

    # Divide into separate months and present as jan, feb etc.
    locator = mdates.MonthLocator()  # every month
    x.set_major_locator(locator)
    fmt = mdates.DateFormatter('%b')
    x.set_major_formatter(fmt)

    # Rotate the ticklables for more space.
    # for item in x.get_ticklabels():
    #     item.set_rotation(45)

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)


    ax.set_ylabel('Temperature ($\circ C$)')
    # ax.set_xlabel('Date')
    ax.set_title('Max and min temperature for each day during the period 2005-2015')

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(bottom=False, left=False, labelleft='off', labelbottom='on')


    plt.show()




if __name__ == "__main__":
    main()