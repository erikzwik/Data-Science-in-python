import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.linalg as sl
import matplotlib.pyplot as plt
import re

nhl_df = pd.read_csv("assets/nhl.csv")
cities = pd.read_html("assets/wikipedia_data.html")[1]
cities = cities.iloc[:-1, [0, 3, 5, 6, 7, 8]]

def explode(df, lst_cols, fill_value='', preserve_index=False):
    # make sure `lst_cols` is list-alike
    if (lst_cols is not None
        and len(lst_cols) > 0
        and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = (pd.DataFrame({
                col:np.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:
        res = res.reset_index(drop=True)
    return res



## This is currently specified for the NHL data. It would be good to give a bit more abstract meaning,
## So that I can use it for all the other leagues as well.

def nhl_correlation():

    nhl_df_2018 = nhl_df[nhl_df['year']==2018].copy()

    # Progression:
    # Transform the nhl_df_2018['team'] columns, so that only the last word remains, without the *
    # Transform cities['NHL'] such that Names are separated by capital letters and remove [note x]


    # Changing nhl_df_2018['team']
    index = 0
    for row in nhl_df_2018['team']:
        team_name = row.split(' ')
        if(team_name[-1][-1] == '*'):
            team_name[-1] = team_name[-1][0:-1]
        elif(team_name[-1] == 'Division'):
            nhl_df_2018.drop(index,inplace=True)
            index+=1
            continue
        nhl_df_2018.at[index, 'team'] = team_name
        for i in team_name:
            if(len(team_name) == 3 and team_name[1] in ['Angeles', 'York', 'Louis','Bay', 'Jersey']):
                nhl_df_2018.at[index,'team'] = team_name[-1]
            elif(len(team_name) == 3):
                nhl_df_2018.at[index, 'team'] = team_name[-2] + ' ' + team_name[-1]
            else:
                nhl_df_2018.at[index, 'team'] = team_name[-1]
        index += 1

    # print(nhl_df_2018)


    # Changing cities['NHL']
    index = 0
    # print(cities['NHL'])
    for row in cities['NHL']:
        row = row.split('[')[0]
        if(len(row) < 2):
            cities.drop(index, inplace=True)
            index += 1
            continue
        team_names = re.findall('[A-Z][^A-Z]*', row)
        if(team_names[0][-1] == ' '):
            cities.at[index, 'NHL'] = [team_names[0] + team_names[1]]
        else:
            cities.at[index, 'NHL'] = team_names



        index += 1

    cities_nhl = cities[['Metropolitan area', 'Population (2016 est.)[8]', 'NHL']].copy()
    cities_nhl.rename(columns = {'Population (2016 est.)[8]' : 'Population', 'NHL' : 'team'}, inplace = True)


    # This is kind of cheating, using already made code.. I'll have to understand it first..
    cities_nhl = explode(cities_nhl, ['team'], fill_value = '', preserve_index = True)

    # print(nhl_df_2018['team'])
    # print(cities_nhl['team'])

    merged_nhl = cities_nhl.merge(nhl_df_2018,how = 'right', on = 'team')
    merged_nhl.drop(['GP', 'OL', 'PTS',
       'PTS%', 'GF', 'GA', 'SRS', 'SOS', 'RPt%', 'ROW', 'year', 'League'], axis=1, inplace = True)
    # df = pd.DataFrame( {'Metropolitan area' : merged_nhl['Metropolitan area'], 'Population' :
    merged_nhl['Ratio'] = merged_nhl['W'].astype('float')/(merged_nhl['W'].astype('float') + merged_nhl['L'].astype('float'))
    merged_nhl.dropna(inplace=True)
    corr_nhl = merged_nhl['Population'].astype('float').corr(merged_nhl['Ratio'],method ='pearson')

    # Really just interestedd in 'Metropolitan area, Population, team, 'W' and 'L'.
    print(merged_nhl)


    plt.plot(merged_nhl['Population'].astype('float'), merged_nhl['Ratio'], '.')
    # plt.show()
    # print('Corelation is: ', corr_nhl)

    b = np.array(merged_nhl['Ratio'])

    A = np.array(merged_nhl['Population'].astype('float')).T
    print(np.shape(A))
    print(np.shape(np.ones(len(A))))
    A = np.array([A,np.ones(len(A))]).T
    print(A)
    a = sl.solve((A.T@A),A.T@b)
    x = np.linspace(0,25000000,30)
    y = x*a[0] + np.ones(30)*a[1]

    plt.plot(x,y)
    plt.show()

    population_by_region = []  # pass in metropolitan area population from cities
    win_loss_by_region = []  # pass in win/loss ratio from nhl_df in the same order as cities["Metropolitan area"]

    return merged_nhl

    # return stats.pearsonr(population_by_region, win_loss_by_region)


def main():
    print(nhl_correlation())

if __name__ == "__main__":
    main()

# assert len(population_by_region) == len(win_loss_by_region), "Q1: Your lists must be the same length"
# assert len(population_by_region) == 28, "Q1: There should be 28 teams being analysed for NHL"
