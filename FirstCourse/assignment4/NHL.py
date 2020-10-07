import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.linalg as sl
import matplotlib.pyplot as plt
import re


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
        col: np.repeat(df[col].values, lens)
        for col in idx_cols},
        index=idx)
           .assign(**{col: np.concatenate(df.loc[lens > 0, col].values)
                      for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens == 0, idx_cols], sort=False)
               .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:
        res = res.reset_index(drop=True)
    return res


def sport_correlation(df_sport, df_cities, sport, give_plot=False):
    df_sport = df_sport[df_sport['year'] == 2018].copy()

    index = 0
    for row in df_sport['team']:
        # Dealing here with special case of (x) in the name, of NBA
        row = row.split('(')
        if(len(row)>1):
            row = row[0][:-1]
        else:
            row = row[0]

        # Trying to seperate name from city
        team_name = row.split(' ')
        if (team_name[-1][-1] == '*' or team_name[-1][-1] == '+'):
            team_name[-1] = team_name[-1][0:-1]
        elif (team_name[-1] == 'Division' or team_name[-2] == 'AFC' or team_name[-2] == 'NFC'):
            df_sport.drop(index, inplace=True)
            index += 1
            continue
        df_sport.at[index, 'team'] = team_name
        for i in team_name:
            if (len(team_name) == 3 and team_name[1] in ['Angeles', 'York', 'Louis', 'Bay',
                                                         'Jersey', 'Jose', 'City', 'Francisco',
                                                         'Diego','State','Orleans','Antonio',
                                                         'England']):
                df_sport.at[index, 'team'] = team_name[-1]
            elif (len(team_name) == 3):
                df_sport.at[index, 'team'] = team_name[-2] + ' ' + team_name[-1]
            else:
                df_sport.at[index, 'team'] = team_name[-1]
        index += 1


    # Changing df_cities[sport]
    index = 0

    for row in df_cities[sport]:
        row = row.split('[')[0]
        if (len(row) < 2):
            df_cities.drop(index, inplace=True)
            index += 1
            continue
        if(row[0] == '7'):
            df_cities.at[index, sport] = [row]
            index += 1
            continue

        trueshit = False
        if(row[0] == '4'):
            trueshit = True

        team_names = re.findall('[A-Z][^A-Z]*', row)
        new_teams = []
        if(trueshit):
            new_teams.append('49ers')
        i = 0
        while i < len(team_names):
            if (team_names[i][-1] == ' '):
                new_teams.append(team_names[i] + team_names[i + 1])
                i += 1
            else:
                new_teams.append(team_names[i])
            i += 1
        df_cities.at[index, sport] = new_teams
        index += 1

    df_cities = df_cities[['Metropolitan area', 'Population (2016 est.)[8]', sport]].copy()

    df_cities.rename(columns={'Population (2016 est.)[8]': 'Population', sport: 'team'}, inplace=True)

    # This is kind of cheating, using already made code.. I'll have to understand it first..
    df_cities = explode(df_cities, ['team'], fill_value='', preserve_index=True)

    df_merge = df_cities.merge(df_sport, how='right', on='team')
    df_merge = df_merge[['Metropolitan area', 'Population', 'W', 'L']]
    df_merge[['Population','W','L']] = df_merge[['Population','W','L']].astype('float')
    df_merge = df_merge.groupby('Metropolitan area').mean()
    df_merge['Ratio'] = df_merge['W'] / (df_merge['W'] + df_merge['L'])
    corr_nhl = df_merge['Population'].corr(df_merge['Ratio'], method='pearson')


    if (give_plot):
        plt.plot(df_merge['Population'], df_merge['Ratio'], '.')
        b = np.array(df_merge['Ratio'])

        A = np.array(df_merge['Population']).T

        A = np.array([A, np.ones(len(A))]).T
        a = sl.solve((A.T @ A), A.T @ b)
        x = np.linspace(0, 25000000, 30)
        y = x * a[0] + np.ones(30) * a[1]

        plt.plot(x, y)
        plt.show()

    df_merge = df_merge.reset_index()

    return df_merge[['Metropolitan area', 'Ratio']]

    # return stats.pearsonr(population_by_region, win_loss_by_region)


# FOR NHL
nhl_df = pd.read_csv("assets/nhl.csv")
# FOR MLB
mlb_df=pd.read_csv("assets/mlb.csv")
# FOR NBA
nba_df = pd.read_csv("assets/nba.csv")
# FOR NFL
nfl_df=pd.read_csv("assets/nfl.csv")
# Cities
cities = pd.read_html("assets/wikipedia_data.html")[1]
cities = cities.iloc[:-1, [0, 3, 5, 6, 7, 8]]

def main():
    df_nhl = sport_correlation(df_sport=nhl_df.copy(), df_cities=cities.copy(), give_plot=False, sport="NHL")
    df_mlb = sport_correlation(df_sport=mlb_df.copy(), df_cities=cities.copy(), give_plot=False, sport="MLB")
    df_nba = sport_correlation(df_sport=nba_df.copy(), df_cities=cities.copy(), give_plot=False, sport="NBA")
    df_nfl = sport_correlation(df_sport=nfl_df.copy(), df_cities=cities.copy(), give_plot=False, sport="NFL")

    df_nhl_nba = df_nhl.merge(df_nba, on='Metropolitan area',suffixes = ('_nhl','_nba'), how='inner')

    df_mlb_nfl = df_mlb.merge(df_nfl, on='Metropolitan area', suffixes=('_mlb', '_nfl'), how='inner')


    ttest_nhl_nba = stats.ttest_rel(df_nhl_nba['Ratio_nhl'],df_nhl_nba['Ratio_nba'])
    ttest_mlb_nfl = stats.ttest_rel(df_mlb_nfl['Ratio_mlb'], df_mlb_nfl['Ratio_nfl'])
    print(ttest_nhl_nba)
    print(ttest_mlb_nfl)
    # print(df_nhl_nba)



if __name__ == "__main__":
    main()

# assert len(population_by_region) == len(win_loss_by_region), "Q1: Your lists must be the same length"
# assert len(population_by_region) == 28, "Q1: There should be 28 teams being analysed for NHL"



# Question 5
# I have access to tables with metropolitan area, W/L, Ration and Population.
# Since, I average for each Metropolitan area, I don't really need the exact teams.
# I simply run ttest on the Ratio of the two areas.
# I should do an inner join on the metropolitan area, and that should generate ratio_x and ration_y
#