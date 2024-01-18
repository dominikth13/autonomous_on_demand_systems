import csv
import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt

from logger import LOGGER

class TimeSeriesDiscretization:

    def discretize_day():
        
        LOGGER.debug("Loaded and prepare data")
        dfs = TimeSeriesDiscretization.prepare_data()

        fig, ax = plt.subplots(2, 4, figsize=(1280/96, 720/96), dpi=96)
        ax = ax.ravel()
        i = 0
        weekdays = ["Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
        weekdays_bkps = {day: [] for day in weekdays}
        for df in dfs:
            LOGGER.debug("Fit the discretization model for ")
            algo = rpt.Dynp(model="l2", min_size=2)
            algo.fit(df['number_of_orders'].values)
            n_bkps = 5
            LOGGER.debug(f"Predict model with {n_bkps} change points")
            result = algo.predict(n_bkps=n_bkps)
            ax[i].plot(df['half_hour_interval'], df['number_of_orders'])
            for bkp in result[:-1]:  # Letztes Element in 'result' ist die Länge der Datenreihe, nicht ein Breakpoint
                weekdays_bkps[weekdays[i]].append(bkp)
                ax[i].axvline(x=df['half_hour_interval'].iloc[bkp], color='k', linestyle='--')
            ax[i].set_title(f"Dynp model with {n_bkps} breakpoints")
            i += 1

        plt.tight_layout()
        plt.savefig('code/data_visualization/discretized_time_series.png', dpi=600)
        plt.show()

        csv_file_path = "code/data_output/time_series_break_points.csv"
        with open(csv_file_path, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(weekdays)
            matrix = [[] for w in weekdays_bkps]
            for i, w in enumerate(weekdays_bkps):
                matrix[i] = list(map(lambda x: x*30, weekdays_bkps[w]))
            writer.writerows(np.array(matrix).transpose())

    def prepare_data() -> list[pd.DataFrame]:
        dfs = []
        start = datetime.datetime(2015, 7, 9)
        for i in range(7):
            df = pd.read_csv(f"code/data/orders_{start.strftime('%Y-%m-%d')}.csv")
            df['pickup_time'] = pd.to_datetime(df['pickup_time'])
            sec = start + datetime.timedelta(7)
            df2 = pd.read_csv(f"code/data/orders_{sec.strftime('%Y-%m-%d')}.csv")
            df2['pickup_time'] = pd.to_datetime(df['pickup_time'])
            df = pd.concat([df, df2], ignore_index=True)
            df['minutes_since_midnight'] = df['pickup_time'].apply(lambda x: x.hour * 60 + x.minute)

            df['half_hour_interval'] = df['minutes_since_midnight'] // 30 #* 30

            df_count = df.groupby('half_hour_interval').size().reset_index(name='number_of_orders')
            dfs.append(df_count)
            start += datetime.timedelta(1)
        return dfs