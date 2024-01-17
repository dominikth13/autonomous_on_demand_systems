from matplotlib import pyplot as plt
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
        for df in dfs:
            LOGGER.debug("Fit the discretization model")
            algo = rpt.Dynp(model="l2", min_size=2)
            algo.fit(df['number_of_orders'].values)

            n_bkps = 5
            LOGGER.debug(f"Predict model with {n_bkps} change points")
            result = algo.predict(n_bkps=n_bkps)
            ax[i].plot(df['half_hour_interval'], df['number_of_orders'])
            for bkp in result[:-1]:  # Letztes Element in 'result' ist die Länge der Datenreihe, nicht ein Breakpoint
                ax[i].axvline(x=df['half_hour_interval'].iloc[bkp], color='k', linestyle='--')
            ax[i].set_title(f"Dynp model with {n_bkps} breakpoints")
            i += 1

        plt.tight_layout()
        plt.savefig('code/data_visualization/discretized_time_series.png', dpi=600)
        plt.show()
    
    def prepare_data() -> list[pd.DataFrame]:
        dfs = []
        for i in range(1,8):
            df = pd.read_csv(f"code/data/orders_2015-07-0{i}.csv")
            df['pickup_time'] = pd.to_datetime(df['pickup_time'])
            df['minutes_since_midnight'] = df['pickup_time'].apply(lambda x: x.hour * 60 + x.minute)

            df['half_hour_interval'] = df['minutes_since_midnight'] // 30 #* 30

            df_count = df.groupby('half_hour_interval').size().reset_index(name='number_of_orders')
            dfs.append(df_count)
        return dfs