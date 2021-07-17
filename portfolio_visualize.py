import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams['font.family'] = 'monospace'


class PortfolioVisualize:

    @staticmethod
    def visualize(df_results, best_indices_sim, gd_results, max_sharpe_port, stocks):
        fig = plt.figure(figsize=(5, 6), dpi=300)
        ax2 = fig.add_axes([0.12, 0.55, 0.8, 0.4])
        ax3 = fig.add_axes([0.12, 0.07, 0.8, 0.4])
        ax2.sharex(ax3)
        _indices = best_indices_sim
        _x_sim = df_results.loc[_indices, 'stdev'].values.flatten()
        _y_sim = df_results.loc[_indices, 'ret'].values.flatten()
        ax2.scatter(_x_sim, _y_sim, color="blue", marker=".", label="Monte Carlo Simulation")
        ax2.scatter(_x_sim[-1:], _y_sim[-1:], color="r", marker=(5, 1, 0), label="Max")
        ax2.quiver(_x_sim[:-1], _y_sim[:-1], _x_sim[1:] - _x_sim[:-1], _y_sim[1:] - _y_sim[:-1],
                   scale_units='xy', angles='xy', width=0.005, scale=1, color="blue", joinstyle="round")
        _x_gd, _y_gd = gd_results[:, 1], gd_results[:, 0]
        ax3.scatter(_x_gd, _y_gd, color="purple", marker=".", label="Gradient Descent Solution")
        ax3.scatter(_x_gd[-1:], _y_gd[-1:], color="r", marker=(5, 1, 0), label="Max")
        ax3.quiver(_x_gd[:-1], _y_gd[:-1], _x_gd[1:] - _x_gd[:-1], _y_gd[1:] - _y_gd[:-1],
                   scale_units='xy', angles='xy', width=0.005, scale=1, color="purple", joinstyle="round")
        ax2.set_xlim(0.23, 0.28)
        ax2.set_ylim(0.3, 0.4)
        ax3.set_ylim(0.3, 0.4)
        ax2.set_ylabel('Return', fontsize=10)
        ax2.set_title('Maximize Sharpe Ratio: Monte Carlo Simulation', fontsize=11)
        ax3.set_title('Maximize Sharpe Ratio: Gradient Descent Solution', fontsize=11)
        ax3.set_ylabel('Return', fontsize=10)
        ax3.set_xlabel('Volatility', fontsize=10)
        ax2.tick_params(axis='x', labelsize=8)
        ax2.tick_params(axis='y', labelsize=8)
        ax3.tick_params(axis='x', labelsize=8)
        ax3.tick_params(axis='y', labelsize=8)
        plt.setp(ax2.get_xticklabels(), visible=False)
        weights_sim = ["{0}{1:.3f}".format(stk.ljust(18), max_sharpe_port[stk]) for j, stk in enumerate(stocks)]
        weights_gd_ = gd_results[-1, 3:]
        weights_gd = ["{0}{1:.3f}".format(stk.ljust(18), weights_gd_[j]) for j, stk in enumerate(stocks)]
        _max_sharpe_sim = "{0}{1:.3f}".format("Max Sharpe Ratio".ljust(18), max_sharpe_port["sharpe"])
        _max_sharpe_gd = "{0}{1:.3f}".format("Max Sharpe Ratio".ljust(18), gd_results[-1, 2])
        _sharpe_sim = "\n".join([_max_sharpe_sim] + weights_sim)
        _text_sharpe_sim = ax2.text(0.26, 0.34, _sharpe_sim, fontsize=8, color='blue',
                                    horizontalalignment='left', verticalalignment='top')
        _sharpe_gd = "\n".join([_max_sharpe_gd] + weights_gd)
        _text_sharpe_gd = ax3.text(0.26, 0.34, _sharpe_gd, fontsize=8, color='blue',
                                   horizontalalignment='left', verticalalignment='top')
        plt.savefig("figure_benchmark.png")
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        plt.show()

    @staticmethod
    def visualize_simulation(df_results, max_sharpe_port, min_vol_port):
        fig = plt.figure(figsize=(5, 5), dpi=300)
        ax = fig.add_axes([0.12, 0.1, 0.8, 0.8])
        # create scatter plot coloured by Sharpe Ratio
        data_points = ax.scatter(df_results.stdev,
                                 df_results.ret,
                                 s=20,
                                 label="Portfolios",
                                 c=df_results.sharpe, cmap='RdYlBu')

        ax.set_title('Portfolio Optimization by Monte Carlo Simulation', fontsize=10)
        ax.set_xlabel('Volatility', fontsize=10)
        ax.set_ylabel('Return', fontsize=10)
        ax.set_xlim(df_results['stdev'].min()-0.02, df_results['stdev'].max()+0.02)
        ax.set_ylim(df_results['ret'].min()-0.01, df_results['ret'].max()+0.01)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_aspect(aspect="equal")

        # plot red star to highlight position of portfolio with highest Sharpe Ratio
        max_sharpe = ax.scatter(max_sharpe_port[1], max_sharpe_port[0], marker=(5, 1, 0), color='r', s=100,
                                label="Max Sharpe Ratio")
        # plot green star to highlight position of minimum variance portfolio
        min_vol = ax.scatter(min_vol_port[1], min_vol_port[0], marker=(5, 1, 0), color='g', s=100,
                             label="Min Volatility")
        ax.legend(loc="upper left", fontsize=8)
        plt.savefig("simulation.png")
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        plt.show()

    @staticmethod
    def plot_benchmark_table(data=None):
        if data is None:
            data = [
                {
                    "Assets": 10,
                    "Portfolios Simulation": 100000,
                    "Learning Rate Gradient Descent": 0.05,
                    "SR Simulation": 1.55,
                    "SR Gradient Descent": 1.56,
                    "Time Simulation": 39.9,
                    "Time Gradient Descent": 0.03
                },
                {
                    "Assets": 20,
                    "Portfolios Simulation": 200000,
                    "Learning Rate Gradient Descent": 0.05,
                    "SR Simulation": 1.70,
                    "SR Gradient Descent": 1.79,
                    "Time Simulation": 67.1,
                    "Time Gradient Descent": 0.03
                },
                {
                    "Assets": 50,
                    "Portfolios Simulation": 500000,
                    "Learning Rate Gradient Descent": 0.05,
                    "SR Simulation": 1.54,
                    "SR Gradient Descent": 1.87,
                    "Time Simulation": 204.1,
                    "Time Gradient Descent": 0.12
                },
                {
                    "Assets": 100,
                    "Portfolios Simulation": 1000000,
                    "Learning Rate Gradient Descent": 0.05,
                    "SR Simulation": 1.42,
                    "SR Gradient Descent": 1.93,
                    "Time Simulation": 394.7,
                    "Time Gradient Descent": 0.26,
                }
            ]
        df = pd.DataFrame(data)
        # df.rename(columns={
        #     "Portfolios Simulation": "Portfolios\nSimulation",
        #     "Learning Rate Gradient Descent": "Learning Rate\nGradient Descent",
        #     "SR Gradient Descent": "SR\nGradient Descent",
        #     "Time Gradient Descent": "Time\nGradient Descent"
        # }, inplace=True)
        col_labels = ["Assets", "Portfolios\nSimulation", "Learning Rate\nGradient Descent",
                      "SR\nSimulation",
                      "SR\nGradient Descent",
                      "Time\nSimulation",
                      "Time\nGradient Descent"]
        df["Assets"] = df["Assets"].astype(int)
        df["Portfolios Simulation"] = df["Portfolios Simulation"].astype(int)
        fig = plt.figure(figsize=(8, 1.4), dpi=200)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        tables = []
        ax.axis('off')
        _values = df.values.tolist()
        for i in range(len(df)):
            _values[i][0], _values[i][1] = int(_values[i][0]), int(_values[i][1])
        tab = ax.table(cellText=_values,
                       cellLoc='center', rowLoc='center',
                       colWidths=[0.07, 0.1, 0.15, 0.1, 0.15, 0.1, 0.15],
                       colLabels=col_labels,
                       # colColours=[None, None, None, "lightgreen", "lightgreen", "lightblue", "lightblue"],
                       # rowLabels=df.index[_s:_e].tolist(),
                       loc="left",
                       bbox=[0.05, 0.02, .9, 0.95])
        tables.append(tab)
        for tab in tables:
            # scalex, scaley = 1, 1
            # tab.scale(scalex, scaley)
            tab.auto_set_font_size(False)
            tab.set_fontsize(7.5)
            for key, cell in tab.get_celld().items():
                cell.set_linewidth(0)
        for row in range(len(df)+1):
            tables[0][(row, 0)].set_facecolor("lightgray")
            tables[0][(row, 1)].set_facecolor("lightgray")
            tables[0][(row, 2)].set_facecolor("lightgray")
            tables[0][(row, 3)].set_facecolor("lightgreen")
            tables[0][(row, 4)].set_facecolor("lightgreen")
            tables[0][(row, 5)].set_facecolor("lightblue")
            tables[0][(row, 6)].set_facecolor("lightblue")
        plt.subplots_adjust(wspace=0.4)
        plt.savefig("data/benchmark_performance.png", dpi=300)


if __name__ == "__main__":
    PortfolioVisualize.plot_benchmark_table()