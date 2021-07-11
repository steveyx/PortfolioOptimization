import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
import datetime as dt
plt.rcParams['font.family'] = 'monospace'


def portfolio_optimization_animation(df_results, max_sharpe_port, min_vol_port, stocks, update_points=10):
    def update_animation(i, df, scatter, max_sharpe, min_vol):
        to_idx = (i + 1) * update_points
        _data = df.loc[:to_idx, ['stdev', 'ret']].values
        _max_sharpe_port = df_results.iloc[df_results.loc[:to_idx, 'sharpe'].idxmax()]
        _min_vol_port = df_results.iloc[df_results.loc[:to_idx, 'stdev'].idxmin()]
        scatter.set_offsets(_data)
        scatter.set_array(df.loc[:to_idx].sharpe)
        max_sharpe.set_offsets([[_max_sharpe_port['stdev'], _max_sharpe_port['ret']]])
        min_vol.set_offsets([[_min_vol_port['stdev'], _min_vol_port['ret']]])

        weights_ret = ["{}:   {:.3f}".format(stk, _max_sharpe_port[stk]) for j, stk in enumerate(stocks)]
        weights_vol = ["{}:   {:.3f}".format(stk, _min_vol_port[stk]) for j, stk in enumerate(stocks)]
        text_max_ret.set_text("\n".join(["Max Sharpe Ratio\n"] + weights_ret))
        text_min_vol.set_text("\n".join(["Min Volatility\n"] + weights_vol))
        return scatter, max_sharpe, min_vol

    start_points = 3
    fig = plt.figure(figsize=(9.6, 5.4))
    ax = fig.add_axes([0.1, 0.1, 0.65, 0.8])
    ax1 = fig.add_axes([0.77, 0.1, 0.2, 0.8])

    total_len = len(df_results)
    # create scatter plot coloured by Sharpe Ratio
    data_points = ax.scatter(df_results.loc[:start_points].stdev,
                             df_results.loc[:start_points].ret,
                             s=20,
                             c=df_results.loc[:start_points].sharpe, cmap='RdYlBu')
    ax.set_title('Portfolio Optimization', fontsize=12)
    ax.set_xlabel('Volatility', fontsize=10)
    ax.set_ylabel('Return', fontsize=10)
    ax.set_xlim(df_results['stdev'].min()-0.03, df_results['stdev'].max()+0.02)
    ax.set_ylim(df_results['ret'].min()-0.04, df_results['ret'].max()+0.02)
    ax1.patch.set_facecolor('ivory')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
    _max_ret_port = "Max Sharpe Ratio\n"
    text_max_ret = ax1.text(0.04, 0.7, _max_ret_port, fontsize=9, color='black',
                            horizontalalignment='left', transform=ax1.transAxes)
    text_min_vol = ax1.text(0.04, 0.5, "Min Volatility\n", fontsize=9, color='black',
                            horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes)

    # plot red star to highlight position of portfolio with highest Sharpe Ratio
    max_sharpe = ax.scatter(max_sharpe_port[1], max_sharpe_port[0], marker=(5, 1, 0), color='r', s=100)
    # plot green star to highlight position of minimum variance portfolio
    min_vol = ax.scatter(min_vol_port[1], min_vol_port[0], marker=(5, 1, 0), color='g', s=100)
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    ani = animation.FuncAnimation(fig, update_animation,
                                  int(total_len / update_points) + 1,
                                  fargs=(df_results, data_points, max_sharpe, min_vol),
                                  interval=100, repeat=False)
    ffwriter = mpl.animation.FFMpegWriter()
    t = dt.datetime.now()
    ani.save('animations/portfolio_opt_{}.mp4'.format(t.strftime("%Y%m%d_%H%M")), writer=ffwriter)
    plt.show()
    plt.savefig("figure_po.png")


def portfolio_optimization_benchmark_animation(df_results, best_results, gd_results,
                                               max_sharpe_port, min_vol_port, stocks,
                                               update_points=10, file_format="gif"):
    def update_animation(i, df, scatter, max_sharpe, min_vol):
        to_idx = (i + 1) * update_points
        _data = df.loc[:to_idx, ['stdev', 'ret']].values
        _max_sharpe_port = df_results.iloc[df_results.loc[:to_idx, 'sharpe'].idxmax()]
        _min_vol_port = df_results.iloc[df_results.loc[:to_idx, 'stdev'].idxmin()]
        scatter.set_offsets(_data)
        scatter.set_array(df.loc[:to_idx].sharpe)
        max_sharpe.set_offsets([[_max_sharpe_port['stdev'], _max_sharpe_port['ret']]])
        min_vol.set_offsets([[_min_vol_port['stdev'], _min_vol_port['ret']]])

        weights_sim = ["{0}{1:.3f}".format(stk.ljust(18), _max_sharpe_port[stk]) for j, stk in enumerate(stocks)]
        weights_gd_ = gd_results[:to_idx, 3:][-1]
        weights_gd = ["{0}{1:.3f}".format(stk.ljust(18), weights_gd_[j]) for j, stk in enumerate(stocks)]
        _max_sharpe_sim = "{0}{1:.3f}".format("Max Sharpe Ratio".ljust(18), _max_sharpe_port["sharpe"])
        _max_sharpe_gd = "{0}{1:.3f}".format("Max Sharpe Ratio".ljust(18), gd_results[:to_idx, 2][-1])
        text_max_sim.set_text("\n".join([_max_sharpe_sim] + weights_sim))
        text_min_gd.set_text("\n".join([_max_sharpe_gd] + weights_gd))
        gd_line.set_data(gd_results[:to_idx, 1], gd_results[:to_idx, 0])
        gd_line_ax3.set_data(gd_results[:to_idx, 1], gd_results[:to_idx, 0])
        _indices = [j for j in best_results if j <= to_idx]
        best_line.set_data(df.loc[_indices].stdev, df.loc[_indices].ret)
        best_line_ax2.set_data(df.loc[_indices].stdev, df.loc[_indices].ret)
        return scatter, max_sharpe, min_vol

    start_points = 3
    fig = plt.figure(figsize=(9.6, 5.4))
    ax = fig.add_axes([0.08, 0.08, 0.45, 0.82])
    # ax1 = fig.add_axes([0.77, 0.1, 0.2, 0.8])
    ax2 = fig.add_axes([0.58, 0.54, 0.4, 0.36])
    ax3 = fig.add_axes([0.58, 0.08, 0.4, 0.36])
    ax2.sharex(ax3)

    total_len = len(df_results)
    # create scatter plot coloured by Sharpe Ratio
    data_points = ax.scatter(df_results.loc[:start_points].stdev,
                             df_results.loc[:start_points].ret,
                             s=20,
                             c=df_results.loc[:start_points].sharpe, cmap='RdYlBu')
    gd_line, = ax.plot([0], [0], color="blue", marker=".", label="Gradient Descent Solution")
    best_line, = ax.plot([0], [0], color="black", label="Monte Carlo Simulation")
    best_line_ax2, = ax2.plot([0], [0], color="black", marker=".", label="Monte Carlo Simulation")
    gd_line_ax3, = ax3.plot([0], [0], color="blue", marker=".", label="Gradient Descent Solution")
    ax.set_title('Portfolio Optimization', fontsize=9)
    ax.set_xlabel('Volatility', fontsize=8)
    ax.set_ylabel('Return', fontsize=8)
    ax.set_xlim(df_results['stdev'].min()-0.01, df_results['stdev'].max()-0.005)
    ax.set_ylim(df_results['ret'].min()-0.01, df_results['ret'].max()-0.005)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    ax2.set_xlim(0.22, 0.3)
    ax2.set_ylim(0.3, 0.42)
    ax3.set_ylim(0.3, 0.42)
    ax2.set_title('Maximize Sharpe Ratio - Monte Carlo Simulation', fontsize=9)
    ax3.set_title('Maximize Sharpe Ratio - Gradient Descent Solution', fontsize=9)
    ax2.tick_params(axis='x', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    ax3.tick_params(axis='x', labelsize=8)
    ax3.tick_params(axis='y', labelsize=8)
    plt.setp(ax2.get_xticklabels(), visible=False)
    _max_ret_port = "Max Sharpe Ratio"
    text_max_sim = ax2.text(0.27, 0.35, _max_ret_port, fontsize=7, color='blue',
                            horizontalalignment='left', verticalalignment='top',
                            # transform=plt.gcf().transFigure
                            )
    text_min_gd = ax3.text(0.27, 0.35, "Max Sharpe Ratio", fontsize=7, color='blue',
                           horizontalalignment='left', verticalalignment='top',
                           # transform=plt.gcf().transFigure
                           )

    # plot red star to highlight position of portfolio with highest Sharpe Ratio
    max_sharpe = ax.scatter(max_sharpe_port[1], max_sharpe_port[0], marker=(5, 1, 0), color='r', s=100,
                            label="Max Sharpe Ratio")
    # plot green star to highlight position of minimum variance portfolio
    min_vol = ax.scatter(min_vol_port[1], min_vol_port[0], marker=(5, 1, 0), color='g', s=100,
                         label="Min Volatility")
    ax.legend(loc="upper left", fontsize=7)
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    ani = animation.FuncAnimation(fig, update_animation,
                                  int(total_len / update_points) + 1,
                                  fargs=(df_results, data_points, max_sharpe, min_vol),
                                  interval=100, repeat=False)
    t = dt.datetime.now()
    if file_format == "gif":
        _save_file = "animations/portfolio_opt_{}.gif".format(t.strftime("%Y%m%d_%H%M"))
        ani.save(_save_file, writer='imagemagick', bitrate=10)
    else:
        _save_file = "animations/portfolio_opt_{}.mp4".format(t.strftime("%Y%m%d_%H%M"))
        ffwriter = animation.FFMpegWriter()
        ani.save(_save_file, writer=ffwriter)

