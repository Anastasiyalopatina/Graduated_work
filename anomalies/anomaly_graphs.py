import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

df_origin = pd.read_excel(open('result.xlsx', 'rb'), sheet_name='Initial_data')
df_anomaly = pd.read_excel(open('result.xlsx', 'rb'), sheet_name='Generate_anomaly')
df_kNN = pd.read_excel(open('result.xlsx', 'rb'), sheet_name='kNN')

col_names = ['Unit price', 'Quantity',
            'Tax 5%', 'Total','COGS', 'Gross income', 'Rating']

x_values = range(1000)

def draw_by_dots(x_idx, additional_df, y_label_name='0', x_label_name='Generate', anomaly_flag=0, where=plt, title_flag=1, fontsize_=8):
    for i in range(len(additional_df[x_idx])):
        marker_color = 'blue'
        marker_form = 'o'
        if anomaly_flag and additional_df[x_idx][i] == -1: 
            marker_color = 'red'
            marker_form = 'o'
        where.scatter(x_values[i], df_origin[x_idx][i], s=1, color=marker_color, marker=marker_form)
        if not isinstance(where, matplotlib.axes.Axes):
            where.ylabel(col_names[x_idx], fontsize=fontsize_)
        else:
            where.set_ylabel(col_names[x_idx], fontsize=fontsize_)
    where.tick_params(labelsize=fontsize_)
    if not isinstance(where, matplotlib.axes.Axes):
        where.xlabel(x_label_name, fontsize=fontsize_)
        where.title(y_label_name, fontsize=fontsize_)
    else:
        where.set_xlabel(x_label_name, fontsize=fontsize_)
        if title_flag:
            where.set_title(col_names[x_idx], fontsize=fontsize_)

def draw_origin_anomalized_detected(safe_flag=0, prob_flag=0):
    fig, axs = plt.subplots(5, 2)
    draw_by_dots(0, df_kNN, x_label_name='kNN', anomaly_flag=1, where=axs[0, 0], title_flag=0, fontsize_=6)
    draw_by_dots(0, df_anomaly, x_label_name='Generate', anomaly_flag=1, where=axs[0, 1], title_flag=0, fontsize_=6)
    draw_by_dots(1, df_kNN, x_label_name='kNN', anomaly_flag=1, where=axs[1, 0], title_flag=0, fontsize_=6)
    draw_by_dots(1, df_anomaly, x_label_name='Generate', anomaly_flag=1, where=axs[1, 1], title_flag=0, fontsize_=6)
    draw_by_dots(2, df_kNN, x_label_name='kNN', anomaly_flag=1, where=axs[2, 0], title_flag=0, fontsize_=6)
    draw_by_dots(2, df_anomaly, x_label_name='Generate', anomaly_flag=1, where=axs[2, 1], title_flag=0, fontsize_=6)
    draw_by_dots(3, df_kNN, x_label_name='kNN', anomaly_flag=1, where=axs[3, 0], title_flag=0, fontsize_=6)
    draw_by_dots(3, df_anomaly, y_label_name='Generate', anomaly_flag=1, where=axs[3, 1], title_flag=0, fontsize_=6)
    draw_by_dots(4, df_kNN, x_label_name='kNN', anomaly_flag=1, where=axs[4, 0], title_flag=0, fontsize_=6)
    draw_by_dots(4, df_anomaly, x_label_name='Generate', anomaly_flag=1, where=axs[4, 1], title_flag=0, fontsize_=6)
    fig.tight_layout(h_pad=0.2)

    plt.show()

draw_origin_anomalized_detected(safe_flag=1)
