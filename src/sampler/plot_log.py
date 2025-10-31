import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_number_of_solutions_per_stage(cfg):
    results_df = pd.read_csv(cfg.out_dir + "results.csv")

    # np.uniqueを使って，system_attrs_stageに含まれるステージの数字の種類を抜き出す
    stage_num_list = np.unique(results_df["system_attrs_stage"].values)

    plot_data = []
    for stage_num in stage_num_list:
        results_df_per_stage = results_df[results_df["system_attrs_stage"] == stage_num]
        plot_data.append(len(results_df_per_stage))

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.bar(
        x=np.arange(1, len(plot_data) + 1),
        height=plot_data,
        width=0.3,
        align="center",
    )

    ax.tick_params(labelsize=25)
    ax.set_xlim(0, len(plot_data) + 1)
    ax.set_xticks([i for i in range(1, len(plot_data) + 1)])
    ax.set_xlabel("Stage number", fontsize=35)

    ax.set_ylabel("Number of solutions", fontsize=35)
    ax.set_ylim(0, cfg.sampler.pop_size)
    ax.grid("y")
    plt.tight_layout()
    plt.savefig(cfg.out_dir + "number_of_solutions_per_stage.png")
    plt.close()
