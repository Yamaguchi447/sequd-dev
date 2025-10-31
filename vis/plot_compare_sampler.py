import glob
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import sort_data

color_list = [n[4:] for n in list(mcolors.TABLEAU_COLORS)]

if __name__ == "__main__":
    input_dir_path = "outputs_comp_sampler_example_20250808"
    output_dir_path = input_dir_path + "fig/"
    os.makedirs(output_dir_path, exist_ok=True)

    sampler_names = [
        "SeqUD",
        "DVESeqUD",
        "DISeqUD",
        "DIVESeqUD"
        "CMAES",
    ]

    for dim in [10]:
        for func_id in [1,2,7,10]:
            for pop_size in [50]:
                fig, ax = plt.subplots(figsize=(16, 9))
                for sampler_index, sampler_name in enumerate(sampler_names):
                    dir_paths = sorted(
                        glob.glob(
                            input_dir_path
                            + f"{sampler_name}/"
                            + f"func_id_{func_id}/"
                            + f"dim_{dim}/"
                            + f"pop_size_{pop_size}/*"
                        )
                    )

                    plot_data = pd.DataFrame()
                    for results_index, dir_path in enumerate(dir_paths):
                        results = pd.read_csv(dir_path + "/results.csv")
                        plot_data[results_index] = np.array(sort_data(results["value"]))

                    MEAN = plot_data.mean(axis=1)
                    SEM = plot_data.sem(axis=1)
                    ax.plot(
                        np.arange(1, len(plot_data) + 1),
                        MEAN,
                        linewidth=4,
                        alpha=0.8,
                        color=color_list[sampler_index],
                        label=f"sampler_name: {sampler_name}",
                    )
                    ax.fill_between(
                        np.arange(1, len(plot_data) + 1),
                        MEAN + SEM,
                        MEAN - SEM,
                        color=color_list[sampler_index],
                        alpha=0.4,
                        linewidth=0,
                    )

                ax.tick_params(labelsize=25)
                ax.set_xlim(1, 400)
                ax.set_xlabel("Function evaluations", fontsize=25)
                # if func_id == 1:
                #     ax.set_ylim(3, 60)
                # elif func_id == 2:
                #     ax.set_ylim(1e4, 1e6)
                # elif func_id == 7:
                #     ax.set_ylim(30, 500)

                ax.set_ylabel("Objective", fontsize=25)
                ax.set_yscale("log")
                ax.grid()
                ax.legend(
                    bbox_to_anchor=(1, 1.05),
                    loc="lower right",
                    borderaxespad=0.2,
                    fontsize=15,
                    ncols=2,
                )
                plt.tight_layout()
                plt.savefig(
                    output_dir_path
                    + f"func_id_{func_id}_dim_{dim}_pop_size_{pop_size}.png"
                )
                plt.close()
