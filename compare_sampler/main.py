import os
import subprocess
from multiprocessing import Pool

import numpy as np


def run_parallel(para_list):
    (
        out_dir,
        func_id,
        dim,
        pop_size,
        sampler,
        sampler_seed,
        max_budget,
    ) = para_list

    command = [
        "python",
        "src/main.py",
        "bbob",
        f"out_dir={out_dir}",
        f"default.r_seed={sampler_seed}",
        f"objective.func_id={func_id}",
        f"objective.dim={dim}",
        f"sampler.pop_size={pop_size}",
        f"sampler.n_trials={max_budget}",
        f"sampler.name={sampler}",
    ]
    subprocess.run(command)


if __name__ == "__main__":
    np.random.seed(1234)
    output_dir_name = "outputs_comp_sampler_example_20251029" # 実験結果が保存されるディレクトリ
    func_id_list = [1,2,7,10] # BBOB関数id番号
    pop_size_list = [50]
    dims = [10]
    RUN_INSTANCES = 15 # シード数

    sampler_list = [
        "SeqUD",
        "DVESeqUD",
        "DISeqUD",
        "DIVESeqUD",
        # "CMAES",
    ]
    sampler_seed_list = [np.random.randint(1, 1e6) for _ in range(RUN_INSTANCES)]
    max_budget = 400 # 1シードあたりのバジェット数

    para_list = []

    for sampler in sampler_list:
        for sampler_seed in sampler_seed_list:
            for func_id in func_id_list:
                for dim in dims:
                    for pop_size in pop_size_list:
                        out_dir = (
                            f"{output_dir_name}/"
                            + f"{sampler}/"
                            + f"func_id_{func_id}/"
                            + f"dim_{dim}/"
                            + f"pop_size_{pop_size}/"
                            + f"r_seed_{sampler_seed}/"
                        )
                        para_list.append(
                            [
                                out_dir,
                                func_id,
                                dim,
                                pop_size,
                                sampler,
                                sampler_seed,
                                max_budget,
                            ]
                        )

    P = Pool(os.cpu_count())
    P.map(run_parallel, para_list)
    P.close()
