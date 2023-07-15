import pandas as pd
import numpy as np
import os
import utils


if __name__ == "__main__":
    list_dataset_name = ["mlls", "tool", "beauty", "kindle", "sport", "ml10m", "home", "elec", "clothing"]  # [7:8]
    list_method = ["rand", "nov", "pop", "qua", "ela", "acc", "dif", "div"]
    Ks = [5, 10, 15, 20][:-1]
    list_seed = [777, 7777, 77777, 73, 79, 83, 89, 97, 101, 103][:5]

    for dn in list_dataset_name:
        for seed in list_seed:
            for K in Ks:
                print(f"Generating {dn} - {seed} - {K} ...")
                save_dir = os.path.join("./data", dn, f"rec{K}", str(seed))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                read_dir = os.path.join("./data", dn, f"rec20", str(seed))
                for method in list_method:
                    rec_mat = np.load(os.path.join(read_dir, f"rec_{method}.npy"))
                    new_rec_mat = rec_mat[:, :K]
                    np.save(os.path.join(save_dir, f"rec_{method}.npy"), new_rec_mat)
                print(f"Finished {dn} - {seed} - {K} ...")
                
    for K in Ks:
        utils.evaluate(list_dataset_name, list_seed, list_method, K)