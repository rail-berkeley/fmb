import glob
import tqdm
import numpy as np

paths = glob.glob("/nfs/kun2/datasets/fmb/grasp/trajectories_*.npy")

n_episodes = 0
for path in tqdm.tqdm(paths):
    n_episodes += len(np.load(path, allow_pickle=True))
print(f"Num episodes: {n_episodes}")
