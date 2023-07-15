import numpy as np
import pandas as pd
import os


def cal_sigma_mu(dataset_name):
	"""
	���� users �� items �ڻ��ľ�ֵ�ͱ�׼��.
	"""
	path = os.path.join("./data", dataset_name)
	emb_user = np.load(os.path.join(path, "emb_user.npy"))
	emb_item = np.load(os.path.join(path, "emb_item.npy"))
	
	# on the global
	ii = emb_item.dot(emb_item.T)
	ii_mean_g, ii_std_g = ii.mean(), ii.std()

	ui = emb_user.dot(emb_item.T)
	ui_mean_g, ui_std_g = ui.mean(), ui.std()
	
	# on the ratings
	mu_ii, mu_ui = [], []
	df = pd.read_csv(os.path.join(path, "rating.csv"))[['userInd', 'itemInd']]
	g = df.groupby('userInd')

	for k, v in g:
		ti = emb_item[v.itemInd.values]
		mu_ii.append(ti.dot(ti.T).mean())
		mu_ui.append(emb_user[k].dot(ti.T).mean())
	
	mu_ii = np.array(mu_ii)
	mu_ui = np.array(mu_ui)
	ii_mean_r, ii_std_r = mu_ii.mean(), mu_ii.std()
	ui_mean_r, ui_std_r = mu_ui.mean(), mu_ui.std()

	return [ui_mean_r, ui_std_r, ii_mean_r, ii_std_r,
			ui_mean_g, ui_std_g, ii_mean_g, ii_std_g]


if __name__ == "__main__":
	list_dataset_name = ["mlls", "tool", "beauty", "kindle", "sport", "ml10m", "home", "elec", "clothing"]
	for dn in list_dataset_name:
		ret = cal_sigma_mu(dn)
		print(ret)
		break