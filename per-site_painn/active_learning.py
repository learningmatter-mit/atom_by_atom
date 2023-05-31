import pickle as pkl
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from typing import Optional

from persite_painn.nn import load_model
from persite_painn.utils import Normalizer, ensemble_inference, get_metal_idx


def check_max_cluster(cluster_bin, max_cluster_num, clusters):
    check = [False for i in range(len(clusters))]
    for i, cluster_name in enumerate(clusters):
        if cluster_bin[cluster_name] < max_cluster_num:
            cluster_bin[cluster_name] += 1
            check[i] = True
    if False in check:
        return (cluster_bin, False)
    else:
        return (cluster_bin, True)


class ActiveLearning:
    def __init__(self, models_path, dataset, original_dataset, num_model=5, uncertainty_fraction=0.1, diversity_sampling_method="PCA_Kmeans", diversity_cluster_num=9, normalizer=True):
        self.dataset = dataset
        self.original_dataset = original_dataset
        self.model_list, self.normalizer = self.make_model_list(models_path, num_model, normalizer)
        self.uncertainty, self.uncertainty_threshold, self.uncertainty_values = self.uncertainty_sampling(uncertainty_fraction)
        self.diversity, self.diversity_embedding_values, self.reduced_embedding, self.cluster_label = self.diversity_sampling(method=diversity_sampling_method, cluster_num=diversity_cluster_num)

    def make_model_list(self, models_path, num_model, normalizer=True):
        model_list = []
        for i in range(num_model):
            path = Path(models_path) / str(i) / "best_model.pth.tar"
            model, checkpoint = load_model(model_path=path)
            model_list.append(model)
        if normalizer:
            _normalizer = {"target": Normalizer(checkpoint['normalizer']["target"])}
        else:
            _normalizer = None

        return model_list, _normalizer

    def uncertainty_sampling(self, uncertainty_fraction):
        uncertainty = {}
        values_bin = []
        print("Uncertainty Sampling...")
        for data in tqdm(self.dataset):
            data_name = data["name"].item()
            metal_idx = get_metal_idx(data_name, self.original_dataset)
            variance = ensemble_inference(self.model_list, data, output_key="target", normalizer=self.normalizer, var=True)[[metal_idx], :].view(-1)
            uncertainty.update({data_name: variance.sum().item()})
            values_bin.append(variance.sum().item())

        sorted_uncertainty = np.argsort(values_bin)
        num_values = int(len(values_bin)*uncertainty_fraction)

        rslt = sorted_uncertainty[-num_values:]

        return uncertainty, rslt[0], values_bin

    def get_atom_embedding(self):
        features = {}

        def get_atom_features(name):
            def hook(model, input, output):
                features[name] = output.detach()
            return hook

        hook_emb_bin = []
        for i in range(len(self.model_list)):
            hook_emb_bin.append(self.model_list[i].readout_block.readoutdict["target"][-1].register_forward_hook(get_atom_features(f"atom_emb{i}")))

        embedding_bin = []
        data_name_bin = []
        for data in tqdm(self.dataset):
            metal_idx = get_metal_idx(data["name"].item(), self.original_dataset)

            embedding = []
            for i, model in enumerate(self.model_list):
                model(data)
                embedding.append(features[f"atom_emb{i}"][metal_idx].unsqueeze(1))

            for i in range(len(metal_idx)):
                stacking_bin = []
                for j in range(len(self.model_list)):
                    stacking_bin.append(embedding[j][i])
                val = torch.mean(torch.stack(stacking_bin, dim=1), dim=1).squeeze(0)
                embedding_bin.append(val.numpy())
                data_name_bin.append(data["name"].item())

        for hook_emb in hook_emb_bin:
            hook_emb.remove()
        embedding_bin = np.asarray(embedding_bin)

        return embedding_bin, data_name_bin

    def pca_kmeans(self, embedding_bin, data_name_bin, num_reduced_axis=16, cluster_num=9):
        pca = PCA(n_components=num_reduced_axis).fit(embedding_bin)
        df_pca = pca.transform(embedding_bin)
        kmeans = KMeans(n_clusters=cluster_num, init='random', random_state=0).fit(df_pca)
        label_kmeans = kmeans.fit_predict(df_pca)

        diversity = {}
        for key, cluster in zip(data_name_bin, label_kmeans):
            if key in list(diversity.keys()):
                new_cluster = diversity.get(key) + [cluster]
                diversity.update({key: new_cluster})
            else:
                diversity.update({key: [cluster]})

        return diversity, df_pca, label_kmeans

    def diversity_sampling(self, method="PCA_Kmeans", cluster_num=9):
        print(f"Diversity Sampling with {method}...")
        diversity_embedding_values, data_name_bin = self.get_atom_embedding()

        if method == "PCA_Kmeans":
            diversity, reduced_embedding, cluster_label = self.pca_kmeans(diversity_embedding_values, data_name_bin, cluster_num=cluster_num)
        else:
            NotImplementedError("Method not implemented")

        return diversity, diversity_embedding_values, reduced_embedding, cluster_label

    def get_next_data(self, num_data, save=True, existing_key_bin: Optional[str]=None):
        diversity_cluster_num = len(np.unique(self.cluster_label))
        max_cluster_num = (num_data // diversity_cluster_num) * 2  # There can be diatomic sites
        key_bin = []
        cluster_bin = [0 for i in range(diversity_cluster_num)]
        total_key_bin = list(self.uncertainty.keys())
        if existing_key_bin is not None:
            existing_key_bin_data = pkl.load(open(existing_key_bin, 'rb'))
            existing_keys = list(existing_key_bin_data.values())
        else:
            existing_keys = []
        success = 0
        while success <= num_data:
            key = random.choice(total_key_bin)
            cluster_bin, check_diversity = check_max_cluster(cluster_bin, max_cluster_num, self.diversity[key])
            if self.uncertainty[key] >= self.uncertainty_threshold or check_diversity:
                if existing_key_bin is not None and key not in existing_keys:
                    key_bin.append(key)
                    total_key_bin.remove(key)
                    success += 1
                else:
                    key_bin.append(key)
                    total_key_bin.remove(key)
                    success += 1                    
            check = [False for i in range(len(cluster_bin))]
            for i, val in enumerate(cluster_bin):
                if val >= max_cluster_num:
                    check[i] = True
            if False not in check:
                break
        if save:
            pkl.dump(self.uncertainty, open("uncertainty.pkl", 'wb'))
            pkl.dump(self.diversity, open("embedding.pkl", 'wb'))
            pkl.dump(key_bin, open("new_data_keys.pkl", 'wb'))

        return key_bin

    def plot_uncertainty(self, plot_name="uncertainty.pdf"):
        plt.hist(self.uncertainty_values, bins=10)
        plt.gca().set(title='Frequency Histogram of $\sigma^2$', ylabel='Frequency')
        plt.gca().set_xlabel("Variance")
        plt.savefig(plot_name)

    def plot_diversity(self, plot_name="diversity.pdf", color_dict={0: "#d53e4f", 1: "#f46d43", 2: "#fdae61", 3: "#fee08b", 4: "#8073ac", 5: "#abd9e9", 6: "#abdda4", 7: "#66c2a5", 8: "#3288bd"}):
        plt.figure(figsize=(5, 5))
        for _, val in enumerate(np.unique(self.cluster_label)):
            plt.scatter(self.reduced_embedding[self.cluster_label == val, 0], self.reduced_embedding[self.cluster_label == val, 1], s=3, alpha=0.3, color=color_dict[val])
        plt.xlabel("PCA1", size=18)
        plt.ylabel("PCA2", size=18)
        plt.tight_layout()
        plt.savefig(plot_name)


if __name__ == "__main__":
    model_path = "/home/hojechun/00-research/01-Spectra/per-site_PaiNN/results/01-sac/1-first"
    print("Loading dataset...")
    original_dataset = pkl.load(open("/home/hojechun/00-research/00-SAC/data_sac_total_1121.pkl", "rb"))
    dataset = torch.load("/home/hojechun/00-research/01-Spectra/per-site_PaiNN/data_cache/data_unlabelled_sac")

    activelearning = ActiveLearning(models_path=model_path, dataset=dataset, original_dataset=original_dataset, num_model=5, uncertainty_fraction=0.1)
    # activelearning.plot_uncertainty()
    # activelearning.plot_diversity()
    activelearning.get_next_data(100)
