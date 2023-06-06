import torch

from torch.utils.data.sampler import Sampler


class ImbalancedDatasetSampler(Sampler):
    """
    Source: https://github.com/ufoym/imbalanced-dataset-sampler/
            blob/master/torchsampler/imbalanced.py
    Sampling class to make sure positive and negative labels
    are represented equally during training.
    Attributes:
        data_length (int): length of dataset
        weights (torch.Tensor): weights of each index in the
            dataset depending.
    """

    def __init__(self, target_name, props):
        """
        Args:
            target_name (str): name of the property being classified
            props (dict): property dictionary
        """

        data_length = len(props[target_name])

        negative_idx = [
            i
            for i, target in enumerate(props[target_name])
            if round(target.item()) == 0
        ]
        positive_idx = [i for i in range(data_length) if i not in negative_idx]

        num_neg = len(negative_idx)
        num_pos = len(positive_idx)

        if num_neg == 0:
            num_neg = 1
        if num_pos == 0:
            num_pos = 1

        negative_weight = num_neg
        positive_weight = num_pos

        self.data_length = data_length
        self.weights = torch.zeros(data_length)
        self.weights[negative_idx] = 1 / negative_weight
        self.weights[positive_idx] = 1 / positive_weight

    def __iter__(self):

        return (
            i
            for i in torch.multinomial(self.weights, self.data_length, replacement=True)
        )

    def __len__(self):
        return self.data_length
