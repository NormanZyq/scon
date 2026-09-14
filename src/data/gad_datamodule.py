import os
# from pygod.utils import load_data
import shutil
from collections.abc import Sequence
from typing import Optional, Any, List

import requests
import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.loader import DataLoader


def _identify_dataset_source(name: str) -> str:
    """Identify which data source to use based on dataset name.

    Args:
        name: Dataset name (case-sensitive)

    Returns:
        'pygod', 'planetoid', or 'tudataset'
    """
    # PyGOD datasets (case-sensitive)
    pygod_names = {"weibo", "reddit", "disney", "books", "enron"}
    if name in pygod_names or name.startswith("inj_") or name.startswith("gen_"):
        return "pygod"

    # Planetoid datasets (case-sensitive)
    planetoid_names = {"Cora", "CiteSeer", "PubMed"}
    if name in planetoid_names:
        return "planetoid"

    # Default to TUDataset for everything else
    return "tudataset"


def load_pygod_data(name, cache_dir=None):
    """Load PyGOD dataset from GitHub repository."""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.pygod/data')
    file_path = os.path.join(cache_dir, name + '.pt')
    zip_path = os.path.join(cache_dir, name + '.pt.zip')

    if os.path.exists(file_path):
        data = torch.load(file_path, weights_only=False)
    else:
        url = "https://github.com/pygod-team/data/raw/main/" + name + ".pt.zip"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s" % url)
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        shutil.unpack_archive(zip_path, cache_dir)
        data = torch.load(file_path, weights_only=False)
    return data


def collapse_anomaly_types(dataset, level='node', indicator='y'):
    if level == 'node':
        for data in dataset:
            data[indicator] = (data[indicator] >= 1).int()


def ignore_anomaly_types(dataset, level='node', indicator='y', type_ids=2):
    # set the indicator to zero if we ignore a type id
    if type(type_ids) == int:
        type_ids = [type_ids]
    if level == 'node':
        for data in dataset:
            for idx in type_ids:
                data[indicator][data[indicator] == idx] = 0


def _load_and_normalize_dataset(name: str, cache_dir: Optional[str] = None) -> List[Data]:
    """Load dataset from appropriate source and normalize node features.

    Args:
        name: Dataset name (determines which loader to use)
        cache_dir: Cache directory for storing datasets

    Returns:
        List of normalized PyG Data objects
    """
    source = _identify_dataset_source(name)

    # Step 1: Load raw data from appropriate source
    if source == "pygod":
        raw_data = load_pygod_data(name, cache_dir)
    elif source == "planetoid":
        root = cache_dir or os.path.join(os.path.expanduser('~'), '.pyg_data', 'Planetoid')
        dataset = Planetoid(root=root, name=name)
        raw_data = dataset  # InMemoryDataset with single graph
    elif source == "tudataset":
        root = cache_dir or os.path.join(os.path.expanduser('~'), '.pyg_data', 'TUDataset')
        dataset = TUDataset(root=root, name=name)
        raw_data = dataset  # InMemoryDataset with multiple graphs
    else:
        raise ValueError(f"Unknown dataset source: {source}")

    # Step 2: Convert to list of Data objects
    if isinstance(raw_data, Data):
        # Single graph from pygod
        data_list = [raw_data]
    elif isinstance(raw_data, Sequence) and not isinstance(raw_data, str):
        # List/tuple of graphs
        data_list = list(raw_data)
    elif hasattr(raw_data, '__len__') and hasattr(raw_data, '__getitem__'):
        # Dataset-like object (TUDataset, Planetoid, etc.)
        data_list = [raw_data[i] for i in range(len(raw_data))]
    else:
        raise TypeError(f"Unsupported data type from {source}: {type(raw_data)}")

    if not data_list:
        raise ValueError(f"Dataset '{name}' is empty")

    # Step 3: Normalize node features across all graphs
    # all_features = []
    # for graph in data_list:
    #     if not isinstance(graph, Data):
    #         raise TypeError(f"Expected PyG Data objects, got {type(graph)}")
    #     if graph.x is not None:
    #         all_features.append(graph.x)
    #
    # if all_features:
    #     # Compute global statistics across all graphs
    #     all_features_cat = torch.cat(all_features, dim=0)
    #     mean = all_features_cat.mean(dim=0, keepdim=True)
    #     std = all_features_cat.std(dim=0, keepdim=True) + 1e-8  # Avoid division by zero
    #
    #     # Apply normalization to each graph
    #     for graph in data_list:
    #         if graph.x is not None:
    #             graph.x = (graph.x - mean) / std
    #
    #     # Recompute statistics after normalization for verification
    #     normalized_features = torch.cat([g.x for g in data_list if g.x is not None], dim=0)
    #
    #     # Print normalization statistics
    #     print("\n" + "=" * 60)
    #     print(f"[Data Normalization] Dataset: {name}")
    #     print(f"  Source: {source}")
    #     print(f"  Number of graphs: {len(data_list)}")
    #     print(f"  Total nodes: {sum(g.num_nodes for g in data_list)}")
    #     print(f"  Features per node: {data_list[0].num_node_features}")
    #     print(f"  Normalized features:")
    #     print(f"    Mean: {normalized_features.mean():.6f} (target: ~0.0)")
    #     print(f"    Std: {normalized_features.std():.6f} (target: ~1.0)")
    #     print(f"    Min: {normalized_features.min():.6f}")
    #     print(f"    Max: {normalized_features.max():.6f}")
    #     print("=" * 60 + "\n")
    # else:
    #     print(f"\n[Warning] Dataset '{name}' has no node features to normalize\n")

    return data_list


class _GraphDataset(Dataset):
    """Dataset wrapper for a list of PyG Data objects."""

    def __init__(self, data_list: Sequence[Any]):
        if not data_list:
            raise ValueError("Graph dataset received an empty data list.")
        self._data_list = []
        for item in data_list:
            if not isinstance(item, Data):
                raise TypeError(f"Expected PyG Data instances, got {type(item)}")
            self._data_list.append(item)

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        return self._data_list[idx]


class GADDataModule(LightningDataModule):
    """LightningDataModule for graph anomaly detection datasets.

    Supports three data sources:
    - PyGOD datasets: weibo, reddit, disney, books, enron, inj_*, gen_*
    - Planetoid datasets: Cora, CiteSeer, PubMed
    - TUDataset: all other dataset names

    Handles both single-graph and multi-graph datasets automatically.
    Normalizes node features globally across all graphs.
    """

    def __init__(self,
                 dataset: str = "disney",
                 batch_size: int = 1,
                 num_workers: int = 0,
                 unify_anomaly_types: bool = False,
                 attr_anomaly=True,
                 struct_anomaly=True,
                 combined_anomaly=True,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data = None
        self.dataset = None
        self._train_ds = None
        self._val_ds = None
        self._test_ds = None
        self.unify_anomaly_types = unify_anomaly_types
        self.ignore_anomaly_types = []
        if not attr_anomaly:
            self.ignore_anomaly_types.append(1)
        if not struct_anomaly:
            self.ignore_anomaly_types.append(2)
        if not combined_anomaly:
            self.ignore_anomaly_types.append(3)

    @property
    def num_features(self) -> int:
        return int(self.data.num_node_features) if self.data is not None else 0

    def prepare_data(self):
        # Datasets are downloaded automatically by respective loaders if needed
        pass

    def setup(self, stage: Optional[str] = None):
        if self.dataset is None:
            # Load and normalize dataset from appropriate source
            data_list = _load_and_normalize_dataset(self.hparams.dataset)

            # Wrap in _GraphDataset for unified interface
            self.dataset = _GraphDataset(data_list)
            self.data = self.dataset[0]
        if self.unify_anomaly_types:
            collapse_anomaly_types(self.dataset, )
        if len(self.ignore_anomaly_types) > 0:
            ignore_anomaly_types(self.dataset, type_ids=self.ignore_anomaly_types)

        self._train_ds = self.dataset
        self._val_ds = self.dataset
        self._test_ds = self.dataset

    def train_dataloader(self):
        # Unsupervised method only
        return DataLoader(
            self._train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
