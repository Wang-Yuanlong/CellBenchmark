import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from resources.maes_microscopy.huggingface_mae import MAEModel
from sc.eval.dt import _PathDataset
from sc.utils.helper_func import get_model
from sc.models.model_config import MCViTConfig

class _PreEmbedDataset(_PathDataset):
    def __init__(self, paths, abs_path, transform=None):
        super().__init__(paths, abs_path, transform)
    
    def _load_parquet(self, rel_path: str) -> torch.Tensor:
        path_parts = rel_path.split('_')
        data_path = os.path.join(
            self.abs_path,
            path_parts[0],
            'Plate' + path_parts[1],
            'embeddings.parquet'
        )
        
        data = pd.read_parquet(data_path)
        row = data[data['well_id'] == rel_path]
        if row.empty:
            raise ValueError(f"Well ID {rel_path} not found in {data_path}")
        emb_array = row.iloc[0, 1:].values.astype(np.float32)
        emb_tensor = torch.from_numpy(emb_array)
        return emb_tensor

    def __getitem__(self, index: int):
        rel_path = self.paths[index]
        emb_tensor = self._load_parquet(rel_path)
        if self.transform:
            emb_tensor = self.transform(emb_tensor)
        return emb_tensor, index

class ModelWrapper(torch.nn.Module):
    def __init__(self, model_type, model):
        super().__init__()
        self.model_type = model_type
        self.model = model

    def forward(self, x):
        if self.model_type == 'baseline':
            return self.model.predict(x)
        else:
            return self.model(x)['pooler_output']

def load_model(model_type, model_path):
    if model_type == 'baseline':
        return ModelWrapper(model_type, MAEModel.from_pretrained(model_path))
    else:
        ckpt_list = sorted(os.listdir(model_path))
        if len(ckpt_list) == 0:
            raise ValueError(f"No checkpoints found in {model_path}")
        if 'encoder' in ckpt_list:
            encoder_ckpt = True
            model_path = os.path.join(model_path, 'encoder')
        else:
            encoder_ckpt = False
            ckpt_list = [ckpt for ckpt in ckpt_list if ckpt.startswith("checkpoint")]
            model_path = os.path.join(model_path, ckpt_list[-1])
        ckpt_path = os.path.join(model_path, "model.safetensors")
        print(f"Loading model from {ckpt_path}...")
        model_config = MCViTConfig(
            image_size=224,
            patch_size=16,
            in_channels=6,
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            intermediate_size=384 * 4,
            use_instance_norm=True,
            use_flash_attention=True,
            use_cls_token=False,
        )
        return ModelWrapper(model_type, get_model(ckpt_path, model_config, encoder_ckpt=encoder_ckpt))
    
class CustomTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        C, H, W = x.shape

        if H != 224 or W != 224:
            x = F.interpolate(x.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        return x

class FourQuadrantCrop:
    def __init__(self, in_size=512, crop_size=256, out_size=224):
        self.in_size = in_size
        self.crop_size = crop_size
        self.out_size = out_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        C, H, W = x.shape
        if (H, W) != (self.in_size, self.in_size):
            raise ValueError(f"Expected {(self.in_size, self.in_size)}, got {(H,W)}")

        TL = x[:, 0:self.crop_size, 0:self.crop_size]
        TR = x[:, 0:self.crop_size, self.in_size-self.crop_size:self.in_size]
        BL = x[:, self.in_size-self.crop_size:self.in_size, 0:self.crop_size]
        BR = x[:, self.in_size-self.crop_size:self.in_size, self.in_size-self.crop_size:self.in_size]

        crops = torch.stack([TL, TR, BL, BR], dim=0)     
        crops = F.interpolate(crops, size=(self.out_size, self.out_size),
                              mode="bilinear", align_corners=False)                  
        return crops
    
class FourCropEncoder(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, x4: torch.Tensor):
        B, N, C, H, W = x4.shape
        feats = self.encoder(x4.view(B*N, C, H, W))             
        feats = feats.view(B, N, -1)                           
        avg = feats.mean(dim=1)                              
        return avg            

def filter_relationships(df: pd.DataFrame):
    """
    Filters a DataFrame of relationships between entities, removing any rows with self-relationships, ie. where
        the same entity appears in both columns, and also removing any duplicate relationships (A-B and B-A).

    Args:
        df (pd.DataFrame): DataFrame containing columns 'entity1' and 'entity2', representing the entities involved in
        each relationship.

    Returns:
        pd.DataFrame: DataFrame containing columns 'entity1' and 'entity2', representing the entities involved in
        each relationship after removing any rows where the same entity appears in both columns.
    """
    df["sorted_entities"] = df.apply(lambda row: tuple(sorted([row.entity1, row.entity2])), axis=1)
    df["entity1"] = df.sorted_entities.apply(lambda x: x[0])
    df["entity2"] = df.sorted_entities.apply(lambda x: x[1])
    return df[["entity1", "entity2"]].query("entity1!=entity2").drop_duplicates()

def get_benchmark_relationships(benchmark_data_dir: str, src: str, filter=True):
    """
    Reads a CSV file containing benchmark data and returns a filtered DataFrame.

    Args:
        benchmark_data_dir (str): The directory containing the benchmark data files.
        src (str): The name of the source containing the benchmark data.
        filter (bool, optional): Whether to filter the DataFrame. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the benchmark relationships.
    """
    df = pd.read_csv(Path(benchmark_data_dir).joinpath(src + ".txt"))
    return filter_relationships(df) if filter else df

def compute_recall(
    null_distribution: np.ndarray,
    query_distribution: np.ndarray,
    recall_threshold_pairs: list,
) -> dict:
    """Compute recall at given percentage thresholds for a query distribution with respect to a null distribution.
    Each recall threshold is a pair of floats (left, right) where left and right are floats between 0 and 1.

    Args:
        null_distribution (np.ndarray): The null distribution to compare against
        query_distribution (np.ndarray): The query distribution
        recall_threshold_pairs (list) A list of pairs of floats (left, right) that represent different recall threshold
            pairs, where left and right are floats between 0 and 1.

    Returns:
        dict: A dictionary of metrics with the following keys:
            - null_distribution_size: the size of the null distribution
            - query_distribution_size: the size of the query distribution
            - recall_{left_threshold}_{right_threshold}: recall at the given percentage threshold pair(s)
    """

    metrics = {}
    metrics["null_distribution_size"] = null_distribution.shape[0]
    metrics["query_distribution_size"] = query_distribution.shape[0]

    sorted_null_distribution = np.sort(null_distribution)
    query_percentage_ranks_left = np.searchsorted(sorted_null_distribution, query_distribution, side="left") / len(
        sorted_null_distribution
    )
    query_percentage_ranks_right = np.searchsorted(sorted_null_distribution, query_distribution, side="right") / len(
        sorted_null_distribution
    )
    for threshold_pair in recall_threshold_pairs:
        left_threshold, right_threshold = np.min(threshold_pair), np.max(threshold_pair)
        metrics[f"recall_{left_threshold}_{right_threshold}"] = sum(
            (query_percentage_ranks_right <= left_threshold) | (query_percentage_ranks_left >= right_threshold)
        ) / len(query_distribution)
    return metrics


def convert_metrics_to_df(metrics: dict, source: str) -> pd.DataFrame:
    """
    Convert metrics dictionary to dataframe to be used in summary.

    Args:
        metrics (dict): metrics dictionary
        source (str): benchmark source name

    Returns:
        pd.DataFrame: a dataframe with metrics
    """
    metrics_dict_with_list = {key: [value] for key, value in metrics.items()}
    metrics_dict_with_list["source"] = [source]
    return pd.DataFrame.from_dict(metrics_dict_with_list)


def known_relationship_benchmark(
    embed_df: pd.DataFrame,
    gene_col: str = "gene",
    benchmark_sources: list = ["CORUM", "HuMAP", "Reactome", "SIGNOR", "StringDB"],
    recall_thr_pairs: list = [(0.05, 0.95), (0.1, 0.9)],
    min_req_entity_cnt: int = 20,
    benchmark_data_dir: str = '../../data/benchmark_annotations/',
    log_stats: bool = False,
) -> pd.DataFrame:
    """
    Perform benchmarking on aggregated map data against biological relationships.

    Args:
        map_data (Bunch): The map data containing `features` and `metadata` attributes.
        pert_col (str, optional): Column name for perturbation labels.
        benchmark_sources (list, optional): List of benchmark sources. Defaults to cst.BENCHMARK_SOURCES.
        recall_thr_pairs (list, optional): List of recall percentage threshold pairs. Defaults to cst.RECALL_PERC_THRS.
        min_req_entity_cnt (int, optional): Minimum required entity count for benchmarking.
            Defaults to cst.MIN_REQ_ENT_CNT.
        benchmark_data_dir (str, optional): Path to benchmark data directory. Defaults to cst.BENCHMARK_DATA_DIR.
        log_stats (bool, optional): Whether to print out the number of statistics used while computing the benchmarks.
            Defaults to False (i.e, no logging).

    Returns:
        pd.DataFrame: a dataframe with benchmarking results. The columns are:
            "source": benchmark source name
            "recall_{low}_{high}": recall at requested thresholds
    """

    if not len(benchmark_sources) > 0 and all([src in os.listdir(benchmark_data_dir) for src in benchmark_sources]):
        ValueError("Invalid benchmark source(s) provided.")
    if gene_col not in embed_df.columns:
        ValueError(f"Provided gene_col '{gene_col}' not found in embed_df columns.")

    features = embed_df.set_index(gene_col).rename_axis(index=None)
    del embed_df
    if not len(features) == len(set(features.index)):
        ValueError("Duplicate perturbation labels in the map.")
    if not len(features) >= min_req_entity_cnt:
        ValueError("Not enough entities in the map for benchmarking.")
    if log_stats:
        print(len(features), "perturbations exist in the map.")
    
    features_array = np.array(features['emb'].to_list()).astype(np.float32)
    features = pd.DataFrame(features_array, index=features.index)

    metrics_lst = []
    cossim_matrix = pd.DataFrame(cosine_similarity(features.values, features.values), index=features.index, columns=features.index)
    cossim_values = cossim_matrix.values[np.triu_indices(cossim_matrix.shape[0], k=1)]
    for s in benchmark_sources:
        rels = get_benchmark_relationships(benchmark_data_dir, s)
        rels = rels[rels.entity1.isin(features.index) & rels.entity2.isin(features.index)]
        query_cossim = np.array([cossim_matrix.loc[e1, e2] for e1, e2 in rels.itertuples(index=False)])
        if log_stats:
            print(len(query_cossim), "relationships are used from the benchmark source", s)
        if len(query_cossim) > 0:
            metrics_lst.append(
                convert_metrics_to_df(metrics=compute_recall(cossim_values, query_cossim, recall_thr_pairs), source=s)
            )
    return pd.concat(metrics_lst, ignore_index=True)

class GeneGeneEvalPrep:
    def __init__(
        self,
        genetic_csv: str,
        well_root: str,
        encoder: nn.Module,                      
        transform=None,               
        device: str = "cuda",
        batch_size: int = 256,
        num_workers: int = 16,
        pin_memory: bool = True,
        amp: bool = True,       
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device).eval()
        self.transform = transform
        self.bs = batch_size
        self.nw = num_workers
        self.pm = pin_memory
        self.amp = amp

        self.genetic = pd.read_csv(genetic_csv)
        self.well_root = well_root

        self.dataset_class = _PreEmbedDataset if isinstance(encoder, nn.Identity) else _PathDataset

        need_gene = {"well_id", "experiment_name", "plate", "gene"}
        assert need_gene.issubset(self.genetic.columns), f"genetic.csv need {need_gene}"


                                         
    @torch.no_grad()
    def _encode_paths(self, paths: List[str]) -> np.ndarray:
        ds = self.dataset_class(paths, self.well_root, self.transform)
        dl = DataLoader(ds, batch_size=self.bs, shuffle=False, num_workers=self.nw,
                        pin_memory=self.pm)
        embs = [None] * len(paths)
        scaler = torch.amp.autocast
        with scaler(device_type=self.device.type, enabled=self.amp):
            for xb, idx in dl:
                xb = xb.to(self.device, non_blocking=True)
                feats = self.encoder(xb)                        
                feats = feats.detach().to("cpu").float().numpy()
                for k, j in enumerate(idx.tolist()):
                    embs[j] = feats[k]
        return np.stack(embs, axis=0)

    @staticmethod
    def cs_stats_from_controls(ctrl: Dict[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Returns: dict[group] -> (mu[D], std[D])
        """
        stats = {}
        for group, X in ctrl.items():
            mu = X.mean(axis=0)
            sd = X.std(axis=0, ddof=0)
            sd[sd < 1e-8] = 1.0
            stats[group] = (mu, sd)
        return stats
 
    @staticmethod
    def fit_pca_cs_from_controls(
        ctrl: Dict[str, np.ndarray],
        n_components: Optional[int] = None,                         
        whiten: bool = False
    ) -> Dict[str, object]:
        """Return {'pca': PCA, 'stats': {group: (mu, sd)}} where mu/sd are in PCA space."""
        if not ctrl:
            raise ValueError("No controls provided to fit PCA-CS.")
        X_all = np.vstack([X for X in ctrl.values() if len(X)])
        pca = PCA(n_components=n_components, whiten=whiten).fit(X_all)
        stats = {}
        for group, X in ctrl.items():
            if not len(X): continue
            Z = pca.transform(X)             
            mu = Z.mean(axis=0)
            sd = Z.std(axis=0, ddof=0)
            sd[sd < 1e-8] = 1.0
            stats[group] = (mu, sd)
        return {"pca": pca, "stats": stats}

    @staticmethod
    def _apply_pca_cs_vec(v: np.ndarray, group: str, pca_cs: Dict[str, object]) -> np.ndarray:
        z = pca_cs["pca"].transform(v[None, :])[0]
        mu, sd = pca_cs["stats"].get(group, (None, None))
        if mu is not None:
            z = (z - mu) / sd
        return z
    
    def compute_genetic_controls(self, batch_key="plate") -> Dict[str, np.ndarray]:
        """
        Returns: dict[group] -> controls matrix [n_ctrl, D]
        Controls = rows where gene == 'EMPTY_control' (case-sensitive per your note).
        """
        df = self.genetic[self.genetic["gene"] == "EMPTY_control"].reset_index(drop=True)
        if df.empty:
            return {}
        out: Dict[str, np.ndarray] = {}
        print("encoding genetic controls (EMPTY_control)...")
        for group, sub in tqdm(df.groupby(batch_key)):
            X = self._encode_paths(sub["well_id"].tolist())
            out[group] = X
        return out

    def encode_all_genetic(self, batch_key="plate") -> pd.DataFrame:
        """
        Encodes all genetic rows (including controls). No C&S here by default.
        Returns columns: ['gene', batch_key,'well_id','emb']
        """
        df = self.genetic.copy().reset_index(drop=True)
        df = df[df['gene'] != 'EMPTY_control'].reset_index(drop=True)
        if df.empty:
            return pd.DataFrame(columns=["gene", batch_key,"well_id","emb"])

                        
        recs = []
        for group, sub in tqdm(df.groupby(batch_key)):
            X = self._encode_paths(sub["well_id"].tolist())         
            for i, (_, row) in enumerate(sub.iterrows()):
                recs.append((row["gene"], group, row["well_id"], X[i]))
        return pd.DataFrame(recs, columns=["gene", batch_key,"well_id","emb"])
    
    def encode_genes(
        self,
        batch_key="plate",
        pca_cs: Optional[Dict[str, object]] = None,
        cs_stats: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> pd.DataFrame:

        df_enc = self.encode_all_genetic(batch_key=batch_key)                                        
        if df_enc.empty:
            return pd.DataFrame(columns=["gene","emb"])

        def _mean_emb(x):
            arr = np.stack(x.to_list(), axis=0)
            return arr.mean(axis=0)

                                                        
        prof_batch = (df_enc.groupby(["gene", batch_key], as_index=False)
                            .agg(emb=("emb", _mean_emb)))

                                            
        if pca_cs is not None:
            prof_batch["emb"] = [
                self._apply_pca_cs_vec(v, group, pca_cs) for v, group in zip(prof_batch["emb"], prof_batch[batch_key])
            ]
        elif cs_stats is not None and len(cs_stats) > 0:
            new_embs = []
            for v, group in zip(prof_batch["emb"], prof_batch[batch_key]):
                mu, sd = cs_stats.get(group, (None, None))
                if mu is not None:
                    v = (v - mu) / sd
                new_embs.append(v)
            prof_batch["emb"] = new_embs

                                                
        prof = (prof_batch.groupby(["gene"], as_index=False)
                        .agg(emb=("emb", _mean_emb)))

        return prof 

def main(args):
    assert args.model_type in ['baseline', 'dino', 'mae', 'simclr', 'wsl'], 'Invalid model type'
    assert args.model_type == 'baseline' or os.path.basename(args.model_path).startswith(args.model_type), 'Model path does not match model type'
    os.makedirs(args.output_path, exist_ok=True)
    item = os.path.basename(args.model_path)
    suffix = '' if args.model_type == 'baseline' else f'_[{item}]'
    suffix += '' if args.batch_key == 'plate' else f'_batch({args.batch_key})'

    if not args.use_pre_embed:
        model = load_model(args.model_type, args.model_path).to(args.device)
        transform = CustomTransform()
        if args.use_four_quad_crops:
            suffix += '_4quad'
            model = FourCropEncoder(model)
            transform = FourQuadrantCrop(in_size=512, crop_size=256, out_size=224)
    else:
        suffix += '_preembed'
        model = nn.Identity()
        transform = None

    prep = GeneGeneEvalPrep(
        genetic_csv=args.genetic_csv,
        well_root=args.well_root,
        encoder=model,
        transform=transform,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        amp=args.amp,
    )

    pca_cs = None
    cs_stats = None
    if args.use_pca_cs:
        print("Fitting PCA-CS from controls...")
        suffix += '_pca_cs'
        ctrl = prep.compute_genetic_controls(batch_key=args.batch_key)
        pca_cs = prep.fit_pca_cs_from_controls(ctrl, n_components=None, whiten=False)
    elif args.use_cs_stats:
        print("Computing C&S stats from controls...")
        suffix += '_cs_stats'
        ctrl = prep.compute_genetic_controls(batch_key=args.batch_key)
        cs_stats = prep.cs_stats_from_controls(ctrl)
    print("Encoding genes...")
    prof = prep.encode_genes(batch_key=args.batch_key, pca_cs=pca_cs, cs_stats=cs_stats)

    print("Running GGI benchmark...")
    results = known_relationship_benchmark(
        embed_df=prof,
        gene_col="gene",
        benchmark_sources=["CORUM", "HuMAP", "Reactome", "SIGNOR", "StringDB"],
        recall_thr_pairs=[(0.05, 0.95), (0.1, 0.9)],
        min_req_entity_cnt=20,
        benchmark_data_dir='./data/benchmark_annotations/',
        log_stats=True,
    )
    results.to_csv(os.path.join(args.output_path, f"ggi_results{suffix}.csv"), index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--genetic_csv', type=str, default='data/rxrx/rxrx3/rxrx3_gene.csv', help='Path to genetic CSV file.')
    parser.add_argument('--well_root', type=str, default='data/rxrx/rxrx3/processed', help='Root directory for well images.')
    parser.add_argument('--model_type', type=str, default='baseline', help='Type of model to use for encoding.')
    parser.add_argument('--model_path', type=str, default='resources/OpenPhenom', help='Path to the pretrained model.')
    parser.add_argument('--batch_key', type=str, default='plate', help='Column name to use as batch key for grouping.')
    parser.add_argument('--use_pca_cs', action='store_true', help='Whether to use PCA-CS normalization.')
    parser.add_argument('--use_cs_stats', action='store_true', help='Whether to use C&S normalization.')
    parser.add_argument('--use_four_quad_crops', action='store_true', help='Whether to use four quadrant cropping.')
    parser.add_argument('--use_pre_embed', action='store_true', help='Whether to use pre-embedded data.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for data loading.')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading.')
    parser.add_argument('--pin_memory', action='store_true', help='Whether to pin memory during data loading.')
    parser.add_argument('--amp', action='store_true', help='Whether to use automatic mixed precision.')
    parser.add_argument('--output_path', type=str, default='eval_results', help='Path to save the results CSV.')
    args = parser.parse_args()
    print(args)
    main(args)