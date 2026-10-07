"""Step 3: cluster-as-surrogate-label for self-supervised patch training.

RootSIFT -> PCA(32, whitened) -> MiniBatchKMeans(5000) on a sample of training
descriptors; then every training patch gets its nearest-centre label, and
ambiguous patches (d1 >= rho * d2) are marked as not kept.
Outputs: WORK/clusters/{pca.npz, centers.npy} and labels/<page_id>.npz
Run:  python cluster_labels.py
"""
import argparse

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

import config as C


def rootsift(d):
    d = d.astype(np.float32)
    d /= np.abs(d).sum(1, keepdims=True) + 1e-7
    return np.sqrt(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    files = sorted((C.WORK / "patches" / "train").glob("*.npz"))
    out = C.WORK / "clusters"
    (out / "labels").mkdir(parents=True, exist_ok=True)

    # 1) sample descriptors
    sample = []
    for f in tqdm(files, desc="sampling"):
        d = np.load(f)["desc"]
        if len(d) > C.DESC_SAMPLE_PER_PAGE:
            d = d[rng.choice(len(d), C.DESC_SAMPLE_PER_PAGE, replace=False)]
        sample.append(d)
    X = rootsift(np.concatenate(sample))
    print("descriptor sample:", X.shape)

    # 2) PCA + k-means
    pca = PCA(n_components=C.PCA_DIM, whiten=True, random_state=a.seed).fit(X)
    Z = pca.transform(X)
    km = MiniBatchKMeans(n_clusters=C.N_CLUSTERS, batch_size=20000, n_init=3,
                         random_state=a.seed, verbose=0).fit(Z)
    np.savez(out / "pca.npz", mean=pca.mean_, components=pca.components_,
             var=pca.explained_variance_)
    np.save(out / "centers.npy", km.cluster_centers_)

    # 3) label all training patches with ratio filter
    centers = km.cluster_centers_.astype(np.float32)
    c2 = (centers ** 2).sum(1)
    kept = total = 0
    used = np.zeros(C.N_CLUSTERS, np.int64)
    for f in tqdm(files, desc="labelling"):
        z = pca.transform(rootsift(np.load(f)["desc"])).astype(np.float32)
        if len(z) == 0:
            np.savez(out / "labels" / f.name, label=np.zeros(0, np.int64), keep=np.zeros(0, bool))
            continue
        d2 = (z ** 2).sum(1, keepdims=True) - 2 * z @ centers.T + c2  # squared dists
        nn = np.argpartition(d2, 1, axis=1)[:, :2]
        first = np.take_along_axis(d2, nn, 1)
        order = np.argsort(first, 1)
        nn = np.take_along_axis(nn, order, 1)
        first = np.sqrt(np.maximum(np.take_along_axis(first, order, 1), 0))
        keep = first[:, 0] < C.RATIO_RHO * first[:, 1]
        label = nn[:, 0]
        np.savez(out / "labels" / f.name, label=label, keep=keep)
        kept += keep.sum(); total += len(keep)
        np.add.at(used, label[keep], 1)
    print(f"kept {kept}/{total} patches ({kept / max(total, 1):.1%}); "
          f"non-empty clusters: {(used > 0).sum()}/{C.N_CLUSTERS}")


if __name__ == "__main__":
    main()
