import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def analyze_data_pattern(points, values):
    """
    使用 kmeans 分群，嘗試 range(2, min(5, len(points)//2)+1)
    以 silhouette_score 找最佳 n_clusters。
    然後依 n_clusters 決定 polynomial degree。
    """
    max_clusters = min(5, len(points)//2)
    best_n_clusters = 2
    best_score = -1

    for n_clusters in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(points)
        if len(np.unique(clusters))>1:
            score = silhouette_score(points, clusters)
            if score>best_score:
                best_score=score
                best_n_clusters=n_clusters
    
    if best_n_clusters <=2:
        return {"degree":2, "n_clusters":best_n_clusters}
    elif best_n_clusters <=3:
        return {"degree":3, "n_clusters":best_n_clusters}
    else:
        return {"degree":4, "n_clusters":best_n_clusters}
