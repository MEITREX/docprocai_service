from dataclasses import dataclass
import json
from psycopg import Connection
from sentence_transformers import util

@dataclass
class PageData:
    id: str
    page: int
    clusters: list[int]


def compute_clusters(db_conn: Connection):
    # Fetch embeddings from the database
    result = db_conn.execute("SELECT id, page, embedding FROM documents").fetchall()
    
    # Prepare the embeddings for clustering
    embeddings = []
    data: list[PageData] = []

    for row in result:
        doc_id, page, embedding = row

        embeddings.append(embedding)
        data.append(PageData(id=doc_id, page=page, clusters=[]))
    
    # Compute clusters
    clusters = util.community_detection(embeddings)

    for cluster_id, cluster in enumerate(clusters):
        for i in cluster:
            data[i].clusters.append(cluster_id)

    update_data = [(json.dumps(d.clusters), d.id, d.page) for d in data]

    # Update the database with cluster ids and cluster elements
    with db_conn.cursor() as cur:
        cur.executemany(
            "UPDATE documents SET clusters = %s WHERE id = %s AND page = %s",
            update_data
        )
        db_conn.commit()


