import time
from qdrant_client import QdrantClient
from qdrant_client.http.models import OptimizersStatus

# --- CONFIG ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "raga_clips_v1"

def check_indexing_status():
    """
    Connects to Qdrant and periodically checks the indexing progress
    of the collection until it's complete.
    """
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=20.0)
        print(f"Connecting to Qdrant at {QDRANT_URL}...")

        while True:
            collection_info = client.get_collection(collection_name=COLLECTION_NAME)
            
            total_points = collection_info.points_count
            indexed_points = collection_info.indexed_vectors_count
            optimizer_status = collection_info.optimizer_status

            if isinstance(optimizer_status, OptimizersStatus) and not optimizer_status.OK:
                print(f"\nOptimizer status is not OK: {optimizer_status.error}")
                print("Indexing may have encountered an error.")
                break
            
            progress_percent = (indexed_points / total_points) * 100 if total_points > 0 else 100
            
            print(
                f"\rIndexing progress: {indexed_points} / {total_points} vectors ({progress_percent:.2f}%) complete. "
                f"Status: [{optimizer_status.name if hasattr(optimizer_status, 'name') else optimizer_status}]",
                end=""
            )

            if indexed_points == total_points:
                print("\n\nIndexing is 100% complete!")
                break

            time.sleep(5)

    except Exception as e:
        print(f"\n\nCould not check status: {e}")
        print("Please ensure Qdrant is running and the collection exists.")

if __name__ == "__main__":
    check_indexing_status() 