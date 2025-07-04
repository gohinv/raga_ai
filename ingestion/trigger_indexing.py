from qdrant_client import QdrantClient
from qdrant_client.http.models import OptimizersConfigDiff

# --- CONFIG ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "raga_clips_v1"
# Set a new, lower threshold to trigger indexing immediately
NEW_INDEXING_THRESHOLD = 1000

def trigger_indexing():
    """
    Connects to Qdrant and updates the collection's indexing threshold
    to force the indexing process to start.
    """
    try:
        client = QdrantClient(url=QDRANT_URL)
        print("Connected to Qdrant.")

        # Get current collection info
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        current_threshold = collection_info.config.optimizer_config.indexing_threshold
        points_count = collection_info.points_count
        
        print(f"Collection '{COLLECTION_NAME}' has {points_count} points.")
        print(f"Current indexing threshold is: {current_threshold}")

        if current_threshold > NEW_INDEXING_THRESHOLD:
            print(f"\nThreshold is preventing indexing. Updating to {NEW_INDEXING_THRESHOLD}...")
            client.update_collection(
                collection_name=COLLECTION_NAME,
                optimizer_config=OptimizersConfigDiff(indexing_threshold=NEW_INDEXING_THRESHOLD)
            )
            print("Successfully updated collection config. Indexing should begin shortly.")
        else:
            print("\nNo update needed. Indexing is either complete or should be in progress.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure Qdrant is running and the collection exists.")

if __name__ == "__main__":
    trigger_indexing() 