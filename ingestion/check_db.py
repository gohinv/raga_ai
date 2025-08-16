from qdrant_client import QdrantClient

# --- CONFIG ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "raga_clips_v1"

def check_database_status():
    """Connects to Qdrant and prints the status of the collection."""
    try:
        client = QdrantClient(url=QDRANT_URL)
        
        # Check if collection exists
        try:
            collection_info = client.get_collection(collection_name=COLLECTION_NAME)
            print(f"Successfully connected to Qdrant and found collection '{COLLECTION_NAME}'.")
            print("\n--- Collection Info ---")
            print(collection_info)
            
            # Retrieve a few sample points
            if collection_info.points_count > 0:
                print("\n--- Sample Points (first 5) ---")
                sample_points, _ = client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=5,
                    with_payload=True,
                    with_vectors=False, # Set to True to also retrieve vectors
                )
                for point in sample_points:
                    print(f"ID: {point.id}, Payload: {point.payload}")
            else:
                print("\nThe collection is empty.")

        except Exception:
            print(f"Collection '{COLLECTION_NAME}' not found.")
            collections = client.get_collections().collections
            if collections:
                print("\nAvailable collections:")
                for collection in collections:
                    print(f"- {collection.name}")
            else:
                print("\nNo collections found in the database.")

    except Exception as e:
        print(f"Could not connect to Qdrant at {QDRANT_URL}.")
        print("Please ensure the Qdrant Docker container is running.")
        print(f"Error: {e}")

if __name__ == "__main__":
    check_database_status() 