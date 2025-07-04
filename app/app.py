# app.py
import os
import uuid
import shutil
from contextlib import asynccontextmanager
import compiam
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from qdrant_client import QdrantClient, models
from ingestion.feature_extractor import MuleEmbedder, extract_features
from compiam.melody.pitch_extraction import FTANetCarnatic
from compiam.melody.tonic_identification.tonic_multipitch import TonicIndianMultiPitch

# --- CONFIG ---
# Ensure this script is run from the project root directory for paths to work correctly.
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "raga_clips_v1"
TOP_K_INITIAL = 25 # Retrieve more candidates for DTW refinement
TOP_K_FINAL = 5    # Final number of results to return
MULE_CONFIG_PATH = "music-audio-representations/supporting_data/configs/mule_embedding_average.yml"
TEMP_UPLOAD_DIR = "/tmp/raga_uploads"

# --- APP STATE ---
# We use a dictionary for app state to be compatible with FastAPI's dependency injection
app_state = {}

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup and cleanup on shutdown."""
    print("Loading models...")
    app_state["qdrant_client"] = QdrantClient(url=QDRANT_URL)
    app_state["ftanet"] = compiam.load_model("melody:ftanet-carnatic")
    app_state["mule_embedder"] = MuleEmbedder(MULE_CONFIG_PATH)
    app_state["tonic_identifier"] = TonicIndianMultiPitch()
    print("Models loaded.")

    if not os.path.exists(TEMP_UPLOAD_DIR):
        os.makedirs(TEMP_UPLOAD_DIR)
    
    yield

    print("Cleaning up temporary directory...")
    if os.path.exists(TEMP_UPLOAD_DIR):
        shutil.rmtree(TEMP_UPLOAD_DIR)

# --- FASTAPI APP ---
app = FastAPI(title="Raga Identification API", lifespan=lifespan)

# --- DTW REFINEMENT ---
def get_contour_for_file(file_path):
    """Extracts just the pitch contour for a given file path."""
    _, contour = extract_features(
        file_path,
        app_state["tonic_identifier"],
        app_state["ftanet"],
        app_state["mule_embedder"]
    )
    return contour

def dtw_re_rank(query_contour, candidates):
    """
    Re-ranks candidates based on DTW distance of pitch contours.
    """
    if query_contour is None or len(query_contour) == 0:
        print("Warning: Query contour is empty. Skipping DTW re-ranking.")
        return candidates # Cannot re-rank without a query contour

    ranked_results = []
    for candidate in candidates:
        # The `source_file` path is relative to the project root, as stored during ingestion.
        candidate_path = candidate.payload.get("source_file")
        if not candidate_path or not os.path.exists(candidate_path):
            print(f"Warning: Could not find source file for candidate {candidate.id} at path: {candidate_path}")
            continue

        candidate_contour = get_contour_for_file(candidate_path)
        if candidate_contour is None or len(candidate_contour) == 0:
            print(f"Warning: Could not extract contour for candidate file: {candidate_path}")
            continue

        # Use subsequence DTW to find the best alignment of the (short) query
        # contour within the (potentially longer) candidate contour.
        dtw_cost_matrix, _ = librosa.sequence.dtw(
            query_contour, candidate_contour, subseq=True,
        )
        
        # For subsequence DTW, the final cost is the minimum of the last row of the cost matrix.
        final_cost = np.min(dtw_cost_matrix[-1, :]) if dtw_cost_matrix.size > 0 else float('inf')

        ranked_results.append({
            "id": candidate.id,
            "payload": candidate.payload,
            "initial_score": candidate.score,
            "dtw_cost": final_cost
        })

    # Sort by DTW cost (lower is better)
    ranked_results.sort(key=lambda x: x["dtw_cost"])
    return ranked_results

# --- API ENDPOINT ---
@app.post("/query/")
async def query_raga(file: UploadFile = File(...)):
    """
    Upload an audio file, find the top-k matching ragas with DTW refinement.
    """
    # 1. Save uploaded file temporarily
    temp_path = os.path.join(TEMP_UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Extract features from the query file
        print(f"Extracting features from {temp_path}...")
        query_vector, query_contour = extract_features(
            temp_path,
            app_state["tonic_identifier"],
            app_state["ftanet"],
            app_state["mule_embedder"]
        )

        if query_vector is None:
            raise HTTPException(status_code=400, detail="Could not process the provided audio file. It might be too short or corrupted.")

        # 3. Query Qdrant for initial candidates
        print(f"Querying vector database for {TOP_K_INITIAL} candidates...")
        search_results = app_state["qdrant_client"].search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=TOP_K_INITIAL,
            with_payload=True
        )

        # 4. Filter for unique source files before DTW
        print(f"Filtering {len(search_results)} candidates down to unique source files...")
        unique_candidates = {}
        for result in search_results:
            source_file = result.payload.get("source_file")
            # Keep only the highest-scoring result for each unique source file
            if source_file not in unique_candidates:
                unique_candidates[source_file] = result
        
        unique_candidate_list = list(unique_candidates.values())
        print(f"Performing DTW re-ranking on {len(unique_candidate_list)} unique candidates...")

        # 5. DTW Re-ranking on the unique set
        final_results = dtw_re_rank(query_contour, unique_candidate_list)
        
        # 6. Format and return results
        return {
            "query_file": file.filename,
            "results": final_results[:TOP_K_FINAL]
        }

    except Exception as e:
        # Use traceback to get more detailed error information in the server logs
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        # 7. Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Raga Identification API. Please use the /docs endpoint for details."}