import os
import tiktoken
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.storage import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

# Load environment variables
load_dotenv()

# Initialize tokenizer for counting tokens
tokenizer = tiktoken.get_encoding(
    "cl100k_base"
)  # Using OpenAI's tokenizer as approximation


def count_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string"""
    return len(tokenizer.encode(text))


def process_folder(
    folder_path: str, max_tokens_per_batch: int = 600_000
) -> List[List[str]]:
    """
    Process a folder and create batches of files not exceeding max_tokens_per_batch
    Returns a list of batches, where each batch is a list of file paths
    """
    all_files = []

    # Walk through the directory and collect all files
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith((".txt", ".md", ".html", ".pdf", ".docx")):
                all_files.append(os.path.join(root, file))

    # Create batches
    batches = []
    current_batch = []
    current_batch_tokens = 0

    for file_path in all_files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            file_tokens = count_tokens(content)

            # If adding this file would exceed the token limit, start a new batch
            if (
                current_batch_tokens + file_tokens > max_tokens_per_batch
                and current_batch
            ):
                batches.append(current_batch)
                current_batch = []
                current_batch_tokens = 0

            # Add the file to the current batch
            current_batch.append(file_path)
            current_batch_tokens += file_tokens

            print(f"Added {file_path} to batch (tokens: {file_tokens})")

        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")

    # Add the last batch if not empty
    if current_batch:
        batches.append(current_batch)

    print(f"\nCreated {len(batches)} batches of files")
    for i, batch in enumerate(batches):
        batch_tokens = sum(
            [
                count_tokens(open(f, "r", encoding="utf-8", errors="ignore").read())
                for f in batch
            ]
        )
        print(f"Batch {i+1}: {len(batch)} files, ~{batch_tokens} tokens")

    return batches


def update_jina_api_key():
    """Prompt user for a new Jina API key and update the environment variable"""
    print("\nYour current Jina API key:", os.environ.get("JINA_API_KEY", "Not set"))
    new_key = input("Enter new Jina API key (press Enter to keep current): ").strip()

    if new_key:
        os.environ["JINA_API_KEY"] = new_key
        print("Jina API key updated")

    return os.environ["JINA_API_KEY"]


def upload_batch_to_qdrant(
    batch_files: List[str], collection_name: str = "online_resources"
):
    """Upload a batch of files to Qdrant"""
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    jina_api_key = os.environ.get("JINA_API_KEY")
    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not all([qdrant_url, qdrant_api_key, jina_api_key, groq_api_key]):
        raise ValueError("Missing required environment variables.")

    # Initialize Qdrant client
    qdrant_client_instance = qdrant_client.QdrantClient(
        url=qdrant_url, api_key=qdrant_api_key
    )

    # Initialize embedding model and LLM
    embed_model = JinaEmbedding(
        api_key=jina_api_key,
        model="jina-embeddings-v3",
    )

    llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key)

    # Configure global settings
    Settings.embed_model = embed_model
    Settings.llm = llm

    # Initialize vector store with optimized settings
    vector_store = QdrantVectorStore(
        client=qdrant_client_instance,
        collection_name=collection_name,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Process each file in the batch
    all_documents = []

    for file_path in batch_files:
        try:
            # Create a document subdirectory
            subdir = os.path.dirname(file_path)
            temp_reader = SimpleDirectoryReader(input_files=[file_path])
            docs = temp_reader.load_data()

            for doc in docs:
                # Add the original file path to metadata
                if not doc.metadata:
                    doc.metadata = {}
                doc.metadata["file_path"] = file_path
                doc.metadata["file_name"] = os.path.basename(file_path)
                all_documents.append(doc)

            print(f"Processed {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    # Create or update the index
    print(f"Creating index with {len(all_documents)} documents...")

    try:
        # Check if collection exists
        collections = qdrant_client_instance.get_collections()
        collection_exists = any(
            col.name == collection_name for col in collections.collections
        )

        if not collection_exists:
            # Create collection with optimized configuration
            print(f"Creating new collection: {collection_name}")

            # Get vector size from embedding model
            vector_size = 1024

            # Create the collection with optimized settings
            qdrant_client_instance.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "size": vector_size,
                    "distance": "Cosine",
                },
                quantization_config={
                    "scalar": {
                        "type": "int8",
                        "always_ram": True,  # Keep quantized vectors in RAM
                    }
                },
                hnsw_config={
                    "m": 16,  # Lower value for faster indexing
                    "ef_construct": 100,  # Lower value for faster indexing
                },
                optimizers_config={
                    "default_segment_number": 2,  # Use fewer segments for better throughput
                },
            )

            # Create index with the documents
            VectorStoreIndex.from_documents(
                all_documents,
                storage_context=storage_context,
            )

        else:
            print(f"Updating existing collection: {collection_name}")
            # Get the existing index
            index = VectorStoreIndex.from_vector_store(
                vector_store, storage_context=storage_context
            )

            # Insert the documents
            for doc in all_documents:
                index.insert(doc)

        print("Successfully uploaded batch to Qdrant")
        return True

    except Exception as e:
        print(f"Error uploading to Qdrant: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload data to Qdrant in batches")
    parser.add_argument(
        "--folder",
        type=str,
        default="../../data/online_resources",
        help="Path to the folder containing documents to upload",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=600000,
        help="Maximum number of tokens per batch",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="online_resources",
        help="Qdrant collection name",
    )

    args = parser.parse_args()

    print(f"Processing folder: {args.folder}")
    batches = process_folder(args.folder, max_tokens_per_batch=args.batch_size)

    for i, batch in enumerate(batches):
        print(f"\n--- Processing Batch {i+1}/{len(batches)} ---")

        # Allow user to update API key before processing each batch
        jina_api_key = update_jina_api_key()

        success = upload_batch_to_qdrant(batch, collection_name=args.collection)

        if success:
            print(f"Batch {i+1}/{len(batches)} successfully uploaded")
        else:
            print(f"Failed to upload batch {i+1}/{len(batches)}")

            retry = input("Retry this batch? (y/n): ").lower()
            if retry == "y":
                i -= 1  # Retry the same batch

        if i < len(batches) - 1:  # If not the last batch
            proceed = input("\nContinue to next batch? (y/n): ").lower()
            if proceed != "y":
                print("Batch processing paused. Run the script again to continue.")
                break

    print("\nBatch processing completed!")


if __name__ == "__main__":
    main()
