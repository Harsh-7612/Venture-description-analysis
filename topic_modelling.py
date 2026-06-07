import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

def get_top_3_topics(model, topic_id):
    """
    Helper function to extract the top 3 words for a given topic ID.
    Returns a comma-separated string.
    """
    words = model.get_topic(topic_id)
    if words is None or not words:
        return ""
    return ", ".join([word for word, _ in words[:3]])

def main():
    print("1. Loading and preprocessing data...")
    # Load dataset
    df = pd.read_csv('p1_filtered.csv')
    
    # Drop duplicates
    df = df.drop_duplicates(subset=['person1_venture_description'], keep='first')
    
    # Keep only necessary columns
    df = df[['person1_id', 'person1_venture_description']].copy()
    
    # Drop rows with missing descriptions to prevent BERTopic from crashing
    df = df.dropna(subset=['person1_venture_description'])
    
    # Extract documents as a list
    docs = df['person1_venture_description'].tolist()

    print("2. Setting up device and embedding model...")
    # Dynamically select GPU if available, else fallback to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize the embedding model once
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    print("3. Initializing and fitting BERTopic... (This may take a while)")
    # Pass the initialized embedding_model to avoid reloading it into memory
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=10,
        verbose=True  # Helpful for logging progress in the console
    )
    
    topics, probs = topic_model.fit_transform(docs)

    print("4. Assigning topics and saving intermediate results...")
    # Assign results to dataframe
    df['topics'] = topics
    df['topic_probability'] = probs  # Fixed spelling
    
    # Save intermediate results
    df.to_csv('with_topic_modelling.csv', index=False)

    print("5. Filtering out outliers and mapping topic names...")
    # Filter out outliers (Topic -1)
    df_filtered = df[df['topics'] != -1].copy()

    # Map topic ID to the top 3 words representing that topic
    cluster_to_name = {
        topic_id: get_top_3_topics(topic_model, topic_id)
        for topic_id in df_filtered["topics"].unique()
    }
    
    df_filtered["top_topics"] = df_filtered["topics"].map(cluster_to_name)

    print("6. Saving final named clusters...")
    df_filtered.to_csv("topic_clusters_named.csv", index=False)
    
    print("Done! Results saved to 'topic_clusters_named.csv'.")

if __name__ == "__main__":
    main()
