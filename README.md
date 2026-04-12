### Venture-description-analysis  

Summary:  

Developed an NLP-based venture description analysis pipeline to automatically cluster and categorize startup/business descriptions into meaningful thematic groups using transformer embeddings and BERTopic. This enables scalable market segmentation, startup landscape analysis, and rapid identification of venture domains from unstructured text data. The pipeline transforms raw venture descriptions into structured topic insights with confidence scores and interpretable labels, making large-scale entrepreneurial data easier to analyze for research and strategic decision-making.  

-----------------------------------------------------------------------------------------------------------------------

Detailed:  

Collected and loaded raw venture-description dataset containing startup/business descriptions from p1_filtered.csv.
Performed data preprocessing and deduplication by removing duplicate entries based on venture description text to ensure topic quality and reduce noise.    

Generated dense semantic embeddings for each venture description using the transformer model
SentenceTransformers with the pretrained all-MiniLM-L6-v2 encoder on GPU.  

Built an unsupervised topic modeling pipeline using BERTopic to discover latent thematic clusters from venture descriptions.  

Configured BERTopic with custom parameters (min_topic_size=10) to control cluster granularity and improve interpretability.  

Computed topic assignments and topic probability/confidence scores for every venture description.  

Filtered out BERTopic’s noise/outlier cluster (-1) to retain only high-confidence meaningful topic groups.
Generated human-interpretable labels for each cluster by extracting and aggregating top representative keywords per topic.  

Exported enriched datasets with:
Topic IDs
Topic probabilities
Human-readable topic labels  

Produced structured venture clusters enabling downstream applications such as:
Market segmentation
Startup ecosystem mapping
Venture trend analysis
Investor/researcher categorization pipelines
