# Text Image Retrieval using Vector Database

This project demonstrates an image retrieval system based on text queries, leveraging vector embeddings and a vector database for efficient search and retrieval. The system is built using ChromaDB and OpenCLIP, which allows for both L2 and cosine similarity-based searches.

## Features

- **Text-to-Image Retrieval**: Search for images that match a given text description using vector embeddings.
- **Image-to-Image Retrieval**: Query similar images using an image as input.
- **Efficient Search**: Utilize a vector database (ChromaDB) to store and efficiently retrieve image embeddings.
- **Support for Multiple Similarity Metrics**: Search images using both L2 distance and cosine similarity.

## Requirements

    Read 'requirements.txt' file

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/pipipip169/Textimage-Retrieval.git
    cd Textimage-Retrieval
    ```

2. **Install dependencies**:

    ```bash
    pip install chromadb open-clip-torch numpy tqdm pillow matplotlib
    ```

3. **Download and unzip the dataset**:

    Download the dataset and unzip it into the `data/` directory:

    ```bash
    https://drive.usercontent.google.com/download?id=1msLVo0g0LFmL9-qZ73vq9YEVZwbzOePF
    ```

## Usage

### 1. Image Embedding

- Extract image embeddings using OpenCLIP and store them in the vector database.

### 2. Create Collections in ChromaDB

- Create collections using either L2 or cosine similarity to store the embeddings.

### 3. Perform Search Queries

- Use the provided search functions to retrieve images based on either a text query or an image query. The search can be performed using L2 distance or cosine similarity.

### 4. Visualize Results

- Use the `plot_results` function to display the query image and the top retrieved images.

## Example Code

Here is an overview of the steps you need to follow:

1. **Initialize the environment**:

    ```python
    import chromadb
    from PIL import Image
    import numpy as np
    from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
    ```

2. **Generate Embeddings and Store in ChromaDB**:

    ```python
    # Initialize the embedding function
    embedding_function = OpenCLIPEmbeddingFunction()

    # Add embeddings to ChromaDB collection
    l2_collection = chroma_client.get_or_create_collection(name="l2_collection", metadata={"hnsw:space": "l2"})
    add_embedding(collection=l2_collection, files_path=files_path)
    ```

3. **Query and Retrieve Images**:

    ```python
    l2_results = search(image_path=test_path, collection=l2_collection, n_results=5)
    plot_results(image_path=test_path, files_path=files_path, results=l2_results)
    ```

## Project Structure

- `data/`: Contains the dataset organized into train and test directories.
- `app.py`: Main script to run the image retrieval system.
- `utils.py`: Utility functions for embedding generation, searching, and plotting results.

## Contributing

Feel free to submit issues or pull requests to improve the project.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

- [ChromaDB](https://www.trychroma.com/) for the vector database.
- [OpenCLIP](https://github.com/mlfoundations/open_clip) for the embedding function.
- [PIL](https://pillow.readthedocs.io/) and [Matplotlib](https://matplotlib.org/) for image processing and visualization.
