# Information Retrieval Project

This project focuses on cross-lingual Information Retrieval using the SWIM-IR dataset. The goal is to retrieve relevant Wikipedia passages written in different languages based on English queries. Each query is associated with exactly one relevant passage, enabling automatic and reproducible evaluation.

## Dataset

1. **Download the Dataset**:
   - The dataset can be downloaded from the [SWIM-IR GitHub repository](https://github.com/google-research-datasets/swim-ir?tab=readme-ov-file#download).
   - Place the downloaded data inside a folder named `data` in the project root directory.

2. **Extract the Dataset**:
   - If the dataset is compressed (e.g., `.tar.gz`), extract it using the following Python code:
     ```python
     import tarfile

     tar_path = "data/swim_ir_v1.tar.gz"
     extract_path = "data/swim_ir_v1"

     with tarfile.open(tar_path, "r:gz") as tar:
         tar.extractall(path=extract_path)
     ```


## Project Structure

- `data/`: Contains the SWIM-IR dataset.
- `project.ipynb`: Jupyter notebook for data exploration and model implementation.
- `interface.py`: Web interface for querying the retrieval system.
- `environment.yml`: Conda environment configuration file.
- `artifacts_cache_bm25/`: Cached BM25 scores from notebook evaluation.
- `artifacts_cache_dense/`: Cached dense retrieval scores from notebook evaluation.
- `df_results_frozen_*.pkl`: Saved evaluation results with tuned hyperparameters.
- `README.md`: Project documentation.

## Running the Project

### 1. Jupyter Notebook (for evaluation and analysis)

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook project.ipynb
    ```bash

### 2. Web Interface (for querying)

1. make sure the right env is activated 
   ```bash
   conda activate ir_project
   ```
2. Install required dependencies for the web interface:
   ```bash
   pip install sentence-transformers rank_bm25 faiss-cpu gradio pandas jieba
   conda install numpy=1.26.4 scipy scikit-learn
   ```
3. Launch the web interface:
   ```bash
   python interface.py
   ```

The interface will initialize (this takes 2-5 minutes on first run):

Loading LaBSE model
Loading 1000 documents per language
Building BM25 indices
Loading/computing document embeddings
Once ready, you'll see:

* Running on local URL:  .... 

THE FIRST SEARCH WILL TAKE A LOT BUT THEN ITS PREATTY QUICK

## Authors
- Delia Mennitti - 19610
- Letizia Meroi - 19041
- Sara Napolitano - 24656

## References
- [SWIM-IR Dataset](https://github.com/google-research-datasets/swim-ir)
- Paper: Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval by Nandan Thakur, Jianmo Ni, Gustavo Hernández Ábrego, John Wieting, Jimmy Lin, and Daniel Cer.



