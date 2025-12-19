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

## Environment Setup

To set up the environment for this project, follow these steps:

1. **Install Conda**:
   - If you don't have Conda installed, download and install it from [Conda's official website](https://docs.conda.io/en/latest/miniconda.html).

2. **Create the Environment**:
   - Use the provided `environment.yml` file to create the environment:
     ```bash
     conda env create -f environment.yml
     ```

3. **Activate the Environment**:
   - Activate the newly created environment:
     ```bash
     conda activate ir_project
     ```

4. **Install Additional Dependencies** (if needed):
   - If you encounter missing dependencies, install them using pip or conda. For example:
     ```bash
     pip install <package_name>
     ```

## Project Structure

- `data/`: Contains the SWIM-IR dataset.
- `project.ipynb`: Jupyter notebook for data exploration and model implementation.
- `environment.yml`: Conda environment configuration file.
- `README.md`: Project documentation.

## Running the Project

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook project.ipynb
   ```

2. Follow the steps in the notebook to load the dataset, preprocess the data, and implement the retrieval model.

## Authors
- Delia Mennitti - 19610
- Letizia Meroi
- Sara Napolitano

## References
- [SWIM-IR Dataset](https://github.com/google-research-datasets/swim-ir)
- Paper: Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval by Nandan Thakur, Jianmo Ni, Gustavo Hernández Ábrego, John Wieting, Jimmy Lin, and Daniel Cer.



