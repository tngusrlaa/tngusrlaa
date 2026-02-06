# Tool README

## Overview

This tool is designed to detect generative AI images by automatically extracting file-level metadata and encoding-related structural features from image files. It supports the generation of machine learning–ready datasets, as well as the complete workflow of model training and inference. Without relying on any visual or pixel-level analysis, the tool utilizes format-level characteristics—including file structure, segments, chunks, and initial byte sequences—to determine whether an image is AI-generated and to identify the originating generative service.

This document provides detailed guidance on the prerequisites, installation, and execution of the tool, covering both the pre-built executable and source-code-based usage. It also includes instructions for dataset generation, model training, and result inspection, along with descriptions of key options and configurations. The theoretical foundation and performance evaluation of the proposed approach are based on the experimental results presented in the referenced paper ([URL]).
## Prerequisites

### 1. For Modifying or Compiling the Source Code

- Ensure that **Python 3.8 or higher** is installed on your system.
- Install the necessary dependencies listed in the `requirements.txt` file.

## Installation

### 1. Setting up Python Environment

1. [Download Python](https://www.python.org/downloads/) and install
2. Verify the installation by running:
   ```bash
   python --version
   ```

### 2. Installing Dependencies

1. Navigate to the directory containing the tool's source code.
2. Run the following command to install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Running the Program

### 1. Running from Source Code

1. **Open a terminal or command prompt.**
2. Navigate to the directory where the source code resides.
3. Run the program with the following command:
   ```bash
   python #1-6 AIorNot.py
   ```

## Additional Notes

- For detailed error logging, refer to the `logs` folder generated during execution.
- If you encounter any issues, ensure all dependencies are installed correctly and that your Python version is compatible.

## Code Structure

Below is the hierarchical structure of the main code and its related components:

```plaintext
#1-6 AIorNot (Main Code)
├── Runtime / Case Management
│   └── Manages runtime environment, paths, and execution settings
│
├── Training Data Preparation
│   ├── Data loading & target definition
│   ├── Feature pruning and preprocessing
│   └── Missing value handling and encoding
│
├── Model Training
│   ├── Candidate model definitions
│   ├── Hyperparameter tuning (cross-validation)
│   └── Binary classification training
│
├── Evaluation & Detection
│   ├── Validation and test evaluation
│   ├── Statistical distribution checks
│   ├── Error analysis
│   └── Visualization and diagnostics
│
└── Results Aggregation
    └── Model comparison and best model selection

```
## Additional Information & Resources

Additional information and resources related to this project and dataset are available at the following links:
https://rural-crib-445.notion.site/Generative-AI-Service-Identification-via-Image-Metastructure-Analysis-2efa81c0160a80b0a261d35cfc786286

- Detection Code (GitHub) : Automated code for detecting generative AI images by extracting file-level metadata and encoding-related structural features.

- Dataset (Kaggle) : Access to the complete image dataset used in the experiments, hosted on Kaggle.

- Text-to-Image Prompts List : The full list of 100 distinct scene description prompts used to generate the images.

- Hardware Specifications
Detailed information on the computing environments used for image generation, dataset construction, and model training.

## Contact

For any questions or issues, please contact the developer at [blind].

## ETC
The initial concept and planning of this research were undertaken as part of a collaborative effort with the (blind). 
