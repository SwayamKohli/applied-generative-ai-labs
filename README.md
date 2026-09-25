# Applied Generative AI & Deep Learning Labs 🚀

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A curated collection of practical assignments, laboratory experiments, and portfolio-grade implementations for **Generative AI & Deep Learning**. 

Every experiment in this repository is built to be:
- **100% Free-Tier Compatible:** Runs seamlessly on Google Colab (CPU or free T4 GPU) without paid API keys.
- **Self-Contained & Reproducible:** Includes automated synthetic mock datasets for zero-friction initial testing, alongside production pipelines for custom data.
- **Thoroughly Documented:** Each module contains a comprehensive, publication-style technical report and architectural analysis.

---

## 📚 Repository Index

| Module / Practical | Focus Area | Frameworks / Tools | Status | Documentation & Code |
| :--- | :--- | :--- | :---: | :---: |
| **Smart Waste Classification Using Deep Neural Network** | Computer Vision, Custom Dataset Pipeline, Multi-class Classification, DNN Regularization | TensorFlow / Keras, Scikit-Learn, Matplotlib | ✅ Completed | [Explore Module](./smart-waste-classification-dnn/) |
| **University Student Handbook Mini-RAG Assistant** | Retrieval-Augmented Generation (RAG), Semantic Vector Search, Document Grounding, Anti-Hallucination | Sentence-Transformers, FAISS, Flan-T5, PyPDF | ✅ Completed | [Explore Module](./student-handbook-rag-assistant/) |

---

## 🗂️ Project Structure

```text
applied-generative-ai-labs/
├── README.md                                    # Master repository index
├── .gitignore                                   # Standard deep learning gitignore
├── smart-waste-classification-dnn/              # Lab 1: Smart Waste Classification
│   ├── README.md                                # Comprehensive 1-page report & theory
│   └── waste_classification_dnn.py              # Colab-ready self-contained Python script
└── student-handbook-rag-assistant/              # Lab 2: Student Handbook Mini-RAG
    ├── README.md                                # Comprehensive architectural report & verification
    ├── handbook_rag_assistant.py                # Self-contained end-to-end RAG script
    ├── student_handbook.pdf                     # Official 6-page University Handbook PDF
    ├── rag_evaluation_results.json              # Machine-readable benchmark audit results
    └── vector_store/                            # Serialized vector embeddings & metadata
```

---

## ⚡ Google Colab Quickstart Workflow

All labs are designed for frictionless copy-paste execution in Google Colab:
1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Select **Runtime > Change runtime type** (free T4 GPU recommended, though standard CPU is also supported).
3. Copy the Python code from the respective module's `.py` file and paste it into a cell.
4. Run the cell. Built-in sample dataset generators allow immediate end-to-end execution, or you can plug in your own custom datasets.

---

## 📜 License
This repository is open-sourced under the [MIT License](LICENSE).
