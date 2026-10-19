# Applied Generative AI & Deep Learning Portfolio 🚀

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A curated collection of practical implementations, technical reports, and portfolio-grade architectures for **Generative AI & Deep Learning**. 

Every project in this repository is built to be:
- **100% Free-Tier Compatible:** Runs seamlessly on Google Colab (CPU or free T4 GPU) without paid API keys.
- **Self-Contained & Reproducible:** Includes automated synthetic mock datasets for zero-friction initial testing, alongside production pipelines for custom data.
- **Thoroughly Documented:** Each module contains a comprehensive, publication-style technical report and architectural analysis.

---

## 📚 Repository Index

| Topic / Implementation | Focus Area | Frameworks / Tools | Status | Documentation & Code |
| :--- | :--- | :--- | :---: | :---: |
| **Smart Waste Classification Using Deep Neural Network** | Computer Vision, Custom Dataset Pipeline, Multi-class Classification, DNN Regularization | TensorFlow / Keras, Scikit-Learn, Matplotlib | ✅ Completed | [Explore Module](./smart-waste-classification-dnn/) |
| **University Student Handbook Mini-RAG Assistant** | Retrieval-Augmented Generation (RAG), Semantic Vector Search, Document Grounding, Anti-Hallucination | Sentence-Transformers, FAISS, Flan-T5, PyPDF | ✅ Completed | [Explore Module](./student-handbook-rag-assistant/) |
| **LLM Fine-Tuning with LoRA on Custom QA Dataset** | Parameter-Efficient Fine-Tuning (PEFT), Instruction Tuning, Tokenization Masking, Model Comparison | PyTorch, Hugging Face Transformers, PEFT, LoRA | ✅ Completed | [Explore Module](./llm-finetuning-lora/) |
| **ANN Binary Classification with Back-Propagation** | Forward & Backward Propagation, Chain Rule Calculus, Binary Cross-Entropy, Decision Boundaries | Python, NumPy, Scikit-Learn, Matplotlib | ✅ Completed | [Explore Module](./ann-binary-classification-backprop/) |
| **ANN Multi-Class Classification with Back-Propagation** | Multi-Class Vectorized Backpropagation, Softmax Jacobian, Categorical Cross-Entropy, Adam | Python, NumPy, Scikit-Learn, Matplotlib | ✅ Completed | [Explore Module](./ann-multiclass-classification-backprop/) |
| **CNN Image Classification with Hyperparameter Tuning** | Deep CNN Architecture, Batch Normalization, Hyperparameter Search, Fashion-MNIST, PyTorch | PyTorch, Torchvision, NumPy, Matplotlib | ✅ Completed | [Explore Module](./cnn-image-classification-hyperparameter-tuning/) |
| **Pre-trained CNN Architectures Benchmark** | Transfer Learning, AlexNet, VGG-16, InceptionNet, ResNet-50, Inference Latency, Top-1/Top-5 | PyTorch, Torchvision, NumPy, Matplotlib | ✅ Completed | [Explore Module](./pretrained-cnn-models-comparison/) |
| **Autoencoder for Semantic Image Segmentation** | Encoder-Decoder Network, Transposed Convolutions, Soft Dice + BCE Loss, mIoU / Dice Evaluation | PyTorch, NumPy, Matplotlib | ✅ Completed | [Explore Module](./autoencoder-image-segmentation/) |

---

## 🗂️ Project Structure

```text
applied-generative-ai-labs/
├── README.md                                    # Master repository index
├── .gitignore                                   # Standard deep learning gitignore
├── smart-waste-classification-dnn/              # Smart Waste Classification Using DNN
│   ├── README.md                                # Comprehensive technical report & metrics
│   └── waste_classification_dnn.py              # Colab-ready self-contained Python script
├── student-handbook-rag-assistant/              # Student Handbook Mini-RAG Assistant
│   ├── README.md                                # Comprehensive architectural report & verification
│   ├── handbook_rag_assistant.py                # Self-contained end-to-end RAG script
│   ├── student_handbook.pdf                     # Official 6-page University Handbook PDF
│   ├── rag_evaluation_results.json              # Machine-readable benchmark audit results
│   ├── rag_performance_evaluation.svg          # Visual metrics dashboard
│   └── vector_store/                            # Serialized vector embeddings & metadata
├── llm-finetuning-lora/                         # LLM Fine-Tuning with LoRA
│   ├── README.md                                # Technical report & side-by-side comparison
│   ├── train_lora_qa.py                         # Self-contained PEFT fine-tuning script
│   ├── custom_qa_dataset.json                   # 25 curated AI engineering QA pairs
│   ├── lora_evaluation_comparison.json          # Colab evaluation side-by-side audit
│   ├── training_loss_curve.svg                  # Step-by-step training loss curve (vector)
│   ├── training_loss_curve.png                  # Training loss trajectory chart
│   ├── lora_qa_adapter_artifacts.zip            # Downloadable adapter weights & artifacts bundle
│   └── lora_qa_adapter/                         # Trained adapter weights (safetensors)
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── ann-binary-classification-backprop/          # ANN Binary Classification with Backpropagation
│   ├── README.md                                # Mathematical derivation & evaluation report
│   ├── ann_binary_classification.py             # Vectorized NumPy MLP from scratch
│   └── evaluation_metrics.json                  # Machine-readable test evaluation metrics
├── ann-multiclass-classification-backprop/      # ANN Multi-Class Classification with Backpropagation
│   ├── README.md                                # Mathematical derivation & evaluation report
│   ├── ann_multiclass_classification.py         # Vectorized NumPy Multi-Class MLP from scratch
│   └── evaluation_metrics.json                  # Machine-readable test evaluation metrics
├── cnn-image-classification-hyperparameter-tuning/ # CNN Image Classification with Hyperparameter Tuning
│   ├── README.md                                # Comprehensive technical report & hyperparameter analysis
│   ├── cnn_hyperparameter_tuning.py             # PyTorch CNN architecture & tuning pipeline
│   ├── hyperparameter_tuning_results.json       # Machine-readable tuning trials & evaluation metrics
│   └── tuning_comparison.png                    # Empirical validation curves & confusion matrix
├── pretrained-cnn-models-comparison/            # Pre-trained CNN Models Comparison
│   ├── README.md                                # Comprehensive comparative report & trade-off analysis
│   ├── pretrained_cnn_comparison.py             # PyTorch benchmark script (AlexNet, VGG, Inception, ResNet)
│   └── pretrained_models_comparison.json        # Machine-readable architectural benchmark metrics
└── autoencoder-image-segmentation/              # Autoencoder for Semantic Image Segmentation
    ├── README.md                                # Architectural report, deconvolution math & metrics
    ├── autoencoder_segmentation.py              # PyTorch Encoder-Decoder segmentation pipeline
    └── segmentation_evaluation_results.json     # Machine-readable mIoU, Dice, and accuracy metrics
```

---

## ⚡ Google Colab Quickstart Workflow

All implementations are designed for frictionless copy-paste execution in Google Colab:
1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Select **Runtime > Change runtime type** (free T4 GPU recommended, though standard CPU is also supported).
3. Copy the Python code from the respective module's `.py` file and paste it into a cell.
4. Run the cell. Built-in sample dataset generators allow immediate end-to-end execution, or you can plug in your own custom datasets.

---

## 📜 License
This repository is open-sourced under the [MIT License](LICENSE).
