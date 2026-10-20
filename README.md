# Applied Generative AI & Deep Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Practical implementations, experimental benchmarks, and technical reports for deep learning and generative AI architectures. Each module is self-contained with reproducible training scripts, evaluation metrics, and documentation.

---

## Repository Index

| Topic / Implementation | Focus Area | Frameworks / Tools | Status | Documentation & Code |
| :--- | :--- | :--- | :---: | :---: |
| **Smart Waste Classification Using Deep Neural Network** | Computer Vision, Custom Dataset Pipeline, Multi-class Classification, DNN Regularization | TensorFlow / Keras, Scikit-Learn, Matplotlib | Completed | [Explore Module](./smart-waste-classification-dnn/) |
| **University Student Handbook Mini-RAG Assistant** | Retrieval-Augmented Generation (RAG), Semantic Vector Search, Document Grounding, Anti-Hallucination | Sentence-Transformers, FAISS, Flan-T5, PyPDF | Completed | [Explore Module](./student-handbook-rag-assistant/) |
| **LLM Fine-Tuning with LoRA on Custom QA Dataset** | Parameter-Efficient Fine-Tuning (PEFT), Instruction Tuning, Tokenization Masking, Model Comparison | PyTorch, Hugging Face Transformers, PEFT, LoRA | Completed | [Explore Module](./llm-finetuning-lora/) |
| **ANN Binary Classification with Back-Propagation** | Forward & Backward Propagation, Chain Rule Calculus, Binary Cross-Entropy, Decision Boundaries | Python, NumPy, Scikit-Learn, Matplotlib | Completed | [Explore Module](./ann-binary-classification-backprop/) |
| **ANN Multi-Class Classification with Back-Propagation** | Multi-Class Vectorized Backpropagation, Softmax Jacobian, Categorical Cross-Entropy, Adam | Python, NumPy, Scikit-Learn, Matplotlib | Completed | [Explore Module](./ann-multiclass-classification-backprop/) |
| **CNN Image Classification with Hyperparameter Tuning** | Deep CNN Architecture, Batch Normalization, Hyperparameter Search, Fashion-MNIST, PyTorch | PyTorch, Torchvision, NumPy, Matplotlib | Completed | [Explore Module](./cnn-image-classification-hyperparameter-tuning/) |
| **Pre-trained CNN Architectures Benchmark** | Transfer Learning, AlexNet, VGG-16, InceptionNet, ResNet-50, Inference Latency, Top-1/Top-5 | PyTorch, Torchvision, NumPy, Matplotlib | Completed | [Explore Module](./pretrained-cnn-models-comparison/) |
| **Autoencoder for Semantic Image Segmentation** | Encoder-Decoder Network, Transposed Convolutions, Soft Dice + BCE Loss, mIoU / Dice Evaluation | PyTorch, NumPy, Matplotlib | Completed | [Explore Module](./autoencoder-image-segmentation/) |
| **GAN for Handwritten Digit Recognition** | Minimax Game, AC-GAN, Multi-Class Recognition Head, Class-Conditioned Generation, MNIST | PyTorch, Torchvision, NumPy, Matplotlib | Completed | [Explore Module](./gan-mnist-digit-recognition/) |
| **Pix2Pix GAN for Image-to-Image Domain Translation** | Conditional GAN, U-Net Skip Connections, 70x70 PatchGAN, L1 Reconstruction Loss, Sketch-to-Photo | PyTorch, Torchvision, NumPy, Matplotlib | Completed | [Explore Module](./pix2pix-image-to-image-translation/) |
| **RNN for Music Composition Generation** | Stacked 2-Layer LSTM, Embedding Layer, Cross-Entropy & Perplexity, Temperature Sampling, MIDI Synthesis | PyTorch, NumPy, Matplotlib | Completed | [Explore Module](./rnn-music-generation/) |
| **Vision Transformer (ViT) for Image Classification** | Patch Embedding (P=4), [CLS] Token, Multi-Head Self-Attention, GELU MLP, CIFAR-10 | PyTorch, Torchvision, NumPy, Matplotlib | Completed | [Explore Module](./vision-transformer-image-classification/) |
| **Convolutional Vision Transformer (CvT)** | Conv Tokenizer (32->8), Depthwise ConvFFN, Multi-Head Self-Attention, Global Pooling, CIFAR-10 | PyTorch, Torchvision, NumPy, Matplotlib | Completed | [Explore Module](./convolutional-vision-transformer/) |

---

## Project Structure

```text
applied-generative-ai-labs/
├── README.md                                    # Master repository index
├── .gitignore                                   # Standard deep learning gitignore
├── smart-waste-classification-dnn/              # Smart Waste Classification Using DNN
│   ├── README.md                                # Technical report & metrics
│   └── waste_classification_dnn.py              # Self-contained classification script
├── student-handbook-rag-assistant/              # Student Handbook Mini-RAG Assistant
│   ├── README.md                                # Architectural report & test evaluation
│   ├── handbook_rag_assistant.py                # End-to-end RAG script
│   ├── student_handbook.pdf                     # University Handbook document
│   ├── rag_evaluation_results.json              # Benchmark evaluation results
│   ├── rag_performance_evaluation.svg          # Metrics visualization dashboard
│   └── vector_store/                            # Serialized vector embeddings & metadata
├── llm-finetuning-lora/                         # LLM Fine-Tuning with LoRA
│   ├── README.md                                # Technical report & comparison
│   ├── train_lora_qa.py                         # PEFT fine-tuning script
│   ├── custom_qa_dataset.json                   # 25 curated AI engineering QA pairs
│   ├── lora_evaluation_comparison.json          # Evaluation audit metrics
│   ├── training_loss_curve.svg                  # Step-by-step training loss curve (vector)
│   ├── training_loss_curve.png                  # Training loss trajectory chart
│   ├── lora_qa_adapter_artifacts.zip            # Downloadable adapter weights bundle
│   └── lora_qa_adapter/                         # Trained adapter weights
├── ann-binary-classification-backprop/          # ANN Binary Classification with Backpropagation
│   ├── README.md                                # Mathematical derivation & evaluation report
│   ├── ann_binary_classification.py             # Vectorized NumPy MLP from scratch
│   └── evaluation_metrics.json                  # Test evaluation metrics
├── ann-multiclass-classification-backprop/      # ANN Multi-Class Classification with Backpropagation
│   ├── README.md                                # Mathematical derivation & evaluation report
│   ├── ann_multiclass_classification.py         # Vectorized NumPy Multi-Class MLP from scratch
│   └── evaluation_metrics.json                  # Test evaluation metrics
├── cnn-image-classification-hyperparameter-tuning/ # CNN Image Classification with Hyperparameter Tuning
│   ├── README.md                                # Technical report & hyperparameter analysis
│   ├── cnn_hyperparameter_tuning.py             # PyTorch CNN tuning pipeline
│   ├── hyperparameter_tuning_results.json       # Tuning trials & evaluation metrics
│   └── tuning_comparison.png                    # Validation curves & confusion matrix
├── pretrained-cnn-models-comparison/            # Pre-trained CNN Models Comparison
│   ├── README.md                                # Comparative report & trade-off analysis
│   ├── pretrained_cnn_comparison.py             # PyTorch benchmark script (AlexNet, VGG, Inception, ResNet)
│   └── pretrained_models_comparison.json        # Benchmark metrics
├── autoencoder-image-segmentation/              # Autoencoder for Semantic Image Segmentation
│   ├── README.md                                # Architectural report & metrics
│   ├── autoencoder_segmentation.py              # PyTorch Encoder-Decoder segmentation pipeline
│   ├── segmentation_evaluation_results.json     # mIoU, Dice, and accuracy metrics
│   └── segmentation_results.png                 # Visual overlays & learning curves
├── gan-mnist-digit-recognition/                 # GAN for Handwritten Digit Recognition
│   ├── README.md                                # Minimax formulation, AC-GAN architecture & metrics
│   ├── gan_mnist_recognition.py                 # PyTorch AC-GAN pipeline
│   └── gan_mnist_evaluation_results.json        # Test accuracy & loss metrics
├── pix2pix-image-to-image-translation/          # Pix2Pix Image-to-Image Domain Translation
│   ├── README.md                                # Technical report & PatchGAN math
│   ├── pix2pix_translation.py                   # PyTorch U-Net & PatchGAN translation script
│   └── translation_evaluation_results.json      # Test metrics and training history
├── rnn-music-generation/                        # Recurrent Neural Network for Music Generation
│   ├── README.md                                # Mathematical formulation, LSTM gates & metrics
│   ├── music_generation_rnn.py                  # PyTorch Stacked LSTM & MIDI synthesis script
│   └── music_generation_results.json           # Perplexity, loss, and generated compositions
├── vision-transformer-image-classification/     # Vision Transformer for Image Classification
│   ├── README.md                                # Technical formulation & self-attention math
│   ├── vision_transformer_classification.py     # PyTorch ViT architecture & pipeline script
│   └── vit_classification_results.json         # Evaluation metrics & history
└── convolutional-vision-transformer/            # Convolutional Vision Transformer
    ├── README.md                                # Technical report, ConvFFN math & metrics
    ├── convolutional_vision_transformer.py      # PyTorch CvT architecture & pipeline script
    └── cvt_classification_results.json         # Evaluation metrics & history
```

---

## Running in Google Colab

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Select **Runtime > Change runtime type** (free T4 GPU recommended).
3. Copy the Python code from the respective module's `.py` file into a cell.
4. Run the cell to execute training, evaluation, and visualization.

---

## License
This repository is open-sourced under the [MIT License](LICENSE).
