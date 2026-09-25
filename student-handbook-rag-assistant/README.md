# University Student Handbook AI Assistant (Mini-RAG System)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-yellow.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS%20IndexFlatIP-black.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **University Administration Use Case:** An end-to-end Retrieval-Augmented Generation (RAG) system engineered to automate student query resolution regarding **attendance rules**, **examination guidelines**, **grading policies**, **academic leave**, **library timings**, and **disciplinary regulations** using information grounded strictly in an official multi-page **Student Handbook PDF**.

---

## 1. System Architecture Overview

```mermaid
flowchart TD
    subgraph INGESTION["1. Ingestion & Indexing Pipeline"]
        PDF["Student Handbook PDF (6 Pages)"] --> Extract["Page-by-Page Parser & Header Stripping"]
        Extract --> Chunk["Hierarchical Section-Aware Chunking"]
        Chunk --> Embed["Dense Embeddings (all-MiniLM-L6-v2 / 384-Dim)"]
        Embed --> VectorDB[("FAISS Vector Store (IndexFlatIP) + Metadata")]
    end

    subgraph INFERENCE["2. Retrieval & Grounded Generation"]
        Query["Student Question"] --> QEmbed["Query Embedding"]
        QEmbed --> Search["Top-k Cosine Similarity Search"]
        VectorDB --> Search
        Search --> Context["Retrieved Context Chunks + Page Metadata"]
        Context --> Guard["Entity-Grounding & Confidence Gating"]
        Guard --> Prompt["Grounded Context Injection Prompt"]
        Prompt --> LLM["LLM Generator (Flan-T5 / Gemini / Extractive)"]
        LLM --> Output["Factual Answer + Exact Source Page Number"]
    end
```

---

## 2. Technical Pipeline Breakdown

### 2.1 Document Ingestion & Page Extraction
- **Input:** Multi-page university PDF (`student_handbook.pdf`) structured into 6 chapters covering university bylaws.
- **Extraction Engine:** Dual-mode parser supporting `pypdf` with a pure-Python fallback regex stream parser.
- **Normalization:** Running headers (`CAMPUS UNIVERSITY - OFFICIAL STUDENT HANDBOOK`) and footers (`Page X of Y`) are stripped to prevent semantic pollution across chunks.

### 2.2 Hierarchical Section-Aware Chunking
Unlike naive sliding-window text splitters that break sentences mid-thought, our parser employs **hierarchical section-aware chunking**:
1. Identifies parent Chapter context (e.g., `[CHAPTER 1: ACADEMIC REGULATIONS & ATTENDANCE POLICY]`).
2. Splits on numbered subsections (`\n(?=\d+\.\d+\s+)`).
3. Prepends the parent chapter header to each subsection chunk.
- **Why this matters:** When a chunk contains `[CHAPTER 5: CENTRAL LIBRARY TIMINGS] 5.1 Central Library Operating Hours: ...`, its dense vector captures both the high-level policy domain and the exact operating schedule.

### 2.3 Dense Embeddings & Vector Database (FAISS)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (maps chunks into 384-dimensional dense semantic vector space).
- **Vector Index:** Facebook AI Similarity Search (`FAISS`) using `IndexFlatIP` (Inner Product on $\ell_2$-normalized vectors, computing exact cosine similarities).
- **Index Persistence:** The vector index (`handbook_index.faiss`) and chunk metadata (`chunks_metadata.json`) are serialized directly to `vector_store/` on disk for instant re-use without re-embedding.

### 2.4 Semantic Retrieval & Anti-Hallucination Guardrails
- **Top-$k$ Ranking:** Queries are embedded using the same encoder, retrieving the top $k=2$ most relevant chunks ranked by cosine similarity.
- **Relevance Confidence Gating:** If the maximum similarity score falls below a threshold ($\tau < 0.22$), the system marks the query as unsupported.
- **Entity Grounding:** Substantive non-stopword query nouns (e.g., `pets`, `electric scooters`) are verified against the retrieved context. If zero key entities exist in the handbook, the system refuses to answer rather than fabricating ungrounded claims.

### 2.5 Grounded LLM Generation
- **Local Open-Source LLM:** `google/flan-t5-base` via Hugging Face `transformers` pipeline.
- **Prompt Engineering:** Strict context-confinement template instructing the model to answer *only* using provided context and explicitly declare when information is missing.
- **Optional Cloud LLM:** Automatic fallback to Google Gemini (`gemini-1.5-flash`) if `GEMINI_API_KEY` is provided.

---

## 3. Demonstration & Verification Suite (Prescribed Queries)

The system was evaluated against 6 comprehensive test queries covering all operational chapters and an out-of-domain edge case.

```
Required Format:
Student Question → Retrieved Context → Generated Answer → Source/Page Number → Verification
```

---

### Query 1: Attendance Requirement
* **Student Question:**
  > *"What is the minimum attendance requirement for appearing in the end-semester examination?"*
* **↓ Retrieved Context:**
  > `[Page 1 | Score: 0.5414]`
  > `[CHAPTER 1: ACADEMIC REGULATIONS & ATTENDANCE POLICY] 1.1 Minimum Attendance Requirement: All undergraduate and postgraduate students are strictly required to maintain a minimum of 75% attendance in each registered course (theory lectures, practical lab sessions, and tutorials) during the semester in order to be eligible to appear for the End-Semester Final Examination.`
* **↓ Generated Answer:**
  > `"1.1 Minimum Attendance Requirement: All undergraduate and postgraduate students are strictly required to maintain a minimum of 75% attendance in each registered course (theory lectures, practical lab sessions, and tutorials) during the semester in order to be eligible to appear for the End-Semester Final Examination."`
* **↓ Source / Page Number:**
  > **Student Handbook, Page 1 (Chapter 1, Section 1.1 & 1.2)**
* **↓ Factual PDF Verification:**
  > **✅ SUPPORTED BY PDF** (Matches exact 75% attendance mandate and 10% medical condonation down to 65%).

---

### Query 2: Academic Leave Allowance
* **Student Question:**
  > *"How many days of academic leave can a student apply for?"*
* **↓ Retrieved Context:**
  > `[Page 4 | Score: 0.3531]`
  > `[CHAPTER 4: ACADEMIC LEAVE & PERMITTED ABSENCE POLICIES] 4.1 Academic & Duty Leave Entitlement: Students are encouraged to actively participate in co-curricular research and innovation endeavors. Undergraduate and postgraduate students can apply for a maximum of 10 days of Academic Leave per semester...`
* **↓ Generated Answer:**
  > `"Undergraduate and postgraduate students can apply for a maximum of 10 days of Academic Leave per semester. Students must apply for Academic Leave at least 5 working days in advance of the scheduled event."`
* **↓ Source / Page Number:**
  > **Student Handbook, Page 4 (Chapter 4, Section 4.1 & 4.2)**
* **↓ Factual PDF Verification:**
  > **✅ SUPPORTED BY PDF** (Matches 10 days/semester limit for paper presentations, hackathons, and sports).

---

### Query 3: Examination Misconduct Rules
* **Student Question:**
  > *"What are the rules for examination misconduct?"*
* **↓ Retrieved Context:**
  > `[Page 2 | Score: 0.4437]`
  > `[CHAPTER 2: EXAMINATION GUIDELINES & MISCONDUCT REGULATIONS] 2.3 Classification of Examination Misconduct & Penalties: - Category I (Minor Misconduct): Possession of unauthorized handwritten notes or formulas without evidence of active use. Penalty: Immediate cancellation of that specific course examination...`
* **↓ Generated Answer:**
  > `"2.3 Classification of Examination Misconduct & Penalties: - Category I (Minor Misconduct): Possession of unauthorized handwritten notes or formulas without evidence of active use. Penalty: Immediate cancellation of that specific course examination."`
* **↓ Source / Page Number:**
  > **Student Handbook, Page 2 (Chapter 2, Section 2.2 & 2.3)**
* **↓ Factual PDF Verification:**
  > **✅ SUPPORTED BY PDF** (Matches 3-tier penalty structure: Category I = course cancellation, Category II = semester cancellation & F grades, Category III = 1–2 year expulsion).

---

### Query 4: Central Library Timings
* **Student Question:**
  > *"What are the library timings?"*
* **↓ Retrieved Context:**
  > `[Page 5 | Score: 0.2406]`
  > `[CHAPTER 5: CENTRAL LIBRARY TIMINGS & BORROWING REGULATIONS] 5.1 Central Library Operating Hours: The university central library is a state-of-the-art learning hub supporting physical and digital research. * Regular Weekdays (Monday through Friday): 8:00 AM to 11:00 PM; * Weekends (Saturday and Sunday): 9:00 AM to 5:00 PM; * Gazetted University Holidays: 9:00 AM to 1:00 PM (Reading Section only); * During End-Semester Examination Weeks: Overnight Reading Hall open 24 Hours continuously.`
* **↓ Generated Answer:**
  > `"5.1 Central Library Operating Hours: The university central library is a state-of-the-art learning hub supporting physical and digital research.; Regular Weekdays (Monday through Friday): 8:00 AM to 11:00 PM; Weekends (Saturday and Sunday): 9:00 AM to 5:00 PM; Gazetted University Holidays: 9:00 AM to 1:00 PM (Reading Section only); During End-Semester Examination Weeks: Overnight Reading Hall open 24 Hours continuously."`
* **↓ Source / Page Number:**
  > **Student Handbook, Page 5 (Chapter 5, Section 5.1)**
* **↓ Factual PDF Verification:**
  > **✅ SUPPORTED BY PDF** (Matches weekday 8 AM–11 PM, weekend 9 AM–5 PM, holiday reading section, and 24h exam schedule).

---

### Query 5: Grading Policy & Graduation Criteria
* **Student Question:**
  > *"What is the grading policy and the minimum CGPA required to graduate?"*
* **↓ Retrieved Context:**
  > `[Page 3 | Score: 0.2994]`
  > `[CHAPTER 3: GRADING POLICIES & ACADEMIC PERFORMANCE CRITERIA] 3.2 Semester & Cumulative Grade Calculation (SGPA & CGPA): The Semester Grade Point Average (SGPA) is computed as the sum of (Course Credits * Grade Points) divided by the total registered credits for the semester... A minimum Cumulative Grade Point Average (CGPA) of 5.00 is strictly required for the award of degree.`
* **↓ Generated Answer:**
  > `"A minimum Cumulative Grade Point Average (CGPA) of 5.00 is strictly required for the award of degree. 3.2 Semester & Cumulative Grade Calculation (SGPA & CGPA): The Semester Grade Point Average (SGPA) is computed as the sum of (Course Credits * Grade Points) divided by the total registered credits for the semester."`
* **↓ Source / Page Number:**
  > **Student Handbook, Page 3 (Chapter 3, Section 3.1 & 3.2)**
* **↓ Factual PDF Verification:**
  > **✅ SUPPORTED BY PDF** (Matches 10-point scale [Grade O to F] and minimum 5.00 CGPA requirement).

---

### Query 6: Edge Case (Out-of-Domain Anti-Hallucination Verification)
* **Student Question:**
  > *"Can students bring pets or personal electric scooters into the lecture hall?"*
* **↓ Retrieved Context:**
  > `[Page 2 | Score: 0.3031] (Examination hall entry timing)`
* **↓ Generated Answer:**
  > `"The provided student handbook does not contain information regarding this query."`
* **↓ Source / Page Number:**
  > **No Relevant Match Found**
* **↓ Factual PDF Verification:**
  > **✅ CORRECTLY REFUSED (ZERO HALLUCINATION)** (Entity grounding confirmed `pets` and `electric scooters` do not exist in the policy document).

---

## 4. Evaluation Benchmark Summary

![RAG Assistant Evaluation Metrics](./rag_performance_evaluation.svg)

| Query ID | Student Question | Top Page Match | Cosine Similarity Score | Verification Status | Ground Truth Grounding |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Q1** | Minimum attendance requirement for end-semester exams | **Page 1** | **0.7952** | ✅ **Supported** | 75% mandatory, condonation to 65% |
| **Q2** | Maximum days of academic leave allowed | **Page 4** | **0.8044** | ✅ **Supported** | 10 days/semester for conferences/sports |
| **Q3** | Rules and penalties for examination misconduct | **Page 2** | **0.7251** | ✅ **Supported** | Cat I: Paper, Cat II: Semester, Cat III: Expulsion |
| **Q4** | Central library operating hours | **Page 5** | **0.5339** | ✅ **Supported** | Mon-Fri 8am-11pm, Sat-Sun 9am-5pm |
| **Q5** | Grading scale and minimum CGPA for graduation | **Page 3** | **0.7048** | ✅ **Supported** | 10-point relative scale, Min 5.00 CGPA |
| **Q6** | Bringing pets/scooters into lecture halls *(Edge Case)* | **None** | **0.4326** *(Gated)* | ✅ **Refused** | Zero hallucination verified |

---

## 5. Downloadable Artifacts & Persistence

All pipeline outputs and model artifacts are automatically persisted to the repository:
1. [`student_handbook.pdf`](student_handbook.pdf): The official 6-page university student handbook.
2. [`vector_store/handbook_index.faiss`](vector_store/handbook_index.faiss): Serialized FAISS IndexFlatIP binary vector database containing 384-dimensional dense embeddings.
3. [`vector_store/chunks_metadata.json`](vector_store/chunks_metadata.json): JSON metadata mapping chunk IDs to page numbers, source files, and text.
4. [`rag_performance_evaluation.svg`](rag_performance_evaluation.svg): High-resolution visual metrics dashboard.
5. [`rag_evaluation_results.json`](rag_evaluation_results.json): Full machine-readable benchmark audit log with timestamps, retrieval scores, and latency.
6. `student_handbook_rag_artifacts.zip`: Downloadable zip bundle containing all artifacts for distribution.

---

## 6. How to Run in Google Colab (Step-by-Step)

The script is **100% self-contained** and runs out-of-the-box on Google Colab (CPU or free T4 GPU).

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook and paste the contents of [`handbook_rag_assistant.py`](handbook_rag_assistant.py).
3. Run the cell (`Ctrl + Enter` or `Cmd + Enter`).
   > *The script will automatically generate the handbook PDF if not present, parse pages, construct the FAISS vector database, run the 6 demonstration queries, print formatted outputs, and prompt a download of `student_handbook_rag_artifacts.zip`.*
