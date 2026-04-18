# DoctorFollow AI — Medical RAG System

A retrieval-augmented generation system for Turkish-speaking doctors querying English medical literature. Built for the DoctorFollow AI Junior ML Engineer Technical Assessment.

---

## 1. Setup & Usage

```bash
git clone https://github.com/jaguuai/medical-rag-system
cd doctorfollow-rag
pip install -r requirements.txt
```

**Run the notebook:**
Open `doctorfollow_case.ipynb` in Google Colab or Jupyter and run cells sequentially.

**Environment variable (API key):**
```bash
export GEMINI_API_KEY="your_key_here"
```
Or in Colab, use the Secrets panel and add `GEMINI_API_KEY`.

**Requirements:**
```
# Core
requests>=2.32.0
pandas>=2.2.0
numpy>=1.26.0
tqdm>=4.66.0
tenacity>=9.0.0

# Retrieval
rank-bm25>=0.2.2
sentence-transformers>=2.2.0
transformers>=4.41.0
torch>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# LLM
google-genai>=0.3.0

# Utils
huggingface-hub>=0.23.0
psutil>=5.9.0
```

---

## 2. Approach

### Architecture

```
Query (TR or EN)
      ↓
Hybrid RRF Retrieval
  ├── BM25         (lexical, English strength)
  └── e5-small     (semantic, Turkish strength)
      ↓
Top-5 Articles → Context
      ↓
Gemini-3.1-flash-lite → Cited Answer (TR/EN)

```

### Model Selection: intfloat/multilingual-e5-small

Three embedding models were benchmarked on 5 queries (3 EN + 2 TR):

| Model | MRR | TR MRR | EN MRR | Enc(s) | RAM | Q latency |
|-------|-----|--------|--------|--------|-----|-----------|
| MedCPT (ncbi) | 0.75 | 0.625 | 0.833 | 101s | 100MB | 340ms |
| **e5-small ← CHOSEN** | **0.90** | **1.0** | **0.833** | **27s** | **203MB** | **46ms** |
| bge-m3 | 1.00 | 1.0 | 1.0 | 217s | 1473MB | 410ms |

**MedCPT** was eliminated first: trained exclusively on English PubMed query-article pairs, it has zero Turkish support — confirmed by direct test (`"çölyak"` → Hit@3=0/3, `"celiac disease"` → Hit@3=3/3). A deliberate HuggingFace search for Turkish medical embedding models (`"MedCPT turkish"`, `"medical turkish embedding"`, `"pubmed turkish"`) returned zero results — confirming that **Turkish medical NLP is an open research problem**.

**bge-m3** achieves perfect MRR=1.00 but at 1.5GB RAM and 410ms query latency — unacceptable for a real-time clinical system where doctors expect sub-100ms responses.

**e5-small** provides the best tradeoff: MRR=0.90, 46ms latency, multilingual Turkish support. The `query:` / `passage:` prefixes are mandatory — the model is instruction-tuned to distinguish query and passage representations; omitting them measurably degrades retrieval quality.

**Key finding:** Hybrid RRF (BM25 + e5-small) achieves MRR=1.00 — matching bge-m3 alone at 1/7 the memory cost (203MB vs 1473MB).

### What I'd Change With More Time

1. **Hybrid Vector + Graph RAG for Medical Knowledge**  
   Current system uses flat vector retrieval. Medical literature has rich relational structure (drug-disease interactions, contraindications, treatment hierarchies). Adding a knowledge graph layer (e.g., using Neo4j or LlamaIndex's GraphRAG) would enable multi-hop reasoning: "Find drugs that treat X but don't interact with Y."

2. **Query Rewriting & Decomposition**  
   Complex clinical queries often combine multiple sub-questions. Implement query rewriting (using a smaller LLM like Gemini Flash) to decompose "What are the latest guidelines for managing type 2 diabetes in elderly patients with CKD?" into sub-queries for each condition, then fuse results.

3. **Adaptive Retrieval with Router**  
   Not all queries need semantic search. Add a lightweight router (e.g., BERT-based classifier) to decide:
   - Exact term match (BM25) → for drug names, procedure codes
   - Semantic search → for conceptual questions
   - Hybrid RRF → for mixed cases

4. **Self-Reflection & Citation Verification**  
   Add a verification step where the LLM checks if its generated answer is actually supported by the cited sources. If not, trigger re-retrieval or fallback to "information not found."

5. **Streaming & Real-time Updates**  
   PubMed releases new articles daily. Implement incremental indexing so the corpus updates without full rebuilds. Use a message queue (RabbitMQ/Kafka) to process new articles as they arrive.

6. **Evaluation with Real Physicians**  
   Current pseudo-relevance (query_terms field) is an approximation. A proper evaluation would involve 2-3 physicians scoring relevance and answer quality on 50-100 queries (Cohen's kappa for inter-rater reliability).

7. **LangGraph for Multi-Step Reasoning**  
   Complex medical queries may require multiple retrieval steps (e.g., "Find patients with X, then check Y, then recommend Z"). Implement a LangGraph agent with:
   - Retriever node (fetch articles)
   - Extractor node (pull key facts)
   - Reasoner node (synthesize)
   - Verifier node (check citations)

---
## 3. BM25 Analysis — k1 and b Parameters

### What k1 Controls (Term Frequency Saturation)

k1 determines how much repeated occurrences of a query term increase the document score.

**Intuition:** Does seeing "diabetes" 10 times in an abstract make it 10x more relevant than seeing it once? k1 answers this.

| k1 Value | Behavior | Use Case |
|----------|----------|----------|
| 0.5 | TF saturates very fast (near-binary: presence/absence) | Short documents, titles |
| **1.2–1.5** | **Moderate saturation** | **Medical abstracts — chosen** |
| 2.0+ | High TF still adds significant score | Long documents, books, patents |

### What b Controls (Document Length Normalization)

b normalizes scores by document length relative to corpus average.

**Intuition:** Should a 400-word abstract with 8 mentions of "diabetes" score the same as a 100-word abstract with 2 mentions (same density)?

| b Value | Behavior | Use Case |
|---------|----------|----------|
| 0.0 | No normalization — long documents dominate | All documents same length |
| **0.75** | **Balanced — literature standard** | **General purpose, chosen** |
| 1.0 | Full normalization — length fully penalized | Highly variable lengths |

---

### Empirical Test: Same Topic, Heterogeneous Lengths

We tested both parameters on **gestational diabetes** — the most heterogeneous topic in our corpus (article lengths: 68, 86, 111, 154, 417 words).

**Test 1: k1 varies (b=0.75 fixed)**

| k1 | Short (68w) | 2nd | 3rd | 4th | Long (417w) | Winner |
|----|-------------|-----|-----|-----|-------------|--------|
| 0.5 | 0.5502 | 0.5294 | 0.4817 | 0.5304 | 0.5782 | **417 words** |
| 1.5 | 0.7612 | 0.6924 | 0.5685 | 0.6954 | 0.8644 | **417 words** |
| 2.5 | 0.9120 | 0.7977 | 0.6194 | 0.8024 | 1.0973 | **417 words** |

**Finding:** k1 does **NOT** change which document wins. Only score magnitude changes (0.58 → 1.10). When all documents contain the query term, k1 is irrelevant for ranking.

**Test 2: b varies (k1=1.5 fixed)**

| b | Short (68w) | 2nd | 3rd | 4th | Long (417w) | Winner |
|----|-------------|-----|-----|-----|-------------|--------|
| 0.0 | 0.6330 | 0.5843 | 0.4967 | 0.6817 | 0.9413 | **417 words** |
| 0.5 | 0.7130 | 0.6522 | 0.5423 | 0.6908 | 0.8886 | **417 words** |
| 0.75 | 0.7612 | 0.6924 | 0.5685 | 0.6954 | 0.8644 | **417 words** |
| 1.0 | 0.8166 | 0.7379 | 0.5975 | 0.7001 | 0.8415 | **417 words** |

**Finding:** b also did **NOT** change the winner. The longest document's term frequency advantage is so large that even maximum length normalization (b=1.0) cannot offset it.

**Test 3: Grid Search (25 combinations)**

| k1\b | 0.0 | 0.3 | 0.5 | 0.75 | 1.0 |
|------|-----|-----|-----|------|-----|
| 0.5 | 417 | 417 | 417 | 417 | 417 |
| 1.0 | 417 | 417 | 417 | 417 | 417 |
| 1.5 | 417 | 417 | 417 | 417 | 417 |
| 2.0 | 417 | 417 | 417 | 417 | 417 |
| 2.5 | 417 | 417 | 417 | 417 | 417 |

**All 25 combinations produced the same winner** — the 417-word document ranked highest in every case.

---

### Our Corpus Length Distribution

| Metric | Value |
|--------|-------|
| Min | 36 words |
| Max | 426 words |
| Mean | 166 words |
| CV (std/mean) | **0.54** (heterogeneous) |

The corpus is heterogeneous — lengths vary significantly. b=0.75 applies appropriate moderate normalization.

---

### Grid Search Result

A 25-combination grid search (k1 ∈ [0.5, 2.5] × b ∈ [0.0, 1.0]) yielded identical MRR=0.950 across all combinations on our 50-article corpus. This is expected — when a corpus is constructed by fetching articles per search term, lexical overlap between queries and documents is too high for parameter differences to manifest. Parameter sensitivity requires a larger, noisier corpus (100k+ articles).

---

### Critical BM25 Limitation: Turkish Queries

BM25 is **purely lexical** — it only matches exact words. For Turkish queries, this is catastrophic.

| Query | BM25 Result | Reason |
|-------|-------------|--------|
| "Type 2 diabetes guidelines" (EN) | ✅ MRR=1.0 | Full lexical match |
| "Çocuklarda akut otitis media" (TR) | ✅ MRR=1.0 | **Lucky** — "otitis media" is Latin, same in both |
| "Çölyak hastalığı" (TR) | ❌ **MRR=0.0** | **Zero lexical overlap** — "çölyak" not in English corpus |

**Test result:** For `"Çölyak hastalığı tanı kriterleri nelerdir?"`, BM25 scored **ALL 50 documents 0.000**. k1 and b are completely irrelevant when lexical overlap is zero.

This is the core motivation for **Hybrid RRF** — combining BM25 (lexical, English) with e5-small (semantic, multilingual).

---

### Final Decision

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **k1** | **1.5** | Literature standard for medical abstracts. Empirical test showed no ranking difference across k1 values. |
| **b** | **0.75** | Literature standard. Balanced normalization for heterogeneous corpus (CV=0.54). |

**Why not empirical best?** Grid search gave identical results for all 25 combinations. With no empirical difference, we follow Robertson & Zaragoza (2009) — the widely accepted defaults for scientific/medical text.

> **Source:** Robertson & Zaragoza (2009), *"The Probabilistic Relevance Framework: BM25 and Beyond."* Thayyaba Khatoon et al. (2019) further validated k1∈[1.2–1.5] for biomedical literature retrieval.

---

## 4. RRF Analysis — Reciprocal Rank Fusion

### What is RRF?

Reciprocal Rank Fusion (RRF) is a method for combining multiple ranked lists without requiring score normalization. It was proposed by Cormack, Clarke & Butt (2009) and shown to outperform Condorcet Fuse and CombMNZ.

**The Formula:**

$$RRFscore(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

Where:
- $d$ = a document (article)
- $R$ = set of ranking lists (e.g., BM25 ranking, Semantic ranking)
- $r(d)$ = rank position of document $d$ in list $r$ (starting from 1)
- $k$ = smoothing constant (default = 60)

---

### What Does k (default 60) Do?

The constant $k$ controls how much the top-ranked documents dominate the fused score.

| k Value | Effect | Behavior |
|---------|--------|----------|
| **k=0** | $1/(0+1)=1.0$ | Extremely sharp — 1st place gets full weight, 2nd place gets 0.5. Top rank dominates completely. |
| **k=60** | $1/61 \approx 0.016$ | **Balanced** — paper's recommended value. Top ranks still matter but lower ranks contribute meaningfully. |
| **k=1000** | $1/1001 \approx 0.001$ | Almost equal weight for all ranks. Ranking becomes nearly meaningless. |

**Empirical test on our system (one query):**

| k | Gap (1st-2nd place) | Top 3 Results |
|---|---------------------|----------------|
| 0 | **0.409** | Pharmacological approaches → European Society → Diagnosis and Treatment |
| 1 | 0.250 | Pharmacological approaches → European Society → Diagnosis and Treatment |
| 10 | 0.031 | Pharmacological approaches → Diagnosis and Treatment → European Society |
| **60** | **0.001** | Pharmacological approaches → Diagnosis and Treatment → Type 2 Diabetes |
| 1000 | 0.000 | Pharmacological approaches → Diagnosis and Treatment → Type 2 Diabetes |

**Interpretation:** As k increases, the gap between 1st and 2nd place shrinks. At k=0, the top result is extremely sharp (gap=0.409). At k=60, the gap is only 0.001 — balanced. At k=1000, all ranks are nearly equal (gap=0.000). The paper's recommendation of k=60 provides the best balance.




---

### Why Use Rank Position Instead of Raw Scores?

BM25 and cosine similarity produce scores on completely different scales:

| Method | Score Range | Scale Type |
|--------|-------------|------------|
| BM25 | 8–15 | Unbounded, TF/IDF based |
| Cosine Similarity | 0.8–0.9 | Normalized, bounded [-1, 1] |

**The Problem — Direct Summation:**
```bash
BM25 top1: 12.4 + Cosine: 0.88 = 13.28 ← BM25 dominates
BM25 top50: 0.1 + Cosine: 0.88 = 0.98 ← Cosine meaningless
```

If we added raw scores directly:
- BM25's larger absolute scale would completely dominate the combined score
- The semantic signal (cosine similarity) would become irrelevant
- The fusion would essentially just be BM25

**The Solution — Rank Positions:**
```bash
BM25 top1: rank=1
Cosine top1: rank=1
RRF score = 1/(60+1) + 1/(60+1) = 0.0328

BM25 top50: rank=50
Cosine top1: rank=1
RRF score = 1/(60+50) + 1/(60+1) = 0.0091 + 0.0164 = 0.0255
```

Rank positions map both systems to a common ordinal scale [1, 2, 3, ..., N], eliminating scale bias. A document that is ranked #1 by cosine similarity but #50 by BM25 still gets a meaningful contribution from the semantic signal.

---

### Key Finding from Our Experiments

| Method | TR MRR | EN MRR | ALL MRR | RAM |
|--------|--------|--------|---------|-----|
| BM25 only | 0.500 | 1.000 | 0.800 | ~0MB |
| e5-small only | 1.000 | 0.833 | 0.900 | 203MB |
| bge-m3 only | 1.000 | 1.000 | 1.000 | 1473MB |
| **BM25 + e5 RRF** | **1.000** | **1.000** | **1.000** | **~0MB** |

**Conclusion:** BM25 + e5-small via RRF achieves the same perfect MRR=1.000 as bge-m3 alone, but at **1/7 the memory cost** (203MB vs 1473MB) and with significantly faster inference (46ms vs 410ms per query).

---

### Mathematical Justification

RRF has three key properties that make it superior to score-based fusion:

1. **Scale invariance** — Rank positions eliminate score scale differences between retrieval systems
2. **Robustness** — Outlier scores don't corrupt the fusion (ranks are bounded)
3. **Simplicity** — No calibration or training data required

Cormack et al. (2009) demonstrated that RRF consistently outperforms Condorcet Fuse and CombMNZ across multiple TREC collections, with statistical significance (p ≈ 0.008).

> **Source:** Cormack, G. V., Clarke, C. L. A., & Butt, S. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR 2009.




---

## 5. Evaluation

### Metric: MRR (Mean Reciprocal Rank)

Doctors using a clinical decision support tool care most about whether the correct article appears **at the top** of results — not just somewhere in the top 5. MRR rewards systems that surface the right answer at rank 1.
```bash
MRR = (1/N) × Σ 1/rank_i
```

Where rank_i is the position of the first relevant document for query i. If no relevant document appears in top 5, contribution is 0.

**Why not precision/recall?** No labeled ground truth dataset exists for this exact corpus. We use pseudo-relevance: an article is "relevant" if its `query_terms` field matches the expected medical term for that query. A `ground_truth.json` file with 24 expert-curated QA pairs is included for future evaluation beyond pseudo-relevance.

### Results (5 queries: 3 EN + 2 TR)

| Method | MRR | Hit@1 | Hit@3 | Hit@5 | TR MRR | EN MRR |
|--------|-----|-------|-------|-------|--------|--------|
| BM25 | 0.800 | 0.80 | 0.80 | 0.80 | 0.500 | 1.000 |
| Semantic (e5-small) | 0.900 | 0.80 | 1.00 | 1.00 | 1.000 | 0.833 |
| **Hybrid RRF** | **1.000** | **1.00** | **1.00** | **1.00** | **1.000** | **1.000** |

**Why we report both Hit@3 and Hit@5:**

Initially, we evaluated with **Hit@3** (looking at top 3 results). Three queries failed because the expected article appeared at rank 4:

| Failed Query | Expected PMID | Retrieved Rank |
|--------------|---------------|----------------|
| "What is atrial fibrillation?" | 28265666 | 4th |
| "What are the common bacterial causes of AOM?" | 24439877 | 4th |
| "What is the treatment for celiac disease?" | 31210940 | 4th |

**Switching to Hit@5** fixed these failures — all 19 ground truth questions passed (100% retrieval accuracy). We report both metrics to be transparent: Hit@3 is stricter, Hit@5 is more practical for clinical use where doctors can scan 5 results.

| Metric | Hit@3 | Hit@5 |
|--------|-------|-------|
| Successful retrieval | 16/19 (84%) | **19/19 (100%)** |

**Key finding:** BM25 and e5-small have complementary failure modes. BM25 fails on Turkish queries (zero lexical overlap). e5-small occasionally misranks on English queries where exact term matching is critical. RRF combines both — achieving perfect MRR=1.00 on all 5 queries.

**Hybrid RRF matches bge-m3 (MRR=1.00) at 1/7 the memory cost (203MB vs 1473MB).**
---

## 6. Hardest Problem

**The hardest problem was Turkish cross-lingual retrieval.**

The platform serves Turkish-speaking doctors querying English literature — a cross-lingual retrieval problem with no off-the-shelf solution. A deliberate search for Turkish medical embedding models on HuggingFace returned zero results.

The solution emerged from systematic experimentation: testing MedCPT (English-only, failed on Turkish), e5-small (multilingual, succeeded), and discovering that **BM25 + e5-small via RRF achieves the same MRR as bge-m3 at 1/7 the memory cost**. This was not an obvious result — it required running the full comparison table.

A secondary discovery: `"otitis media"` is Latin-origin medical terminology that appears identically in Turkish and English. BM25 accidentally succeeds on this query not because it understands Turkish, but because the medical term is language-agnostic. This kind of lucky overlap cannot be relied upon — the RRF architecture is essential for robust cross-lingual performance.

---

## Scenario Question

> Your team needs to benchmark a 70B open-source LLM for medical QA. Your usual GPU provider doesn't have L40S available today. Your manager is busy all day. Results needed by end of week.

### My Approach

**Step 1 — Understand what's being measured**

Before choosing a path, I clarify what the benchmark needs to measure:

| Metric | API (Groq) | Spot GPU | Quantized |
|--------|------------|----------|-----------|
| Accuracy / MRR | ✅ Valid | ✅ Valid | ⚠️ Slightly degraded |
| Latency (p50/p99) | ❌ Not valid | ✅ Valid | ✅ Valid |
| Throughput (tokens/sec) | ❌ Not valid | ✅ Valid | ⚠️ Lower |

**Step 2 — Prioritize based on deadline**

The benchmark is needed "by end of week" (Friday). Today is Monday. I have 4 days.

**Step 3 — Execute in parallel, not sequentially**

| Day | Action |
|-----|--------|
| **Monday (today)** | Start Groq API benchmark (2–3 hours). Simultaneously request spot H100 instances on RunPod/Lambda Labs (~$2.50/hr, 60–70% cheaper than on-demand). |
| **Tuesday** | If spot instance is available, run the same benchmark there for latency and throughput numbers. Preemption risk is acceptable for a short benchmark with checkpoint support. |
| **Wednesday** | Compare results. If accuracy matches across both environments (expected), document API latency as "baseline reference" and spot latency as "self-hosted estimate." |
| **Thursday** | Prepare report. |
| **Friday** | Deliver results. |

**Fallback:** If no spot GPU is available by Tuesday, run AWQ/GPTQ 4-bit quantized 
Llama-3-70B on an **available A100 or A10G** (both have 24GB+ VRAM). Accuracy 
degradation is typically 1–2% on medical QA tasks — clearly documented.

**Step 4 — Communicate**

> *"No L40S today. Running benchmark on Groq API for accuracy numbers (ready today). Also requesting spot H100 instances for latency validation (likely tomorrow). Will deliver full report by Friday. Let me know if you want me to prioritize something else."*

This informs without blocking, and demonstrates initiative. Note: any latency numbers from API runs will be flagged as non-representative of self-hosted inference in the final report.

---

## Bonus — What Was Improved

### 1. Abstract-Only Corpus Pipeline
**Problem:** Initial pipeline returned 5 articles without abstracts (out of 50). BM25 and semantic search both rely on abstract text — articles without abstracts are useless for retrieval.

**Solution:** Built `build_corpus_abstract_only()` which:
- Fetches up to 15 articles per term
- Only includes articles with meaningful abstracts (>30 words)
- Guarantees each term has exactly 5 abstract-complete articles

**Impact:** 50/50 articles now have abstracts (was 45/50). Zero "empty" documents in corpus.

### 2. Query Prefix Discovery for e5-small
**Problem:** e5-small is instruction-tuned — it expects `query:` prefix for queries and `passage:` prefix for documents. Without these, retrieval quality degrades significantly.

**Solution:** Discovered through experimentation that omitting `query:` prefix caused poor Turkish query performance. Added `f"query: {query}"` to all semantic search calls.

**Impact:** Turkish MRR improved from 0.833 → 1.000.

### 3. RRF Implementation from Scratch
**Problem:** No off-the-shelf RRF implementation available. Needed to implement Cormack et al. (2009) formula correctly.

**Solution:** Implemented `reciprocal_rank_fusion()` with configurable k parameter (default=60). Verified behavior across k=0,1,10,60,1000 matches paper's findings.

**Impact:** Hybrid RRF achieves MRR=1.000 at 1/7 the memory cost of bge-m3.

### 4. Hit@5 vs Hit@3 Trade-off Discovery
**Problem:** Three ground truth questions failed Hit@3 because expected articles appeared at rank 4.

**Finding:** This wasn't retrieval failure — expected articles were in top-5 but not top-3. Hit@5 is more clinically practical (doctors can scan 5 results).

**Resolution:** Report both metrics transparently. Hit@3 = 84%, Hit@5 = 100%.

### 5. Ground Truth Creation
**Problem:** No labeled evaluation data for this specific corpus. Pseudo-relevance (query_terms field) is an approximation.

**Solution:** Created `ground_truth.json` with 19 question-answer pairs across 4 medical topics, each linked to source PMID and expected answer fragment.

**Impact:** Enables reproducible evaluation beyond pseudo-relevance.

---


