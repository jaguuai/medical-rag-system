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

## 4. RRF Analysis

### What does k (default 60) do?

RRF formula: `RRF(d) = Σ 1 / (k + rank(d))`

k is a smoothing constant that controls how much the top-ranked documents dominate.

| k | Effect |
|---|--------|
| k=0 | 1st place scores 1.0 — extremely sharp, top rank dominates everything |
| k=60 | 1st place scores 1/61 ≈ 0.016 — balanced, paper's recommendation |
| k=1000 | 1st place scores 1/1001 ≈ 0.001 — rankings nearly equal weight, position becomes meaningless |

Empirically observed: at k=0 the gap between 1st and 2nd place was 0.409; at k=60 it was 0.001265; at k=1000 it was 0.000005. Lower k gives sharper ranking but risks over-committing to a single system's top result.

Cormack et al. (2009) recommend k=60 as robust across multiple retrieval settings — we adopt this default.

### Why rank position instead of raw scores?

BM25 scores are absolute TF/IDF-based numbers (typically 8–15 in our corpus). Cosine similarity is normalized to [-1, 1] (typically 0.83–0.88).

Adding these directly:
```
BM25 top1:    12.4  +  cosine: 0.88  =  13.28  ← BM25 dominates
BM25 top50:    0.1  +  cosine: 0.88  =   0.98  ← cosine meaningless
```

BM25's larger absolute scale completely dominates the combined score, making the semantic signal irrelevant. Rank position maps both systems to a common ordinal scale [1, 2, 3, ...N], eliminating scale bias and allowing fair combination.

---

## 5. Evaluation

### Metric: MRR (Mean Reciprocal Rank)

Doctors using a clinical decision support tool care most about whether the correct article appears **at the top** of results — not just somewhere in the top 5. MRR rewards systems that surface the right answer at rank 1.

```
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

**My approach:**

**Option 1 — API-based benchmark (fastest):**
Groq offers Llama-3.3-70B at sub-second latency on their free tier. I would run the benchmark against the Groq API immediately — results in hours, not days. Tradeoff: latency profile differs from self-hosted GPU; throughput is limited by API rate limits.

**Option 2 — Alternative GPU providers:**
RunPod and Lambda Labs typically have H100/A100 availability when L40S is unavailable. I would check both dashboards, spin up a spot instance (H100 SXM at ~$3.50/hr), and run the benchmark there. Two H100s can handle Llama-3.3-70B in FP8 at reasonable throughput.

**Option 3 — Together.ai or Fireworks.ai:**
Both offer serverless 70B inference with millisecond cold starts. Pricing is per-token, suitable for a one-time benchmark.

**Communication:**
I would send my manager a brief async message (Slack/email) outlining the plan and chosen tradeoff before proceeding — not to ask for permission, but to keep them informed.

**Documentation:**
Any benchmark run via API must note that latency numbers are not representative of self-hosted inference. I would clearly flag this in the results and run a small self-hosted sample on available hardware to cross-validate quality metrics.

---

## Bonus — What Was Improved

### 1. Model Benchmarking (not in spec)
Instead of picking a model arbitrarily, three embedding models were systematically benchmarked: MedCPT, e5-small, and bge-m3 — across both Turkish and English queries. This revealed that e5-small+BM25 RRF matches bge-m3 at 1/7 the cost.

### 2. Turkish NLP Gap Analysis
Confirmed via HuggingFace search that no Turkish medical embedding model exists. Documented this as a research gap in the README.

### 3. Ground Truth Dataset
Created `ground_truth.json` with 24 expert-curated QA pairs across 4 medical topics with source PMIDs, enabling reproducible evaluation beyond pseudo-relevance.

### 4. Abstract-Only Corpus Pipeline
The initial pipeline returned articles without abstracts (5/50 missing). Built `build_corpus_abstract_only()` which fetches additional candidates until exactly 5 abstract-complete articles are collected per term.

---

## AI Tool Usage

This project was built with Claude (Anthropic) as a coding assistant, used for: debugging XML parsing errors, structuring the evaluation framework, and drafting README sections. All architectural decisions, model selections, and experimental results are original. Per assessment rules, AI tool usage is disclosed here.
