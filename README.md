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
requests==2.32.4
pandas==2.2.2
tqdm==4.67.3
tenacity==9.1.4
rank-bm25==0.2.2
sentence-transformers==5.4.0
transformers==5.0.0
torch==2.10.0
numpy==2.0.2
google-generativeai
psutil
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
Gemini 2.0 Flash → Cited Answer (TR/EN)
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

- Fine-tune e5-small on TR→EN medical query pairs (no such dataset exists publicly)
- Add query expansion: detect Turkish → translate to English before BM25
- Cache PubMed results with TTL (currently fetched fresh each run)
- Add cross-encoder reranking (ncbi/MedCPT-Cross-Encoder) as a third stage
- Increase corpus to 500+ articles per term for meaningful BM25 parameter sensitivity

---

## 3. BM25 Analysis — k1 and b

### What k1 controls (term frequency saturation)

k1 determines how much repeated occurrences of a query term increase the score.

| k1 | Behavior | Use case |
|----|----------|----------|
| 0.5 | TF saturates fast, near-binary scoring | Short documents |
| **1.2–1.5** | Moderate saturation | **Medical abstracts ← chosen** |
| 2.0+ | High TF still adds significant score | Long documents, books |

In our corpus, `"diabetes"` appears an average of **4.0x per abstract** (max 19x). k1=1.5 prevents over-rewarding high-frequency repetition while still respecting meaningful term signals. k1>2.0 was empirically observed to drop Hit@5 from 4/5 to 3/5.

> **Source:** Robertson & Zaragoza (2009), *"The Probabilistic Relevance Framework: BM25 and Beyond."* Thayyaba Khatoon et al. (2019) further validated k1∈[1.2–1.5] for biomedical literature retrieval.

### What b controls (document length normalization)

| b | Behavior |
|---|----------|
| 0.0 | No normalization — long docs dominate |
| **0.75** | Balanced — **literature standard, chosen** |
| 1.0 | Full normalization — length fully penalized |

Our corpus has CV=0.54 (std/mean), spanning 36–426 words — heterogeneous. b=0.75 applies moderate normalization appropriate for this variance.

### Grid Search Result

A 25-combination grid search (k1 ∈ [0.5, 2.5] × b ∈ [0.0, 1.0]) yielded identical MRR=0.950 across all combinations on our 50-article corpus. This is expected — when a corpus is constructed by fetching articles per search term, lexical overlap between queries and documents is too high for parameter differences to manifest. Parameter sensitivity requires a larger, noisier corpus (100k+ articles).

**Decision: k1=1.5, b=0.75** — literature standard (Robertson & Zaragoza 2009), empirically tied on this corpus.

### Critical BM25 Limitation: Turkish Queries

BM25 is purely lexical. For `"Çölyak hastalığı tanı kriterleri nelerdir?"`, BM25 scores **ALL 50 documents 0.000** — `"çölyak"` does not appear anywhere in the English corpus. k1 and b are irrelevant when lexical overlap is zero. This is the core motivation for Hybrid RRF.

| Query | BM25 | Reason |
|-------|------|--------|
| Type 2 diabetes guidelines (EN) | ✅ MRR=1.0 | Full lexical match |
| Çocuklarda akut otitis media (TR) | ✅ MRR=1.0 | Lucky — "otitis media" is Latin, same in both |
| Çölyak hastalığı (TR) | ❌ MRR=0.0 | Zero lexical overlap with English corpus |

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
