# References — Lab Code Index

This index lets an AI agent or developer find specific code patterns in the course lab notebooks **without reading them in full**. Each entry shows: what the code does, where it lives (file + line range), and any important notes.

---

## Lab01_OpenSearch.ipynb (717 lines)

### OpenSearch Client Setup
**Lines 65–98** — Connect to `api.novasearch.org:443`, check index exists, print settings/mappings/doc count.
```python
client = OpenSearch(
    hosts=[{'host': host, 'port': port}],
    http_compress=True, http_auth=(user, password),
    use_ssl=True, url_prefix='opensearch_v3',
    verify_certs=False, ssl_assert_hostname=False, ssl_show_warn=False
)
```
- `client.indices.exists(index=index_name)` — check before operations
- `client.indices.get_settings()`, `get_mapping()`, `client.count()` — verify index

### Close/Delete Index
**Lines 105–106** — `client.indices.close(index=index_name)`  
**Lines 211–223** — Delete index pattern (guarded). _Contains a deliberate syntax error to prevent accidental deletion._

### Index Creation — BM25 + KNN (basic)
**Lines 129–166** — Create index with `number_of_shards=4`, `number_of_replicas=0`, `refresh_interval="-1"`, `knn=true`. Fields: `doc_id` (keyword), `tags` (keyword), `json` (flat_object), `contents` (text, standard analyzer, BM25).

### Index Creation — KNN Vector Field
**Lines 404–448** — Add `sentence_embedding` field (`knn_vector`, dim=768, HNSW faiss innerproduct, ef_construction=256, m=48). This is the pattern for dense retrieval.
```python
"sentence_embedding": {
    "type": "knn_vector", "dimension": 768,
    "method": {"name": "hnsw", "space_type": "innerproduct",
               "engine": "faiss", "parameters": {"ef_construction": 256, "m": 48}}
}
```

### Refresh Settings After Indexing
**Lines 173–193** — `client.indices.put_settings()` to update `refresh_interval` to `"1s"` after bulk index.

### Analyzer Test
**Lines 219–231** — `client.indices.analyze(body={"analyzer": "standard", "text": ...})` to test tokenization.

### Simple Document Indexing (one at a time)
**Lines 243–264** — `client.index(index=index_name, id=1, body=doc)` — single document insert.

### Text Search (BM25 match)
**Lines 297–319** — BM25 `match` query pattern:
```python
query = {"size": 5, "_source": ["doc_id"], "query": {"match": {"contents": {"query": qtxt}}}}
response = client.search(body=query, index=index_name)
```
Note: `'size': 5` — must be explicit. Default returns 10.

### Term Query
**Lines 329–348** — `{"query": {"term": {"tags": "red"}}}` — exact keyword match.

### Boolean Query
**Lines 358–390** — `bool` with `must` (term) + `should` (multi_match) — combine filters with ranking.

### Sentence Encoder (msmarco-distilbert-base-v2)
**Lines 473–509** — **Key pattern used throughout the project:**
```python
from transformers import AutoTokenizer, AutoModel
import torch, torch.nn.functional as F

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode(texts):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return F.normalize(embeddings, p=2, dim=1)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/msmarco-distilbert-base-v2")
```
- Uses `return_dict=True` and `.last_hidden_state`
- NOT using `sentence-transformers` high-level API

### Index Document with Embedding
**Lines 518–533** — Store embedding alongside text:
```python
doc = {'doc_id': 'documentA', 'contents': text, 'sentence_embedding': emb[0].numpy()}
client.index(index=index_name, id=1, body=doc)
```

### KNN Search Query
**Lines 541–567** — Dense vector query pattern:
```python
query = {
    "size": 5, "_source": ["doc_id"],
    "query": {"knn": {"sentence_embedding": {"vector": query_emb[0].numpy(), "k": 2}}}
}
```
Note: set both `"size"` (top-level) and `"k"` (inside knn block) to same value.

### Custom Analyzer (edge_ngram)
**Lines 627–676** — Custom analyzer with `edge_ngram_filter` for autocomplete queries.

### Fine-tuning Dual Encoders
**Lines 600–616** — `SentenceTransformer` + `CosineSimilarityLoss` + `DataLoader.fit()` — for reference only, not used in this project.

### Spacy NER
**Lines 683–716** — `spacy.load("en_core_web_sm")`, token info (lemma, POS, dep), `doc.ents` — for reference.

---

## Lab02_vLLM_NLP_Tasks.ipynb (602 lines)

### vLLM Client Setup
**Lines 48–57** — Connect to `amalia.novasearch.org/vlm/v1`:
```python
from openai import OpenAI
client = OpenAI(base_url="https://amalia.novasearch.org/vlm/v1", api_key="amalia012026")
```

### List Available Models
**Lines 60–67** — `client.models.list()` → `MODEL = models.data[0].id`.

### Raw Text Completion
**Lines 78–85** — `client.completions.create(model=MODEL, prompt=..., max_tokens=20, temperature=0.7)`.

### Chat Completion (single turn)
**Lines 93–103** — Standard chat pattern:
```python
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
    max_tokens=200, temperature=0.3,
)
print(response.choices[0].message.content)
```

### Multi-turn Conversation
**Lines 111–129** — Accumulate `history` list, append user + assistant messages each turn.

### Single-turn Helper Function
**Lines 140–151** — `ask(system_prompt, user_prompt, temperature=0.2)` helper — reused throughout.

### Text Summarization
**Lines 159–175** — Zero-shot summarization prompt pattern.

### Sentiment Analysis
**Lines 193–210** — Prompt for single-word sentiment label (Positive/Negative/Neutral).

### Named Entity Recognition (NER)
**Lines 225–245** — Prompt for `entity | TYPE` format extraction.

### Machine Translation
**Lines 260–274** — Multi-language translation prompt.

### Extractive QA
**Lines 282–308** — Context + question → answer from provided passage only.

### Text Classification
**Lines 316–333** — Classify text into fixed category list.

### Output Length Control
**Lines 348–369** — Prompt-based length constraints ("one sentence", "bullet points", etc.).

### Query Rewriting
**Lines 379–400** — Rewrite search query for better retrieval — relevant to RAG pipelines.

### Structured Output (JSON mode)
**Lines 416–442** — Key pattern for Phase 2 judge:
```python
response = client.chat.completions.create(
    model=MODEL,
    messages=[...],
    response_format={"type": "json_object"},
    max_tokens=256, temperature=0.0,
)
result = json.loads(response.choices[0].message.content)
```

### Streaming Responses
**Lines 458–476** — `stream=True` + iterate chunks: `chunk.choices[0].delta.content`.

### Generation Parameters Table
**Lines 479–491** — Reference: temperature, top_p, top_k, max_tokens, presence_penalty, frequency_penalty.

---

## Lab03_Retrieval_Evaluation.ipynb (923 lines)

### Imports and Seeding
**Lines 39–47** — `numpy`, `matplotlib`, `collections.defaultdict`, `random`. `plt.rcParams.update({'figure.dpi': 120})`.

### Toy Corpus and Rankings
**Lines 70–83** — Setup: `relevance_labels = [True, False, True, ...]`, `ranking_A`, `ranking_B` (0-indexed doc IDs).

### Precision@k
**Lines 86–119** — Implementation + sklearn version:
```python
def precision_at_k(ranking, relevance, k):
    top_k = ranking[:k]
    n_relevant_in_top_k = sum(relevance[doc_id] for doc_id in top_k)
    return n_relevant_in_top_k / k
```

### Recall@k
**Lines 86–119** — Same cell as above:
```python
def recall_at_k(ranking, relevance, k):
    top_k = ranking[:k]
    n_relevant_in_top_k = sum(relevance[doc_id] for doc_id in top_k)
    total_relevant = sum(relevance)
    return n_relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0
```

### PR Curve
**Lines 180–244** — `pr_curve(ranking, relevance)` — raw curve at each relevant doc rank. Returns `(recalls, precisions)`.
```python
def pr_curve(ranking, relevance):
    total_relevant = sum(relevance)
    precisions, recalls = [], []
    n_relevant_seen = 0
    for rank, doc_id in enumerate(ranking, start=1):
        if relevance[doc_id]:
            n_relevant_seen += 1
            precisions.append(n_relevant_seen / rank)
            recalls.append(n_relevant_seen / total_relevant)
    return recalls, precisions
```

### 11-Point Interpolated PR Curve
**Lines 200–213** — `interpolated_pr_curve(recalls, precisions, n_points=11)`:
```python
recall_levels = np.linspace(0, 1, n_points)
interp_precisions = []
for r_level in recall_levels:
    relevant_precisions = [p for r, p in zip(recalls, precisions) if r >= r_level]
    interp_precisions.append(max(relevant_precisions) if relevant_precisions else 0.0)
return recall_levels, np.array(interp_precisions)
```

### Plotting PR Curves
**Lines 214–244** — `fig, axes = plt.subplots(1, 2, figsize=(13, 5))`. Raw and interpolated side by side.
- `ax.step(r, p, where='post', color='steelblue', lw=2, label='System A')`
- `ax.fill_between(rl, mean_p, alpha=0.1, color='steelblue')`

### Average Precision
**Lines 268–305** — `average_precision(ranking, relevance)`:
```python
def average_precision(ranking, relevance):
    total_relevant = sum(relevance)
    if total_relevant == 0:
        return 0.0
    score = 0.0
    n_relevant_seen = 0
    for rank, doc_id in enumerate(ranking, start=1):
        if relevance[doc_id]:
            n_relevant_seen += 1
            score += n_relevant_seen / rank
    return score / total_relevant
```

### MAP (Mean Average Precision)
**Lines 321–382** — `mean_average_precision(queries)` — `queries` is list of `(labels, ranking)`:
```python
def mean_average_precision(queries):
    return np.mean([average_precision(rank, labels) for labels, rank in queries])
```
**IMPORTANT NOTE:** Lab03 uses `np.mean()` which includes topics with 0 relevant docs (AP=0). For TREC standard, exclude 0-relevant topics and log a warning.

### MRR (Mean Reciprocal Rank)
**Lines 406–428** — `reciprocal_rank(ranking, relevance)` + `mean_reciprocal_rank(queries)`:
```python
def reciprocal_rank(ranking, relevance):
    for rank, doc_id in enumerate(ranking, start=1):
        if relevance[doc_id]:
            return 1.0 / rank
    return 0.0
```

### Mean PR Curve (across queries)
**Lines 438–488** — `mean_pr_curve(queries, n_points=11)`:
```python
def mean_pr_curve(queries, n_points=11):
    all_interp = []
    for labels, ranking in queries:
        r, p = pr_curve(ranking, labels)
        _, ip = interpolated_pr_curve(r, p, n_points)
        all_interp.append(ip)
    recall_levels = np.linspace(0, 1, n_points)
    return recall_levels, np.mean(all_interp, axis=0)
```

### Comparison Plot (PR curves + bar chart)
**Lines 438–488** — `fig, axes = plt.subplots(1, 2, figsize=(13, 5))`:
- Left: mean PR curves with `fill_between` for variance, MAP in legend
- Right: bar chart MAP+MRR with value labels via `ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.3f}', ...)`

### Real Corpus + LMD Index Creation
**Lines 644–665** — Create LMD index (real OpenSearch pattern for eval):
```python
index_body = {
    "settings": {"similarity": {"default": {"type": "LMDirichlet", "mu": 2000}}},
    "mappings": {"properties": {"doc_id": {"type": "keyword"}, "text": {"type": "text", "analyzer": "english"}}}
}
```
Note: Lab03 uses `"analyzer": "english"` here — but for biomedical text we use `"standard"` instead.

### Bulk Indexing Pattern
**Lines 668–694** — `opensearchpy.helpers.bulk()`:
```python
from opensearchpy.helpers import bulk
actions = [{"_index": INDEX_NAME, "_id": doc["id"], "_source": {"doc_id": ..., "text": ...}} for doc in corpus]
success, errors = bulk(client, actions)
client.indices.refresh(index=INDEX_NAME)
```

### LMD Search Function
**Lines 719–748** — `lmd_search(query_text, index, n_results=30)`:
```python
response = client.search(index=index, body={
    "size": n_results,
    "query": {"match": {"text": {"query": query_text}}},
    "_source": ["doc_id", "topic"],
})
return [(h["_source"]["doc_id"], h["_score"]) for h in response["hits"]["hits"]]
```

### `results_to_ranking()` — Critical Conversion Function
**Lines 756–805** — Convert OpenSearch results to metric format (int-indexed):
```python
def results_to_ranking(results, qrels_set, all_doc_ids):
    id_to_idx = {doc_id: i for i, doc_id in enumerate(all_doc_ids)}
    relevance = [doc_id in qrels_set for doc_id in all_doc_ids]
    retrieved_ids = [r[0] for r in results]
    retrieved_set = set(retrieved_ids)
    not_retrieved = [doc_id for doc_id in all_doc_ids if doc_id not in retrieved_set]
    full_ranking_ids = retrieved_ids + not_retrieved
    ranking = [id_to_idx[doc_id] for doc_id in full_ranking_ids]
    return relevance, ranking
```

### Full Evaluation Loop (per-query metrics)
**Lines 756–805** — Per-query table: AP, RR, P@5, R@5, P@10, R@10 + mean row.

### Per-query PR curve + relevance heatmap
**Lines 811–855** — Two subplots: (1) individual PR curves + mean curve, (2) `imshow` heatmap of relevant docs in top-15 positions (RdYlGn colormap).

---

## example_of_cover_notebook_for_colab.ipynb

Reference notebook showing the standard Colab/local detection + repo setup pattern. See `PHASE_1_PLAN.md` Section 12 for the exact code template used in this project.

---

## similatiry_guides.md

Reference guide on OpenSearch similarity functions (BM25, LMJelinekMercer, LMDirichlet). Key formulas and parameter explanations for the three similarity plugins used in the index.

## vectors_guide.md

Reference guide on vector representations, embedding spaces, cosine similarity, inner product, HNSW parameters. Explains why `space_type=innerproduct` + L2-normalised vectors = cosine similarity.

---

## Quick Reference — Key Patterns This Project Uses

| Task                      | Lab   | Lines            | Pattern                                                                               |
| ------------------------- | ----- | ---------------- | ------------------------------------------------------------------------------------- |
| OpenSearch client connect | Lab01 | 65–98            | `OpenSearch(hosts, http_auth, use_ssl, url_prefix='opensearch_v3', ...)`              |
| Index creation (full)     | Lab01 | 129–166, 404–448 | `client.indices.create(index=name, body=index_body)`                                  |
| Bulk index documents      | Lab03 | 668–694          | `helpers.bulk(client, actions)` + `refresh()`                                         |
| BM25 match query          | Lab01 | 297–319          | `{"query": {"match": {"contents": {"query": text}}}, "size": 100}`                    |
| KNN vector query          | Lab01 | 541–567          | `{"query": {"knn": {"embedding": {"vector": emb.numpy(), "k": size}}}, "size": size}` |
| Encode text → embedding   | Lab01 | 473–509          | `AutoTokenizer + AutoModel + mean_pooling + F.normalize`                              |
| LMD / LM-JM search        | Lab03 | 719–748          | Same `match` query on the field with LMD/LMJM similarity                              |
| Bulk index actions        | Lab03 | 668–694          | `[{"_index": name, "_id": id, "_source": {...}}]` → `bulk()`                          |
| Precision@k, Recall@k     | Lab03 | 86–119           | `sum(relevance[d] for d in ranking[:k]) / k`                                          |
| Average Precision         | Lab03 | 268–305          | Accumulate `n_rel_seen / rank` at each relevant doc                                   |
| MRR                       | Lab03 | 406–428          | `1.0 / rank` at first relevant doc                                                    |
| 11-pt interpolated PR     | Lab03 | 200–213          | `max(p for r, p in zip(recalls, precs) if r >= r_level)`                              |
| results_to_ranking()      | Lab03 | 756–805          | Convert `(doc_id, score)` list → int-indexed `(relevance, ranking)`                   |
| vLLM chat completion      | Lab02 | 93–103           | `openai.OpenAI(base_url=...).chat.completions.create(...)`                            |
| JSON structured output    | Lab02 | 416–442          | `response_format={"type": "json_object"}` + `json.loads()`                            |
