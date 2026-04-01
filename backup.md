---
## 1.2. Data Loading & Analysis

**Corpus:**  
4194 PubMed abstracts from `filtered_pubmed_abstracts.json`.  
Example:
> ```json
> {"id": "21824858", 
>  "contents": "[Skeletal and extra-skeletal consequences of vitamin D deficiency].\nVitamin D is obtained from cutaneous production when 7-dehydrocholesterol is converted to vitamin D(3) (cholecalciferol) by ultraviolet B radiation or by oral intake of vitamin D....\n"}
> ```

---

**Topics:**  
65 clinical queries for us to use (we can query by one of the fields, or all 3 combined).  
Example of topic from file `BioGen_2024_topics.json`:
> ```json
> {"id": 132,
>  "topic": "low vitamin D",
>  "question": "What is the effect of low vitamin D?",
>  "narrative": "The patient diagnosed with vitamin D deficiency would like to know how the problem affects his health."}
> ```

---

**Ground Truth:**  
Derived from `biogen_2024_submissions.json`  
> We divided for train (odd query-topic pairs = 32 queries) → test (even query-topic pairs = 33 queries) split.  
> For each answer to a query we might have different relevance judgement information which we use as Qrels:  
> - For the whole answer we see - `is_answer_accurate`: yes/no  
> - For each sentence in the answer (`answer_sentence_relevance`: required/optional/not relevant)  
> - For each sentence we can see a set of citations and respective assessment (sometimes null, other times many, depending on the model, etc.)  
> - For each citation we can see the evidence support text and the relation (supporting/contradicting/neutral):  
> 
> Example:  
> ```json
> "question_id": "132",
> "machine_generated_answers": {
>   "iiresearch_trec_bio2024_t5base_run": {
>     "answer": "Vitamin D plays a crucial role in various bodily functions, and its deficiency can have several adverse effects. Low vitamin D levels are associated with...",
>     "is_answer_accurate": "yes",
>     "answer_sentences": [
>       {
>         "answer_sentence_id": "1",
>         "answer_sentence": "Vitamin D plays a crucial role in various bodily functions, and its deficiency can have several adverse effects.",
>         "answer_sentence_relevance": "required",
>         "citation_assessment": null
>       },
>       ...
>     ]
>   },
>   "listgalore_gpt-4o_arenuggetsallyouneed": {
>     "answer": "Low vitamin D has significant health implications, including an increased risk of cardiovascular disease, fractures, and osteoporosis [21824858, 23609564, 24622671]. It leads to bone loss or osteomalacia...",
>     "is_answer_accurate": "yes",
>     "answer_sentences": [
>       {
>         "answer_sentence_id": "1",
>         "answer_sentence": "Low vitamin D has significant health implications, including an increased risk of cardiovascular disease, fractures, and osteoporosis [21824858, 23609564, 24622671].",
>         "answer_sentence_relevance": "required",
>         "citation_assessment": [
>           {
>             "cited_pmid": "21824858",
>             "evidence_support": "In utero and during childhood, vitamin D deficiency can cause growth retardation and skeletal deformities and may increase...",
>             "evidence_relation": "supporting"
>           },
>           ...
>         ]
>       },
>       ...
>     ]
>   }
> }
> ```

---

**Qrels & NDCG**  
From the Ground Truth data we derive the Qrels (relevance judgements) for evaluating our retrieval models.  
Basically, we look at the citations in the Ground Truth answers and if a citation has `"evidence_relation": "supporting"` for a query topic, we call it relevant. We maintain two flavours of qrels:

| Flavour    | Scoring                                 | Used for              |
| ---------- | --------------------------------------- | --------------------- |
| **Binary** | supporting=1, everything else=0         | MAP, MRR, P@10, R@100 |
| **Graded** | supporting=2, neutral=1, not relevant=0 | NDCG@10               |

**Why two?**  
MAP and precision are inherently binary metrics — a document is either relevant or not. The professor notes say: *"for computing the precision we will need to define a threshold"*. Our threshold: only `supporting` citations count as relevant for binary metrics.

---

**NDCG** (Normalised Discounted Cumulative Gain) answers: *"how good is the ranking, considering that some documents are MORE relevant than others?"*

The intuition behind NDCG:
- **Gain:** Each document has a relevance score (0, 1, or 2). A "supporting" doc (score=2) is worth more than a "neutral" one (score=1).
- **Discounted:** Documents lower in the ranking get discounted — a relevant doc at rank 1 is worth more than the same doc at rank 50. The discount factor is $\frac{1}{\log_2(\text{rank}+1)}$.
- **Normalised:** We divide by the ideal DCG (what you'd get if you ranked all documents perfectly). So NDCG=1.0 means perfect ranking; 0.0 means nothing relevant was found.

**Why use NDCG instead of just MAP?**  
MAP treats all relevant docs equally (relevant or not). NDCG says *"a highly relevant doc ranked first matters more than a marginally relevant one."* In our case: a supporting citation at rank 1 is better than a neutral citation at rank 1.

> See §16 for the full discussion of binary vs graded evaluation and our threshold choices.