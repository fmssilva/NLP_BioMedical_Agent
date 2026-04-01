check the notebook 


in set up and environment, confirm that it will work also in colab. 
example in terms of .env, how do we do for colab to access .env to get keys for opensarch to work for example? what is the best practice for that?


C:\Users\franc\Desktop\NLP_Biomedical_Agent\results

then confirm that our csv of model finetunnning etc... are also called to be generated in th notebook directly so if a new user runs the notebook, everything will work » we will fine tune the model and generate the needed csv and then they will be loaded automatically. 


then lets confirm something. now we have 2 flavours of wrels
**Qrels & NDCG**  
From the Ground Truth data we derive the Qrels (relevance judgements) for evaluating our retrieval models.  
Basically, we look at the citations in the Ground Truth answers and if a citation has `"evidence_relation": "supporting"` for a query topic, we call it relevant. We maintain two flavours of qrels:

| Flavour    | Scoring                                 | Used for              |
| ---------- | --------------------------------------- | --------------------- |
| **Binary** | supporting=1, everything else=0         | MAP, MRR, P@10, R@100 |
| **Graded** | supporting=2, neutral=1, not relevant=0 | NDCG@10               |


and we say **Why two?**  
MAP and precision are inherently binary metrics — a document is either relevant or not. The professor notes say: *"for computing the precision we will need to define a threshold"*. Our threshold: only `supporting` citations count as relevant for binary metrics.

but couldn't we just have the graded qrels and use them for both? For MAP/MRR/P@10, we would treat `supporting` as relevant (1) and everything else as non-relevant (0). For NDCG, we would use the full graded scale. This would simplify our evaluation pipeline by maintaining a single qrels file or not? 

and also, i think the qrel is not correctly done. we need to implement compreensive tests to the logic to confirm it is actually creating a map of query_id → doc_id → relevance_score correctly. 

