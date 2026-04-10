# **BioMedical Natural Language Agents**
#### Natural Language Processing and Search Course 2025/2026

[This project guide is largely based on TREC BioGen 2024 and 2025](https://trec-biogen.github.io/docs/)

João Magalhães and David Semedo

**Version:** Mar 23, 2026

**Deadlines:**


Phase 1: April 13, 2026​ Code + Report
Phase 2: May 4, 2026 Code + Report
Phase 3: June 1, 2026​ Code + Report

## **Introduction**


With the advancement of large language models (LLMs), the biomedical domain has
seen significant progress and improvement in multiple tasks such as biomedical
question answering, lay language summarization of the biomedical literature, clinical
note summation, etc.


However, hallucinations or confabulations remain one of the key challenges when using
LLMs in the biomedical domain. Inaccuracies may be particularly harmful in high-risk
situations, such as making clinical decisions or appraising biomedical research.


The project is structured into three phases:


1.​ **Phase 1 - Search and Evaluation:** Given a query topic you will study methods

to retrieve relevant documents. The setup of your experimental test bed and the
corresponding evaluation methodology is an important aspect of this part.


2.​ **Phase 2 - RAG and LLM-Judges:** In this phase, you will study methods to

generate a single answer that is leveraged by the retrieved documents. An
LLM-as-a-judge will be used to measure the quality of the generated answers.


3.​ **Phase 3 - Deep Research Agent:** The final phase will address LLM agentic

patterns to plan, aggregate evidence and create a report based on the requested
topic. Exploration methods will be used to determine sub-topics and structure the
result accordingly.


![proj_guide_NLPS_-_TREC_2025_BioGen_img_001.png](pdf_images\proj_guide_NLPS_-_TREC_2025_BioGen_img_001.png)
## **Resources and Evaluation**

### **PubMed Articles**

We are using a sample of the PubMed corpus to retrieve relevant documents. The
processed corpus is available for download on the course website. All articles should be
indexed in OpenSearch by their title, abstract and PMID. Although it is not needed, you
are allowed to query PubMed to get the full article text.

### **Query-Topics**


The query-topics pairs are available for download on the course website. A query-topic
is structured as this example:


**id:** 116

**topic:** "natural treatments for sleep apnea"
**question:** "Are there ways to prevent sleep apnea or treat it naturally?"
**narrative:** "The patient is looking for natural remedies to prevent and treat sleep apnea."


Your query can be one of the three available fields, all three fields combined.


### **Ground Truth**

The ground truth for each pair (Query-topic, PMID) is available on the course website.
The ground truth should be divided as follows:


  - ​ **Training set:**  - dd query-topic pairs.

  - ​ **Test set:** even query-topics pairs.


**Ground Truth Transformation**


Annotations need to be transformed into ground truth. Specifically, the
**"citation_assessment"** fields need to be analyzed to extract the relevance judgments:
numerical values must be assigned to the labels (such as **"required"**, **"supporting"**,
and **"borderline"** ) to enable the calculation of the **nDCG** (Normalized Discounted
Cumulative Gain).

## **Phase 1: Grounding Topic on PubMed Data**

### **Document retrieval**


Given a biomedical topic (question) and a stable version of PubMed documents, you
should be able to retrieve a set of documents that are relevant to the topic.


To achieve this goal, you should implement the following retrieval strategies:


**#1 BM25 similarity**
[https://docs.opensearch.org/latest/im-plugin/similarity/#bm25-similarity-default](https://docs.opensearch.org/latest/im-plugin/similarity/#bm25-similarity-default)


**#2 Language Model with Jelineck-Mercer smoothing**
[https://docs.opensearch.org/latest/im-plugin/similarity/#lm-jelinek-mercer-similarity](https://docs.opensearch.org/latest/im-plugin/similarity/#lm-jelinek-mercer-similarity)


**#3 Language Model with Dirichlet smoothing**
[https://docs.opensearch.org/latest/im-plugin/similarity/#lm-dirichlet-similarity](https://docs.opensearch.org/latest/im-plugin/similarity/#lm-dirichlet-similarity)


**#4 Knn indexes with LLM embeddings**
[https://docs.opensearch.org/latest/vector-search/getting-started/vector-search-options](https://docs.opensearch.org/latest/vector-search/getting-started/vector-search-options/)


To address the above questions, you should build on the provided tutorials to parse and

index documents.


**Configuration per Field Index:** Note that you need to define the various **similarity**
**metrics** at the beginning of the index field definition:

```
"index":{
"number_of_replicas":0,
"number_of_shards":4,
"refresh_interval":"-1",
"knn":"true",
"similarity": {
"my_lm_dirichlet": {
"type": "LMDirichlet",
"mu": 2000

}

}

}

### **Evaluation**

```

To assess the retrieved documents, you will use both system utility metrics and system
stability metrics. For system utility metrics, you should use:


  - ​ **Precision@10** to measure the percentage number of correct documents in the

top 10 results.

  - ​ **Recall@100** to measure how many relevant results are not accessible to users.

  - ​ **NDCG** to measure how the graded relevance evolves over the ranked results.


For system stability metrics, you should use:


  - ​ **Precision-recall curves** are an informative visualization of the performance of

the system across the entire rank of results.


At least three specific query precision-recall curves should be discussed:


     - ​ The query with the highest **AP** (Average Precision).

     - ​ The query with the lowest **AP** .

     - ​ One additional query for comparison.


Following this analysis, the **mAP** (mean Average Precision) across all queries must be

calculated.


## **Phase 2: Factually Grounded RAG**

### **Reference Sentences**

In phase 1, you retrieved the candidate documents. Each document abstract is
composed of multiple sentences and, in this phase, you will devise a strategy to select
the sentence, or the span of sentences, that are better aligned with the query.


To achieve this goal you need to leverage a cross-encoder model, e.g. [MedCPT,](https://huggingface.co/ncbi/MedCPT-Cross-Encoder)
BioBert,, to estimate the relation between the query topic and the candidate reference
sentence(s). You should select the top 3 reference sentence(s), per article.

### **Exercise: Embeddings and Self-Attention**


In this exercise, you will examine the positional embeddings, the contextual embeddings
and self-attention as input embeddings go through the different layers. Implement the
following visualizations with a BERT model:


1.​ **Positional Embeddings.** Consider a text encoder. Insert a sequence of text with

the same word repeated 200 times. Visualize the embeddings of the 200 tokens:

a.​ (1) compute the distance of all tokens to the first token, (2) plot in 2d all


tokens with a color-code indexed to the distance.

b.​ Compute the distance from all tokens to all other tokens. Visualize in a


matrix.

2.​ **Contextual embeddings.** Visualize the contextual word embeddings from layer

0 to layer 11. Observe how they change from layer to layer.

3.​ **Self-Attention.** Examine the self-attention mechanism of a transformer


cross-encoder. Do critical analysis of your observations.


Discuss what you observe. To address the above questions, you should build on the
provided tutorials to visualize word embeddings and self-attention matrices.


See the examples below.


![proj_guide_NLPS_-_TREC_2025_BioGen_img_002.png](pdf_images\proj_guide_NLPS_-_TREC_2025_BioGen_img_002.png)

![proj_guide_NLPS_-_TREC_2025_BioGen_img_003.png](pdf_images\proj_guide_NLPS_-_TREC_2025_BioGen_img_003.png)
### **Answer Generation**

Given the biomedical topic (question) and the selected reference sentence(s), you will

            implement a LLM based strategy to generate answers that also have attributions (cited
references from PubMed) for each answer sentence.


The generated answer must meet the following requirements:


  - ​ The total length of the generated answer should be within 250 words.

  - ​ There should be no more than three PMIDs per answer sentence.

  - ​ The PMIDs must be selected only from the valid set of the provided PubMed


articles.

### **Evaluation with an LLM-Judge**


Generative models can generate information that is not repeated from the corpus dataset, but
that is still correct. In these cases, because it is too expensive to recruit human annotators for
every generated answer, it is common to use frontier LLMs as judges of the generated answers.

You will use GPT4o, available through IAedu API as your judge.

1.​ Login to IAedu: [https://iaedu.pt/pt](https://iaedu.pt/pt)
2.​ Scroll down to “Others” and locate and select OpenAI GPT4o.
3.​ Click on the settings wheel, “API info” and follow the instructions to access the API.



![proj_guide_NLPS_-_TREC_2025_BioGen_img_004.png](pdf_images\proj_guide_NLPS_-_TREC_2025_BioGen_img_004.png)
**Reference sentences alignment:** You will ask the model to judge each one of of the selected
sentences according to the following prompt criteria:



![proj_guide_NLPS_-_TREC_2025_BioGen_img_005.png](pdf_images\proj_guide_NLPS_-_TREC_2025_BioGen_img_005.png)



**Answer entailment:** You will ask the model to judge each one of the final answers. The
LLM-judge will assess the entailment criteria, i.e., if the generated answer is fully entailed by the

reference sentences:



![proj_guide_NLPS_-_TREC_2025_BioGen_img_006.png](pdf_images\proj_guide_NLPS_-_TREC_2025_BioGen_img_006.png)


The above prompts are uncallibrated and not specific to this domain. You should verify and
adapt the evaluation prompt to improve its precision in your domain. To achieve this, you can
select a sample of 5-10 examples and manually inspect the quality of the outputted evaluation
judgments.

## **Phase 3: Biomedical Deep Research Agent**


An agentic pattern like ReAct (Reason + Act) will be able to:


  - ​ **Plan:** Identify sub-topics (Weight loss, Side effects, Long-term outcomes).

  - ​ **Browse:** Execute specific Phase 1/2 cycles for each sub-topic.

  - ​ **Synthesize:** Aggregate conflicting evidence into a final, nuanced report.To be


defined.


More details will be provided.

## **Project Report**


There should be one single report that is written incrementally throughout each phase. It
should be structured in way that on each phase you add one more component and
evaluation to the overall system:


1.​ Introduction


2.​ BioGen NL Agent

a.​ Data parsing, indexing and search​ Phase 1
b.​ LLM Augmented Generation​ Phase 2
c.​ LLM Agentic patterns​ Phase 3

3.​ Evaluation


a.​ Experimental Setup: Datasets, Metrics and Protocols

b.​ Results and Discussion

4.​ Conclusion


a.​ Achievements

b.​ Limitations


## **Resources and References**

**OpenSearch Indexing and Similarity models​**
See the lab guides.

                         [https://docs.opensearch.org/latest/im](https://docs.opensearch.org/latest/im-plugin/similarity) plugin/similarity

**LLM and Embeddings Encoder API**
See lab guides.

**Dual Encoders**

[https://sbert.net/docs/quickstart.html](https://sbert.net/docs/quickstart.html)

**Frontier LLM as a Judge**
[https://iaedu.pt/pt](https://iaedu.pt/pt)

**Evaluation methods**

[https://github.com/AmenRa/ranx](https://github.com/AmenRa/ranx)

**TREC BioGen 2024**

[https://trec.nist.gov/pubs/trec33/papers/Overview_biogen.pdf](https://trec.nist.gov/pubs/trec33/papers/Overview_biogen.pdf)

**TREC BioGen 2025**

[https://arxiv.org/pdf/2603.21582](https://arxiv.org/pdf/2603.21582)

**HealthBench:**

[https://openai.com/index/healthbench/](https://openai.com/index/healthbench/)

**Medical QA:**

          Toward expert [level medical question answering with large language models | Nature Medicine](https://www.nature.com/articles/s41591-024-03423-7)


