**PPLN Task 1 - Teacher Recommendations**

For this part:


The initial recommendation is that our queries will be one of the fields above, and then, when
you get the hang of querying the index works, use all 3 (one idea is to combine all of those 3
fields in a single string - topic, question, narrative)

This part talks about how to split our data:


Split the Biogen2024topics.json dataset 50/50: 50% for training, 50% for test. Important:
when doing the split, do it based on queries only! Do not split the documents.
The ground truth is the file with which contains “submission” in the name.
Also, for the split, use odd query-topic pairs for train (1, 3, 5, …) and even for test (2, 4, 6,
…)

After you do this, you will need to create an OpenSearch index so you can query the

documents:



![prof_guide_professor_notes_img_001.png](pdf_images\prof_guide_professor_notes_img_001.png)

![prof_guide_professor_notes_img_002.png](pdf_images\prof_guide_professor_notes_img_002.png)
![prof_guide_professor_notes_img_003.png](pdf_images\prof_guide_professor_notes_img_003.png)

Note that there are 4 different retrieval strategies. #1, #2, #3 are ad-hoc BoW strategies,
while #4 uses an encoder to get the embeddings (vector-based strategy). One okay idea
would be to have an index for each of the strategies, however, this is not recommended!!!
Use one single OpenSearch index which contains all the strategies, and then we can query
the field for the method we’re testing! This can be done by setting different fields for different
strategies, and different similarity methods. This will facilitate our lives A LOT in the
development of the project.

Notes:


  - ​ Just index the contents of the documents.

  - ​ Each line in the JSONL file is a JSON. This can also be helpful to count the number

    - f examples by getting the number of lines in the document.

**For the evaluation part:**
In this project we’re not using binary relevance, we’re using graded relevance. In binary
relevance a document is either relevant or not (True/False), but in graded relevance there is
a scale, for example, [0, 5], where 0 represents a document that is not relevant at all for our
query, and 5 a document which is highly relevant, e.g.:
(q1, dk) -> 5
(q1, dj) -> 3
(q1, dl) -> 0
This makes us able to distinguish between relevance level, and it is used in NDCG (3rd
metric in evaluation); it also promotes having the most relevant closer to the top. The scale

- f the relevance must be chosen by us, it can either be [0,3], [0, 5], [0, 7], …, test different

values.


Also, still in this topic, for computing the precision (remember that it measures
accurate hits) we will need to define a threshold to mean if the document is either relevant or
not. For example, in a relevance scale of [0,5], the threshold can be either 2.5, or 1, or other
values… you must also test to find out which value is best
Note: do not compute different thresholds for different categories, since it will be unfair (não
percebi bem mas tem a ver com a computação da precisão, e médias, e coisas do género,
mas ele deu bastante ênfase em não fazermos isto)
We will do a PR for each query, can’t do a PR for multiple queries. So, 1 query -> 1
PR curve. We then use mAP to get the Mean Average Precision which kind of joins all the
curves together and computes the average. After getting the mAP display 2 or 3 (this
number is already enough) individual PR curves and analyse to see if they differ a lot from


the mAP one, and if anything looks weird. For example, if the mAP is good but the individual
PR curves you choose are like the ones below it is not a good sign (very bad sign actually):


**Final notes:** we have everything in the notebooks of labs 1 and 3!!! You just need to adapt it!
The only things which are not in those notebooks are: preparing the documents and setting

the threshold for the relevance scale.



![prof_guide_professor_notes_img_004.png](pdf_images\prof_guide_professor_notes_img_004.png)
