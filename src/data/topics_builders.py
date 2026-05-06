from src.data.query_builder import build_query
import spacy

nlp = spacy.load("en_core_sci_sm")

def build_topic_sentences(topics, retriever, ranker_model, corpus, limit_sentences_per_doc=True):
    topic_sentences_spacy = {}
    for topic in topics:
        query = build_query(topic, "topic+question")
        results = retriever.search(query, size=10)

        abstract_sentences = []
        for abstract_id, _ in results:
            abstract = corpus[abstract_id].replace("\n", " ").strip()
            doc = nlp(abstract)
            spacy_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            scores = ranker_model.score_query_vs_sentences(query, spacy_sentences)

            if limit_sentences_per_doc:
                top3 = sorted(scores, key=lambda pair: pair[1], reverse=True)[:3]
                abstract_sentences.extend((sentence, score, abstract_id) for sentence, score in top3)
            else:
                abstract_sentences.extend((sentence, score, abstract_id) for sentence, score in scores)

        abstract_sentences.sort(key=lambda pair: pair[1], reverse=True)

        topic_sentences_spacy[topic["id"]] = (topic, abstract_sentences[:10])

    return topic_sentences_spacy