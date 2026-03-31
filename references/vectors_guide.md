OpenSearch
Menu
Search for anything
Documentation
Vector search 
Getting started
Preparing vectors
Preparing vectors
In OpenSearch, you can either bring your own vectors or let OpenSearch generate them automatically from your data. Letting OpenSearch automatically generate your embeddings reduces data preprocessing effort at ingestion and search time.

Option 1: Bring your own raw vectors or generated embeddings
You already have pre-computed embeddings or raw vectors from external tools or services.

Ingestion: Ingest pregenerated embeddings directly into OpenSearch.

Pre-generated embeddings ingestion

Search: Perform vector search to find the vectors that are closest to a query vector.

Pre-generated embeddings search

Steps
Getting started with vector search

Use raw vectors or embeddings generated outside of OpenSearch

Option 2: Generate embeddings within OpenSearch
Use this option to let OpenSearch automatically generate vector embeddings from your data using a machine learning (ML) model.

Ingestion: You ingest plain data, and OpenSearch uses an ML model to generate embeddings dynamically.

Auto-generated embeddings ingestion

Search: At query time, OpenSearch uses the same ML model to convert your input data to embeddings, and these embeddings are used for vector search.

Auto-generated embeddings search

Steps
Generating embeddings automatically

Automatically convert data to embeddings within OpenSearch

Getting started with semantic and hybrid search

Learn how to implement semantic and hybrid search

OpenSearch Links
Get Involved
Code of Conduct
Forum
GitHub
Slack
Resources
About
Release Schedule
Maintenance Policy
FAQ
Testimonials
Trademark and Brand Policy
Privacy
Contact Us
Connect
Twitter
LinkedIn
YouTube
Meetup
Facebook
Copyright © OpenSearch Project a Series of LF Projects, LLC
For web site terms of use, trademark policy and other project policies please see https://lfprojects.org.