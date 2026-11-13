"""Stage 5: the RAG layer (public data only).

Patient facts come from the evidence packet; general medical knowledge comes from passages
retrieved here and cited as R1..Rn. Everything is built from public text: PubMed abstracts
(NCBI E-utilities), StatPearls (NCBI Bookshelf) and clinical guidelines.

Library
    text.py        text normalisation shared by every source parser
    corpus.py      Document, Section and Passage; the Parquet corpus and its content hash
    drivers.py     the evidence keys a driver can name (channels, static groups, image concepts)
    sources/       PubMed (E-utilities), StatPearls (Bookshelf archive), guidelines (PDF/JATS)
    chunking.py    sentence splitting and balanced chunks of at most 300 tokens within a section
    bm25.py        BM25 (bm25s) with a deterministic top k
    encoders.py    MedCPT query and article encoders and cross-encoder at pinned revisions
    dense.py       resumable passage embeddings and the exact FAISS inner-product index
    fusion.py      reciprocal-rank fusion and cross-encoder reranking
    queries.py     the fixed query template over driver labels (no patient values)
    query_bank.py  building a query bank, and checking one built on another machine
    retrieve.py    the query-bank schema and ``retrieve(packet) -> passages`` (deterministic)
    evaluate.py    blind labelling sheet, precision@3 and nDCG@5, paired tests
    config.py, version.py, embed.py

Commands (docs/lab_runbook.md, Stage 5)
    fetch          download PubMed records and the StatPearls archive (laptop)
    build_corpus   parse and chunk every source into the passage corpus (laptop)
    build_index    BM25, then check and install the GPU output (laptop)
    gpu_bundle     collect what the GPU machine needs (laptop)
    gpu_step       embeddings, FAISS and the query bank (Kaggle/Colab GPU; laptop CPU for smoke)
    verify         re-check every hash of a copied index (lab)
    label          grade the labelling sheet in a terminal (you)
    build_all      every step on this machine (smoke config)
"""
