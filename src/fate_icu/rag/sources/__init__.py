"""Corpus sources: each module turns downloaded public records into ``Document`` objects.

eutils.py      NCBI E-utilities client: rate limit, retries, contact email
pubmed.py      PubMed searches (year-sliced under the 10,000-ID cap), cached records, XML parsing
jats.py        JATS/BITS XML (StatPearls articles, PMC guideline XML): titled sections of paragraphs
statpearls.py  the StatPearls archive from NCBI Bookshelf: resumable download, article parsing
guidelines.py  clinical guidelines from PDF (pypdf) or JATS XML
"""
