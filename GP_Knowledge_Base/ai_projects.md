AI Projects

I have used AI primarily to solve practical problems in credit risk and underwriting: reducing manual work, improving consistency, surfacing information that would otherwise be difficult to find, and helping risk teams make better decisions.

Credit Risk Knowledge Assistant (RAG)

Problem

Credit policies and standard operating procedures can be difficult to navigate, particularly for new joiners. Important guidance may be spread across several documents, and junior team members may hesitate to ask basic questions.

Solution

I built a Retrieval-Augmented Generation (RAG) knowledge assistant that allows the risk team to ask natural-language questions across credit risk policies and SOPs. The system retrieves the relevant source material before generating an answer, rather than relying on the language model's general knowledge.

Business impact

The objective was to make policy knowledge easier to access, reduce time spent searching through documentation, improve consistency in how policies are interpreted, and give new joiners a lower-friction way of finding answers.

Technical approach

Retrieval-Augmented Generation (RAG)

Vector search across policy and SOP documentation

Natural-language Q&A

Source-grounded responses

Automated Negative News Monitoring

Problem

Manual adverse media searches are difficult to scale across an SME portfolio across multiple countries and industries. We can easily miss relevant information when reporting appears only in local language or overseas news sources.

Solution

I built a daily negative news monitoring process using a web searching AI agent and a Google search API. The tool searched for both company and director information and could identify relevant local language news articles across the customer portfolio.

Business impact

The process expanded the coverage of adverse media monitoring beyond manual focused searches. It surfaced fraud related cases in overseas local-language reporting that would otherwise have been difficult for the team to identify manually, improving the quality of ongoing portfolio monitoring.

Technical approach

Automated daily portfolio searches

Google search API

Company and director-level search terms

Local-language news coverage

AI-assisted review of returned articles

Financial Statement OCR and Spreading Agent

Problem

Financial underwriting requires analysts to extract and structure relevant information from company financial statements. This is repetitive, time consuming work and creates a bottleneck when cases need to be reviewed at scale.

Solution

I worked on an OCR based financial spreading agent that extracts the relevant information from company financial statements and returns structured values together with a confidence measure.

Business impact

The aim was to reduce manual financial spreading, shorten the time required to prepare a case for underwriting, and create a foundation for more automated credit decisions. Confidence measures allow lower-confidence extractions to be routed for human review rather than blindly automated.

Technical approach

OCR / document extraction

Structured financial-data extraction

Confidence-based outputs

Human-in-the-loop review

Integration into automated underwriting workflows

In-Depth Review (IDR) Underwriting Tool

Problem

A credit analyst may need to inspect several different data sources during underwriting. Repeatedly collecting and checking the same information makes reviews slower and increases the risk that relevant information is overlooked.

Solution

I developed an in-depth review tool for the credit risk team that brings together the data available at underwriting and pre-populates the customer's case profile. The review includes checks such as company information, director addresses and the full credit-agency response.

Business impact

The tool reduces repetitive case preparation, gives analysts a more consistent starting point for each review, and allows more of their time to be spent on judgement and exceptions rather than assembling information manually.

Technical approach

Aggregation of multiple underwriting data sources

Automated case pre-population

Director and company-data checks

Credit-agency data integration

Analyst review workflow

LifeRAG — Interactive CV

Problem

A traditional CV necessarily compresses years of projects, decisions and experience into a small number of bullet points.

Solution

I built LifeRAG, an interactive RAG application that allows recruiters and hiring managers to ask questions about my experience, projects and technical work. The application retrieves relevant information from a curated professional knowledge base before generating a source grounded answer.

Business / portfolio value

The project demonstrates an end-to-end applied AI use case: defining a user problem, building a retrieval pipeline, creating a usable interface, deploying it publicly, and observing how real users interact with it.

Technical approach

Python

Streamlit

LangChain

Vector embeddings and similarity search

Chroma vector database

OpenAI language model

PostgreSQL interaction logging

Public Streamlit deployment