# LifeRAG 🎩

LifeRAG is a small personal project I built while learning more about RAG and AI applications.

The idea came from thinking about my CV. A normal CV is useful but there is only so much detail you can fit into a couple of pages. I thought it would be interesting to build a tool where people could ask questions about my experience instead.

You can ask about my career, projects, technical experience, leadership or even a few personal interests, and the app will try to find the most relevant information and answer from that.

## What it does

The app uses a small knowledge base made up of information about:

* My work experience
* Credit risk projects
* AI and automation work
* Leadership experience
* Technical skills
* Personal projects
* My CV and LinkedIn profile

When someone asks a question the app searches the knowledge base first and then sends the most relevant information to the language model.

The main reason for doing it this way is to stop the model from just making things up about me.

## The butler

For no particularly serious reason, I made the assistant respond like a slightly posh personal butler.

The personality only affects the tone. The model is still told to stick to the information in the knowledge base and not invent experience that is not there.

## Why I built it

This is really just a learning project.

It gave me a reason to work with RAG, embeddings, evaluations, vector databases, APIs, databases and deployment, while also ending up with something I can actually share.

It is definitely not meant to be a production grade AI system.

The goal was simply to build something myself, get it online and use it as a slightly more interesting way for people to explore my CV.


## A few example questions

You could ask things like:

* What did George do at X ?
* What has he built using AI?
* What is his credit risk experience?
* What kind of leader is he?
* How technical is he?
* What projects has he built outside work?
* What does he do outside work?

## How it works

At a very simple level:

```text
Knowledge base
      ↓
Split into chunks
      ↓
Create embeddings
      ↓
Store in Chroma
      ↓
Retrieve the most relevant chunks
      ↓
Send them to the OpenAI model
      ↓
Show the answer in Streamlit
```

Most of the knowledge base is written in Markdown, with my CV and LinkedIn profile included as PDFs.

For the Markdown files, I split the content by headings so that individual projects stay together where possible.

For example:

```text
Pliant
└── Credit Decision Engine
    ├── Problem
    ├── Solution
    └── Business Impact
```

This worked better than chopping everything into lots of small random chunks because the model gets more of the context around each project.

## Sources

I also wanted the app to show where the answer came from.

Next to the chat, it shows the sections of the knowledge base that were used, for example:

```text
Pliant — Credit Decision Engine
Klarna — Experimentation and Credit Bureau Strategy
Technical Skills — Applied AI
```

You can also expand the source and see a short section of the underlying text.

It is mainly there so someone using the app can see what the answer was based on.

## Tech stack

* Python
* Streamlit
* LangChain
* OpenAI API
* Chroma
* Hugging Face embeddings
* PostgreSQL
* SQLAlchemy
* Git / GitHub

The app is deployed on Streamlit Community Cloud.

## Project structure

```text
LifeRAG/
│
├── streamlit_app.py
│
├── implementation/
│   ├── answer.py
│   ├── ingest.py
│   ├── sources.py
│   └── db.py
│
├── GP_Knowledge_Base/
│   ├── pliant.md
│   ├── klarna.md
│   ├── stenn.md
│   ├── equifax.md
│   ├── leadership.md
│   ├── ai_projects.md
│   ├── technical_skills.md
│   ├── personal_projects_and_interests.md
│   └── CV/
│
├── requirements.txt
└── README.md
```

## Conversation logging

The app also saves conversations to PostgreSQL.

I added this because I was interested to see what people actually ask when they use it.

For each interaction I save things like the question, answer, retrieved context and session ID.

If the database connection fails, the chatbot still works normally.


## Running it locally

Clone the repo:

```bash
git clone https://github.com/Gpenster/LifeRAG.git
cd LifeRAG
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the requirements:

```bash
pip install -r requirements.txt
```

Add an OpenAI API key:

```text
OPENAI_API_KEY=your-key-here
```

Build the vector database:

```bash
python -m implementation.ingest
```

Then run the app:

```bash
streamlit run streamlit_app.py
```

## Updating the knowledge base

If I change any of the Markdown files or PDFs, I rebuild the vector database with:

```bash
python -m implementation.ingest
```

That reloads the documents and updates the embeddings in Chroma.

