# 12-Week AI Engineering Plan

Source roadmap: [louisfb01/start-ai-engineering](https://github.com/louisfb01/start-ai-engineering)
Started: 2026-11-07 · Target finish: 2027-01-30

**Profile:** working developer, has shipped an LLM feature · ~15 hrs/week · build-first · goal: AI engineering role · focus: agents + tools/MCP
**Weekly rhythm:** ~3h read/watch · ~10h build · ~2h evals + write-up (`log/week-NN.md`)
**Difficulty markers:** numbers (4–8) follow the roadmap's 1–10 scale.

## Setup (before week 1)

- [x] Pick ONE real corpus for the whole plan: **arXiv papers**
- [ ] API keys: **Anthropic** (chosen), plus Tavily or Exa · budget ~$30–60/month
- [x] Pick one coding agent (Claude Code / Cursor) and stick with it for the full 12 weeks
- [x] Buy: Chip Huyen, [*AI Engineering*](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) (reference throughout)
- [x] Join one community: [Towards AI Discord](https://discord.gg/YPsA3s3aw2) or [MLOps Community](https://mlops.community/)
- [x] Create a public GitHub remote for this repo

---

## Stage 1 — Reliable LLM components + evals (Weeks 1–2)

### Week 1 — Typed extractor → [`01-extractor/`](01-extractor/)
**Build**
- [x] Extract structured Pydantic objects from messy corpus documents
- [x] Retries on validation failure
- [x] Prompts versioned in files, not inline strings

**Resources**
- [ ] 4 · [AI Engineering Foundations webinar](https://youtu.be/ljOwBCdiHmg) — Bouchard
- [ ] 4 · [Here's how I use LLMs to help me write code](https://simonwillison.net/2025/Mar/11/using-llms-for-code/) — Willison
- [ ] 5 · [Instructor docs](https://python.useinstructor.com/)
- [ ] 6 · [Structured data extraction with LLM schemas](https://simonwillison.net/2025/Feb/28/llm-schemas/) — Willison
- [ ] 6 · [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — Karpathy (1.5×, spread over stage 1)

### Week 2 — Eval harness for the extractor
**Build**
- [ ] 50-example golden set
- [ ] Rule-based checks + LLM-as-judge, run via pytest
- [ ] Error analysis → fixes → record before/after score

**Resources**
- [ ] 5 · [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) + [LLM Evals FAQ](https://hamel.dev/blog/posts/evals-faq/) — Hamel Husain
- [ ] 5 · [Your job is to deliver code you have proven to work](https://simonwillison.net/2025/Dec/18/code-proven-to-work/) — Willison
- [ ] 6 · [Task-Specific LLM Evals](https://eugeneyan.com/writing/evals/) + [Evaluating LLM-Evaluators](https://eugeneyan.com/writing/llm-evaluators/) — Eugene Yan
- [ ] Chip Huyen book: evaluation chapters

### ✅ Stage 1 checkpoint
- [ ] I can show a metric that moved from X to Y because of a specific prompt/schema change, and explain why

---

## Stage 2 — RAG done properly (Weeks 3–5)

### Week 3 — Baseline cited Q&A → [`02-rag/`](02-rag/)
**Build**
- [ ] Retrieval eval set (question → expected chunks) written BEFORE tuning
- [ ] Baseline Q&A with citations on LanceDB or Qdrant

**Resources**
- [ ] 4 · [Why RAG Is Not Training Your AI](https://www.louisbouchard.ai/why-rag-is-not-training-your-ai/)
- [ ] 4 · [LlamaIndex Introduction to RAG](https://developers.llamaindex.ai/python/framework/understanding/rag/)
- [ ] 4 · [Qdrant docs](https://qdrant.tech/documentation/) or [LanceDB docs](https://lancedb.github.io/lancedb/)

### Week 4 — Measured retrieval upgrades
**Build**
- [ ] BM25 hybrid search
- [ ] Reranker
- [ ] Contextual chunking
- [ ] Ablation table: recall@k + answer quality per variant
- [ ] Ragas wired in

**Resources**
- [ ] 4 · [Cohere Embed and Rerank](https://docs.cohere.com/docs/embeddings)
- [ ] 5 · [Ragas](https://docs.ragas.io/en/stable/)
- [ ] 5 · [Is RAG Still Needed in the Era of Long Context LLMs?](https://pub.towardsai.net/is-rag-still-needed-in-the-era-of-long-context-llms-3d89907ce624)
- [ ] 6 · [Hybrid Search RAG That Actually Works](https://pub.towardsai.net/hybrid-search-rag-that-actually-works-bm25-vectors-reranking-in-python-0c02ade0799d)
- [ ] 6 · [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — Anthropic

### Week 5 — Error-driven fixes + design doc
**Build**
- [ ] Error analysis on failed questions
- [ ] Query rewriting + metadata filters
- [ ] One-page design doc (`02-rag/DESIGN.md`)

**Resources**
- [ ] 5 · [Context engineering](https://simonwillison.net/2025/Jun/27/context-engineering/) + [How to Fix Your Context](https://simonwillison.net/2025/Jun/29/how-to-fix-your-context/) — Willison
- [ ] 6 · [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — Anthropic
- [ ] 7 · [Systematically Improving RAG](https://jxnl.co/writing/2024/05/22/systematically-improving-your-rag/) — Jason Liu
- [ ] 7 · [Context Engineering for Production-Ready RAG](https://pub.towardsai.net/context-engineering-4a17018c41cf)

### ✅ Stage 2 checkpoint — portfolio piece #1
- [ ] README has the ablation table
- [ ] 3 failures I fixed (and how) + 1 I deliberately didn't fix (and why)

---

## Stage 3 — Tools, MCP, agents (Weeks 6–9)

### Week 6 — Agent loop from scratch → [`03-agent-mcp/`](03-agent-mcp/)
**Build**
- [ ] Raw tool calling, no framework, 3–4 tools (RAG search, web search, calculator/SQL)
- [ ] Step + token budget, retries, structured tool errors returned to the model

**Resources**
- [ ] 5 · [Tool use overview](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview) — Anthropic
- [ ] 6 · [Building Effective AI Agents](https://www.anthropic.com/research/building-effective-agents) — Anthropic
- [ ] 7 · [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) — Anthropic
- [ ] 7 · [Creating an Advanced AI Agent From Scratch with Python in 2026](https://pub.towardsai.net/creating-an-advanced-ai-agent-from-scratch-with-python-in-2026-part-1-ce74a23f6514)

### Week 7 — RAG system as an MCP server
**Build**
- [ ] MCP server exposing the stage-2 RAG system
- [ ] Used from Claude Code/Desktop
- [ ] Own minimal MCP client

**Resources**
- [ ] 5 · [Hugging Face MCP Course](https://huggingface.co/learn/mcp-course/) or [Anthropic Intro to MCP](https://anthropic.skilljar.com/introduction-to-model-context-protocol)
- [ ] 6 · MCP docs: [architecture](https://modelcontextprotocol.io/docs/learn/architecture), [build a server](https://modelcontextprotocol.io/docs/develop/build-server), [build a client](https://modelcontextprotocol.io/docs/develop/build-client)
- [ ] 7 · [Code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp) — Anthropic
- [ ] 8 · [MCP has prompt injection security problems](https://simonwillison.net/2025/Apr/9/mcp-prompt-injection/) — Willison

### Week 8 — Research assistant in LangGraph
**Build**
- [ ] Plan → search (Tavily/Exa) → read → cited brief
- [ ] Checkpointing + human-in-the-loop approval
- [ ] Written answer: which parts should be a fixed workflow instead?

**Resources**
- [ ] 5 · [AI Agents in LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) — DeepLearning.AI
- [ ] 6 · [Agents and Workflows](https://www.louisbouchard.ai/agents-and-workflows/) + [Agent Architecture Guide](https://github.com/louisfb01/ai-engineering-cheatsheets/blob/main/Agent_Architecture_Guide.md) — Bouchard
- [ ] 7 · [Agents](https://huyenchip.com/2025/01/07/agents.html) — Chip Huyen
- [ ] 7 · [12-Factor Agents](https://github.com/humanlayer/12-factor-agents)
- [ ] Optional 6 · [Designing Real-World AI Agents Workshop](https://github.com/iusztinpaul/designing-real-world-ai-agents-workshop) (`implement_yourself/` path)

### Week 9 — Observability + agent evals
**Build**
- [ ] Tracing (Phoenix / LangSmith / Logfire)
- [ ] Trajectory evals (right tools, right order)
- [ ] Cost + latency dashboard

**Resources**
- [ ] 5 · [Arize Phoenix](https://arize.com/docs/phoenix) or [Logfire](https://logfire.pydantic.dev/docs/)
- [ ] 6 · [Harness Engineering](https://www.louisbouchard.ai/harness-engineering/) — Bouchard
- [ ] 7 · [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) — Anthropic
- [ ] 7 · [Agent Observability and Evaluation: 2026 Guide](https://pub.towardsai.net/agent-observability-and-evaluation-a-2026-developers-guide-to-building-reliable-ai-agents-f4547e4beb14)
- [ ] 8 · [How we built our multi-agent research system](https://www.anthropic.com/engineering/built-multi-agent-research-system) — Anthropic

### ✅ Stage 3 checkpoint — portfolio piece #2
- [ ] For any bad run I can open the trace and point to the failing step
- [ ] An eval catches that failure if it comes back

---

## Stage 4 — Security, production, hiring (Weeks 10–12)

### Week 10 — Red-team your agent → [`04-capstone/`](04-capstone/)
**Build**
- [ ] Prompt injections hidden in retrieved docs and web pages
- [ ] Permission tiers, output validation, human approval for risky actions
- [ ] Threat model (`04-capstone/THREAT_MODEL.md`)

**Resources**
- [ ] 6 · [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/)
- [ ] 6 · [Red Teaming LLM Applications](https://www.deeplearning.ai/short-courses/red-teaming-llm-applications/) — DeepLearning.AI
- [ ] 7 · [The Lethal Trifecta for AI Agents](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/) — Willison
- [ ] 8 · [Design Patterns for Securing LLM Agents against Prompt Injections](https://simonwillison.net/2025/Jun/13/prompt-injection-design-patterns/)

### Week 11 — Deploy
**Build**
- [ ] FastAPI service, containerized, on Modal (or similar)
- [ ] LiteLLM provider fallback
- [ ] Prompt caching + rate limits
- [ ] Evals run in CI on every PR

**Resources**
- [ ] 5 · [LiteLLM docs](https://docs.litellm.ai/)
- [ ] 5 · [Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — Anthropic
- [ ] 5 · [Modal docs](https://modal.com/docs/guide)
- [ ] 5 · [Automated Testing for LLMOps](https://www.deeplearning.ai/short-courses/automated-testing-llmops/) — DeepLearning.AI
- [ ] 7 · [Serverless vs. self-hosted inference](https://bentoml.com/llm/llm-inference-basics/serverless-vs-self-hosted-llm-inference) — BentoML

### Week 12 — Polish + job push
**Build**
- [ ] 3-minute demo video
- [ ] Architecture write-up + cost-per-query breakdown
- [ ] Rehearse the "questions you should be able to answer" lists from the roadmap

**Resources**
- [ ] [What I Look For When Hiring AI Engineers](https://youtu.be/D89pj9cqUm4) — Bouchard
- [ ] Roadmap section: [How to find an AI engineering job](https://github.com/louisfb01/start-ai-engineering#findajob)
- [ ] 6 · [A Field Guide to Rapidly Improving AI Products](https://hamel.dev/blog/posts/field-guide/) — Hamel Husain

### ✅ Final checkpoint
For the capstone I can answer cleanly:
- [ ] Why this architecture (and workflow vs agent)?
- [ ] Where and how does it fail?
- [ ] How is it evaluated, offline and online?
- [ ] What gets logged, and how do I debug a bad output?
- [ ] What does it cost per query, and is it the cheapest design that clears the bar?
- [ ] Why is it safe enough for real users? What happens when a dependency fails?

---

## Ongoing
- Weekly: ~30 min of [Latent Space](https://www.latent.space/) + [Hamel's blog](https://hamel.dev/)
- Every Friday: run the six questions above against the current project; anything you can't answer becomes next week's work
- Post each stage write-up to your community for feedback

## After week 12 (optional)
- [ ] Fine-tuning: [HF smol course](https://huggingface.co/learn/smol-course/unit1/3) → [Philipp Schmid's guide](https://www.philschmid.de/fine-tune-llms-in-2025)
- [ ] Multimodal: [Docling](https://docling-project.github.io/docling/) + [Document AI course](https://www.deeplearning.ai/short-courses/document-ai-from-ocr-to-agentic-doc-extraction/) → receipt/invoice parser as an agent tool
