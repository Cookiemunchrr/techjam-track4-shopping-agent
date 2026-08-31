# TikTok TechJam 2026 — Track 4 problem statement (organizer text)

> Transcribed verbatim from the organizer's launch statement, sections 4.3, 4.5 and 4.6
> plus the four core pillars. This is the authority for scope, deliverables and judging.
> Where this file and any working note in this repository disagree, **this file wins**.

## Challenge

Participants are challenged to architect an intelligent, next-generation shopping agent
capable of navigating real-world customer dynamics. Moving beyond rigid search filters,
the engineered system must demonstrate deep cognitive understanding, runtime
architectural agility, and commercial efficiency using the provided Amazon dataset.

Specifically, the system should be built upon the following four core pillars:

### I. Core Architecture: Intent Routing & Hybrid Pipeline

- **Dual-Track Routing:** Instantly detect the user's underlying intent — triggering a
  high-precision filter track for targeted "Buying" to lock hard constraints, and a
  diverse dense retrieval track for open-ended "Browsing" to unlock cross-category
  scenario matching.
- **Pipeline Base:** Construct an in-memory data stream featuring "Multi-Route Retrieval
  → LLM Semantic Ranking" (combining keyword, category, and vector similarity).

### II. Dialog Strategy: Multi-Turn Scenario Evolution

- **Dynamic State Machine:** Build a robust conversational state tracker to gracefully
  handle dynamic Information Accumulation (incremental slots) and abrupt Intent Override
  (slot erasure and rewriting).
- **Proactive Guidance:** Trigger an immediate retrieval cutoff when facing
  Over-Generality (candidate pool overload) to actively generate structured, proactive
  clarification prompts that guide user convergence.

### III. Self-Evolution: Dynamic Context Programming

- **Runtime Adaptation:** Leverage accumulated dialog history to perform Personalized
  Context Distillation, continuously updating short-term session states and long-term
  user profiles.
- **Adaptive Orchestration:** Utilize dynamic Context Programming to achieve runtime
  workflow re-orchestration and strategy alignment, ensuring the agent iteratively
  refines its own guidance logic.

### IV. Evaluation Matrix: Product & Efficiency Metrics

Anchored on the final purchased record within the Amazon dataset, performance is
quantified across three dimensions:

- **Coverage (Hit Rate@K):** Measures the catalog recall and boundary capability during
  the retrieval stage.
- **Precision (MRR / Top-K Hit Rate):** Evaluates the LLM's accuracy in pushing the exact
  purchased item to the absolute top of the recommendation list.
- **Efficiency (MTTC — Mean Turns to Conversion):** Heavily rewards systems that guide the
  user to the correct product in fewer interaction rounds, penalizing unnecessary
  conversational cognitive load.

## 4.3 Constraints & Scope

| Category | Constraints & Scope Details |
|---|---|
| **In scope** | Designing highly sensitive intent-detection modules to split traffic into "Buying" and "Browsing" tracks.<br><br>Implementing heterogeneous retrieval routing (weights, custom dynamic truncation, and slot decay over time).<br><br>Engineering runtime-adaptive memory layers for personalized context distillation.<br><br>Fine-tuning prompt strategies or local scoring logic for the LLM ranking stage to compress decision paths. |
| **Out of scope** | UI/UX Development (evaluated purely via automated backend APIs and headless pipelines).<br><br>Training or full-parameter fine-tuning of base foundational LLMs.<br><br>Deploying heavy external industrial vector DB clusters (must run entirely in-memory for light execution).<br><br>Multi-Modal Processing (restricted strictly to text catalogs, structured metadata, and text dialogs). |
| **Limits** | **Max Turns:** Hard limit of **10 turns** per session (forced termination and zero score if exceeded).<br><br>**Catalog Mutation:** The Amazon product dataset is strictly read-only; no structural mutations or mock ASIN injections are allowed. |
| **Allowed assumptions** | Inputs are pre-cleaned text strings (no need to account for spelling correction, typos, or ASR noise).<br><br>Product catalog, pricing, and category trees are static for the duration of the hackathon.<br><br>Each session is simulated as an isolated single-user interaction (no multi-user concurrency stress needed). |

## 4.5 Deliverables

**1. Written Project Description (via Devpost)** — a clear written description of the
project that includes:

- How your solution addresses the problem statement
- Development tools used (e.g. VSCode, Colab, Jupyter)
- APIs used (e.g. OpenAI GPT-4o, Google Maps API)
- Libraries and frameworks used (e.g. Hugging Face Transformers, PyTorch, scikit-learn, pandas)
- Datasets and assets used (e.g. Google Local Reviews dataset, manually labelled data)

**2. Public Code/GitHub Repository** — a link to a public repository containing:

- Well-structured, commented code covering all components of your solution
- A README file that includes:
  - Project overview
  - Setup and installation instructions
  - Steps to reproduce your results
  - A brief reflection on your solution's limitations and what you would improve given more time
  - Team member contributions (if applicable, i.e. team participants, non-solo participants)

> **Transcription note.** The source screenshot ends at "Team member contributions".
> If section 4.5 continues past that point (a demonstration video is the usual third
> item), transcribe it here before treating this list as complete.

## 4.6 Judging Criteria

| Judging Criteria | Definition | Weight |
|---|---|---|
| **Technical Execution** | The solution demonstrates strong engineering fundamentals, such as well-structured code, thoughtful architecture, and effective use of APIs or models. The demo runs reliably, and the technical complexity reflects deliberate, capable decision-making. | 35% |
| **Innovation & Problem Insight** | The project demonstrates originality in both idea and approach. It stands out for the sharpness of its problem understanding — how clearly the team has framed the challenge, why it matters, and how directly the solution addresses it. | 20% |
| **Impact & Relevance** | The project has clear potential to deliver value to real users or stakeholders — with meaningful reach, tangible benefit, and relevance that goes beyond solving for the hackathon prompt alone. | 20% |
| **Feasibility & Practicality** | The solution is realistic and buildable beyond a prototype. The approach is technically and operationally sustainable — resource usage is proportionate, the architecture holds under real-world conditions, and the implementation is grounded rather than speculative. | 15% |
| **Presentation & Communication** | **[Final Event Only]**: The team communicates their work with clarity. The pitch tells a coherent story; from problem to solution to potential, and the team is able to respond to questions with depth, demonstrating genuine understanding of their own project. | 10% |
