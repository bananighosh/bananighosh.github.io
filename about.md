---
layout: default
title: About
permalink: /about/
---

<div class="about" markdown="1">

<div class="about-head">
  <img class="avatar" src="https://github.com/bananighosh.png" alt="Banani Ghosh" width="112" height="112">
  <div>
    <h1>Banani Ghosh</h1>
    <p class="tagline">ML Performance Engineer, AWS Trainium · Sunnyvale, CA</p>
  </div>
</div>

I like making ML systems fast. I work on distributed training and inference performance for AWS Trainium (Trn3 and Trn4). In vLLM, I added kernel auto-tuning that sped up a GPU inference path by up to 2.46x on NVIDIA B200, and I am building adaptive-k speculative decoding for MoE models.

<p class="profile-links"><a href="https://github.com/bananighosh">GitHub</a> · <a href="https://www.linkedin.com/in/banani-ghosh/">LinkedIn</a></p>

## Work

### ML Performance Engineer, AWS Trainium
<p class="meta">Amazon · Sunnyvale, CA · Current</p>

Distributed training and inference performance on Trn3 and Trn4.

### Software Development Engineer, Amazon Quick
<p class="meta">Amazon · Sunnyvale, CA · From Jun 2025</p>

Optimized agentic execution graphs for 500K+ users (25% fewer model invocations, 20% lower p95 latency, 2x end-to-end throughput) and built complexity-aware routing across model tiers.

### Open-source contributor, vLLM
<p class="meta">vLLM · Ongoing</p>

- Added config-driven auto-tuning and benchmark tooling to a GPU inference kernel path, improving it by up to 2.46x on NVIDIA B200.
- Utility-driven adaptive-k speculative decoding for MoE inference, selecting the draft-token budget per request to balance acceptance against verification cost, on NVIDIA B200 and GB10.

### Machine Learning Engineering Intern
<p class="meta">Root Access · New York, NY · Jan 2025 – May 2025</p>

Fine-tuned Llama 3 8B with QLoRA into an assistant for firmware engineers, with multimodal RAG across datasheets.

### Computer Scientist
<p class="meta">Adobe · Noida, India · May 2015 – Jul 2023</p>

- Led the SaaS/CaaS teams for Adobe software provisioning on AWS and Kubernetes, with GitOps CI/CD, onboarding 72% more microservices.
- Migrated Adobe software to AWS for an 80% performance improvement, handling 1 billion client queries in 60 seconds.
- Built infrastructure for Adobe's crash reporting solution to handle millions of crashes, improving its efficiency by 55%.

### Software Development Engineer
<p class="meta">Aricent · Gurugram, India · Jan 2014 – May 2015</p>

C and C++ development on a security gateway controller and a TCP performance-enhancing proxy.

## Education

### M.S. in Computer Engineering, New York University
<p class="meta">Aug 2023 – May 2025</p>

Coursework in High Performance Machine Learning, Computing Systems and Architecture, Machine Learning, Deep Learning, and Natural Language Processing.

## Talks and awards

- Speaker, Grace Hopper Celebration: Mining Broken User Journeys for Faster Crash Resolution (2017)
- Poster presenter, AI meets Science, New York: GenAI-powered therapeutic analysis (2024)
- Hardware winner, AMD Pervasive AI Developer Contest (2024)
- Winner, Adobe Summer Hackathon: on-premise GenAI chat assistant (2023)

</div>
