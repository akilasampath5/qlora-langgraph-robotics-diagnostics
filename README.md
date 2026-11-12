# Autonomous Industrial Diagnostics: Tool-Calling QLoRA + LangGraph Multi-Agent Orchestration

An end-to-end Machine Learning Engineering portfolio project demonstrating **parameter-efficient fine-tuning (4-bit QLoRA)** of a compact 3B language model for structured function calling, paired with a **stateful multi-agent LangGraph system** for robotic surface-preparation diagnostics.

---

## System Architecture

Rather than deploying three separate LLMs, this architecture uses **Single-Model Dynamic Adapter Swapping** on a single 16GB T4 GPU: one 4-bit quantized `Qwen2.5-3B-Instruct` backbone serves all three LangGraph agents, toggling the fine-tuned `tool_lora` adapter on exclusively for the Tool-Calling Specialist node.

```text
                       ┌──────────────────────────┐
                       │    Incident Alert Input   │
                       └─────────────┬────────────┘
                                     ▼
                       ┌──────────────────────────┐
                       │   1. Planner Agent       │
                       │ (Base Weights - Triage)  │
                       └─────────────┬────────────┘
                                     ▼
                  ┌────────────────────────────────────┐
                  │  2. LoRA Tool-Calling Specialist   │◄───┐
                  │   (tool_lora Adapter Active)       │    │ Bounded ReAct
                  └──────────────────┬─────────────────┘    │ Loop (max=4)
                                     │ <tool_call> JSON     │
                                     ▼                      │
                  ┌────────────────────────────────────┐    │
                  │  3. Deterministic Tool Executor    │────┘
                  │ (Actuator Telemetry & Maint. Logs) │
                  └──────────────────┬─────────────────┘
                                     │ No more tool calls
                                     ▼
                       ┌──────────────────────────┐
                       │   4. Reviewer Agent      │
                       │ (Base Weights - RCA)     │
                       └─────────────┬────────────┘
                                     ▼
                       ┌──────────────────────────┐
                       │  Root Cause & Action Plan │
                       └──────────────────────────┘
