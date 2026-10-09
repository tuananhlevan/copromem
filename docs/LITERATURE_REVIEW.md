# Literature review supporting CoProMem

**Scope and evidence.** This narrative review covers 61 canonical papers dated 2022–2026 (38 CORE; 23 IMPORTANT). The catalog in `papers.csv` links each record to a primary publication or arXiv page. It is a targeted, reproducible scoping review rather than a PRISMA systematic review. Mechanism descriptions come from publication pages and, for a smaller set of pivotal papers, the papers' methods and related-work sections; numerical results below are reported results, not independent replications. Citation keys refer to `references.bib`.

## 1. Executive Summary

The central problem is **computational continuity**: a model call sees a bounded context, while an agent's obligations, environment, user relationship, and learned procedures extend beyond that call. The field has moved through four overlapping designs: (i) preserving conversation or action traces; (ii) selecting traces for the current context; (iii) consolidating traces into episodes, user profiles, world models, or skills; and (iv) controlling writes, reads, updates, and deletion as part of the agent policy. Generative Agents, Reflexion, Voyager, and MemGPT established different versions of this loop [park2023generative; shinn2023reflexion; wang2024voyager; packer2023memgpt].

The most consequential recent change is a move from passive archives toward **operational memory**. AWM and ReMe store reusable procedures rather than just facts [wang2025awm; cao2026reme]. CoEx and LedgerAgent keep task/world state separate from conversational history [kim2025coex; uddin2026ledgeragent]. Memory-R1 and AgeMem train or expose explicit management actions [yan2026memoryr1; yu2026agemem]. Benchmarks now test updating, forgetting, multimodal recall, and memory-guided action rather than isolated fact recall [wu2025longmemeval; hu2026memoryagentbench; uddin2026memora; shen2026mem2act; he2026memoryarena].

A strong conclusion from the comparative literature is that no representation alone solves continuity. Graphs can help relational and temporal questions, but retrieval granularity, indexing, maintenance, model choice, and evaluation protocol all affect results [hu2026graphs]. Memory can also harm an agent by propagating erroneous examples or narrowing exploration [xiong2026experience; zhao2026demystify]. These findings matter especially for a paper on procedural memory and handoffs: the unit of reuse, the trigger for writing, the validation of a stored procedure, and its effect on downstream action all need explicit treatment.

## 2. Conceptual Definitions

| Concept | Operational definition in this review | Distinction |
|---|---|---|
| Memory | Information retained beyond the current model call and made available for later use. | A storage and control capability; it may be text, graph, code, or learned token. |
| Context | Tokens or other inputs visible to the model in the current inference. | A retrieved memory becomes context when inserted into a call; a long context alone need not persist. |
| State | The current values of variables describing a task, world, user, execution, or agent. | State is about *what is true now*; memory can contain past or contradictory values. |
| Experience | A record of actions, observations, feedback, outcomes, or trajectories. | Becomes useful memory when selected or abstracted for later decisions. |
| Knowledge | Relatively stable facts, relations, or procedures abstracted from experience. | It may be stored externally or in parameters. |
| Profile | Persistent user or agent attributes, preferences, and persona. | Requires identity scope and revision when preferences change. |
| Scratchpad | Temporary reasoning or workspace within a task. | Usually transient and not automatically a durable memory. |
| Working memory | Active information maintained and manipulated over steps. | Typically task-local; in MemGPT-like systems it can be paged from longer storage. |
| Long-term memory | Information available across sessions or tasks. | Its horizon and access policy matter more than its name. |
| Agent state | Current plan, pending subgoals, tool outputs, constraints, commitments, and environment beliefs. | Can be implicit in a transcript or explicit in a ledger or belief state. |

Terminology varies. Reflexion calls a buffer of textual lessons *episodic memory*, while Voyager's executable skill library is closer to procedural memory [shinn2023reflexion; wang2024voyager]. MemGPT calls the active prompt *main context* and pages information from external tiers [packer2023memgpt]. CoEx models a world-facing belief state, whereas LedgerAgent holds a typed operational ledger [kim2025coex; uddin2026ledgeragent]. These are different state variables, even if each is rendered as text before an LLM call.

A useful layer model is: **parameters → latent state → inference context → working memory → persistent records → episodic trajectories → structured world/task state → cross-session persistence → continual adaptation**. TokMem occupies learned latent procedural state; ReAct mainly uses context and task-local trace; Generative Agents and Mem0 operate on external episodic/semantic stores; CoEx and LedgerAgent explicitly represent world/task state; AWM and ReasoningBank abstract experience for future tasks [wu2026tokmem; yao2023react; park2023generative; chhikara2025mem0; kim2025coex; uddin2026ledgeragent; wang2025awm; ouyang2026reasoningbank].

## 3. Taxonomy

The literature supports nine overlapping families. A paper can be in several families, but `papers.csv` assigns one primary category for counting. The storage–reflection–experience progression of a recent survey is useful background but does not replace the primary systems reviewed below [luo2026survey].

1. **Context and working-memory control:** paging, folding, evolving briefs, compressed or latent tokens [packer2023memgpt; kang2025memoryos; wang2025r3mem; wan2026compass; su2026ufold; wu2026tokmem].
2. **Episodic and autobiographical stores:** events, trajectories, narratives, reflections and temporal links [park2023generative; shinn2023reflexion; ong2025theanine; lu2026seem].
3. **Semantic and profile consolidation:** user portraits, persona summaries, cross-event abstraction and selective distillation [zhong2024memorybank; li2025ldagent; tan2025rmm; ma2026nemori; dai2026recmem; li2026timem].
4. **Procedural experience and skill reuse:** successful/failing trajectories become instructions, workflows, scripts, or latent procedure tokens [wang2024voyager; zhao2024expel; wang2025awm; ouyang2026reasoningbank; cao2026reme; sun2026magnet; wu2026tokmem].
5. **Structured and graph memory:** relation-, entity-, event-, temporal-, and causal-aware stores [xu2025amem; zhang2025associa; banerjee2026apex; jiang2026magma; hu2026evermemos; hu2026graphs].
6. **Explicit task/world/execution state:** belief states, plans, ledgers, typed memory operations and handoffs [yao2023react; zhang2024dynamic; kim2025coex; uddin2026ledgeragent; wang2026text2mem].
7. **Learned or delegated memory control:** management actions trained with RL, specialist memory agents, feedback-aligned selection [yan2026memoryr1; yu2026agemem; ma2026finemem; huang2026ama].
8. **Multi-agent and cognitive organization:** role workflows, shared or specialist memory, modular cognitive stores [li2023camel; hong2024metagpt; qian2024chatdev; li2026bmam; huang2026ama].
9. **Evaluation and critical comparison:** multi-session recall, update/forget, hallucination, action, multimodal memory and behavioral side effects [maharana2024locomo; wu2025longmemeval; chen2025halumem; hu2026memoryagentbench; shen2026mem2act; he2026memoryarena; xiong2026experience; zhao2026demystify].

## 4. Core Literature by Category

### 4.1 Retain and retrieve: early architectures

Multi-Session Chat established that conversation extending across meetings is materially different from short dialogue [xu2022goldfish]. Generative Agents records events, scores them for retrieval, and creates reflections that influence plans [park2023generative]. Reflexion instead stores verbal feedback after a trial and uses it in a later trial; its memory is outcome-conditioned rather than a general autobiography [shinn2023reflexion]. MemGPT made the context boundary explicit with agent-triggered paging [packer2023memgpt]. MemoryBank adapted this pattern to personal dialogue and time-based decay [zhong2024memorybank]. These papers frame write/read decisions, but generally do not enforce invariants on the quality or consistency of stored content.

### 4.2 Experience becomes procedure

Voyager's skill library stores executable code in a form that can be selected and reused, while ExpeL distills natural-language insights from accumulated traces [wang2024voyager; zhao2024expel]. MSI-Agent extracts insights at multiple scales, while CDMem indexes trajectory, task and cross-task abstractions in a graph [fu2024msi; gao2025cdmem]. AWM induces workflows from web trajectories, showing that the reusable unit can be a sequence of actions rather than a single fact [wang2025awm]. ReasoningBank distills reasoning from successes and failures, and ReMe adds validation and utility-based refinement to procedural memory [ouyang2026reasoningbank; cao2026reme]. MAGNET shows that task intent may survive GUI changes even when visual grounding must be updated [sun2026magnet]. TokMem encodes procedures as trainable tokens, shifting procedure storage toward the model interface [wu2026tokmem]. These systems differ in the **artifact** they preserve (code, natural-language insight, workflow, token), the **transfer horizon**, and whether they **revise** a procedure after failure.

### 4.3 Personal memory and temporal evolution

LD-Agent separates event summaries from user/agent personas; RMM adjusts memory granularity and retrieval through prospective and retrospective reflection [li2025ldagent; tan2025rmm]. THEANINE retains apparently obsolete memories because they may explain preference changes, linking events in temporal and causal timelines rather than deleting them [ong2025theanine]. MemoryOS uses explicit tier promotion; TiMem uses a temporal tree; EverMemOS consolidates episode cells into semantic scenes and reconstructs context from those scenes [kang2025memoryos; li2026timem; hu2026evermemos]. Nemori asks *which* experiences merit semantic distillation using predictability, while RecMem delays costly extraction until recurrence is observed [ma2026nemori; dai2026recmem]. These designs make a useful distinction: a stale **current-value** record should be superseded, but its historical event can remain available for temporal reasoning.

### 4.4 Graphs, event structure and conflict

A-Mem links notes and lets new additions revise old note attributes [xu2025amem]. Associa traverses an event-centric graph for multi-intent recall [zhang2025associa]. StructMem adds event bindings and periodic semantic consolidation; Chain-of-Memory constructs inference paths from retrieved fragments [xu2026structmem; xu2026chain]. APEX-MEM keeps append-only temporal events and resolves competing values at read time; MAGMA separates semantic, temporal, causal and entity edges; SEEM combines narrative episodes with relational facts and provenance pointers [banerjee2026apex; jiang2026magma; lu2026seem]. Hindsight explicitly separates facts, observations, experience and opinions [latimer2026hindsight]. A controlled comparison finds that graph structure cannot be credited for a gain without aligning indexing, ranking, maintenance and backbone settings [hu2026graphs]. Thus, structured memory is most compelling where queries require relationship, provenance or time, rather than as a universal replacement for simpler stores.

### 4.5 Context, task state and agent organization

ReAct establishes the action–observation trace, but that trace is an implicit and growing record of task state [yao2023react]. Dynamic Planning updates GUI plans from environmental feedback, and COMPASS assigns progress-brief management to a separate role [zhang2024dynamic; wan2026compass]. U-Fold keeps full history but supplies compact intent-aware summaries and tool logs during inference [su2026ufold]. LightMem delegates writing and retrieval to small models and moves consolidation offline, trading online memory work for preparation cost [zhang2026lightmem]. The distinction between *archive* and *working view* is critical: a lossy summary can support a current call yet must not become the sole source of truth if later goals require omitted facts.

CoEx maintains a neurosymbolic belief state for a changing world, while LedgerAgent puts current task facts and policy constraints into a typed ledger checked before tool execution [kim2025coex; uddin2026ledgeragent]. Text2Mem proposes typed, executable memory operations, making mutation itself inspectable [wang2026text2mem]. MetaGPT and ChatDev show role and artifact handoffs in multi-agent workflows; AMA goes further by assigning constructor, retriever, judge and refresher roles to a shared memory lifecycle [hong2024metagpt; qian2024chatdev; huang2026ama]. Their coordination state is related to but distinct from durable learned experience.

### 4.6 Learning and evaluating memory control

Memory-R1 learns ADD/UPDATE/DELETE/NOOP decisions; AgeMem places short- and long-term memory operations into the agent's action space; Fine-Mem aligns decisions with finer feedback [yan2026memoryr1; yu2026agemem; ma2026finemem]. The strongest evidence of their value must come from prospective tasks: retaining a fact is insufficient if the agent fails to use it correctly. LoCoMo and LongMemEval stress long-range dialogue and temporal/update questions; MemBench tests factual and reflective memory with capacity and efficiency dimensions; Mem-Gallery adds multimodal conversational memory [tan2025membench; bei2026memgallery]. HaluMem localizes write and update hallucinations; Memora penalizes obsolete memories; MemoryAgentBench tests selective forgetting; Mem2ActBench and MemoryArena require memory to guide actions [maharana2024locomo; wu2025longmemeval; chen2025halumem; uddin2026memora; hu2026memoryagentbench; shen2026mem2act; he2026memoryarena]. The latter two are especially relevant to procedural memory, because they join remembered constraints to downstream execution.

## 5. Architectural Comparison

| Paper | Year | Memory / state | Representation | Persistence | Retrieval | Update | Forgetting | Agent setting |
|---|---:|---|---|---|---|---|---|---|
| Generative Agents [park2023generative] | 2023 | Episodes + reflections | Text | Simulated days | Recency, relevance, importance | Append events | Implicit recency | Social simulation |
| Reflexion [shinn2023reflexion] | 2023 | Trial lessons | Text buffer | Repeated trials | Recent reflection | Append reflection | Buffer bound | Decision/coding agent |
| MemGPT [packer2023memgpt] | 2023 | Working + archival | Tiered text | Multi-session | Agent paging | Agent write | Tier management | Conversational agent |
| Voyager [wang2024voyager] | 2024 | Skills + environment | Executable code | Cross-task | Skill selection | Add/refine skill | No explicit policy | Minecraft |
| MemoryBank [zhong2024memorybank] | 2024 | User/profile memory | Text summaries | Multi-session | Relevant memory | Update user portrait | Time/importance decay | Companion dialogue |
| ExpeL [zhao2024expel] | 2024 | Experience + insight | Trace + text insight | Cross-task | Relevant precedent | Distill insight | No explicit policy | Task agents |
| A-Mem [xu2025amem] | 2025 | Linked notes | Note graph | Multi-session | Linked/semantic notes | Evolve note metadata | No explicit policy | LLM agent memory |
| AWM [wang2025awm] | 2025 | Workflows | Action routines | Cross-task | Workflow selection | Online/offline induction | No explicit policy | Web agents |
| THEANINE [ong2025theanine] | 2025 | Temporal episodes | Linked timeline | Multi-session | Timeline assembly | Link events | Retains old events | Dialogue agent |
| MemoryOS [kang2025memoryos] | 2025 | Short/mid/long-term | Tiered text | Multi-session | Tier retrieval | FIFO/page promotion | FIFO at short tier | Personal dialogue |
| CoEx [kim2025coex] | 2025 | World belief | Text + symbolic code | Cross-task | Belief-state lookup | Observation update | Revision | Exploration agent |
| ReMe [cao2026reme] | 2026 | Procedures | Distilled insight | Cross-task | Scenario-aware | Validate/refine | Utility pruning | Tool agents |
| ReasoningBank [ouyang2026reasoningbank] | 2026 | Success/failure reasoning | Text lessons | Cross-task | Relevant reasoning | Trajectory distillation | Not specified | Web/code agent |
| APEX-MEM [banerjee2026apex] | 2026 | Time-varying facts | Temporal property graph | Multi-session | Multi-tool read agent | Append-only events | Read-time resolution | Dialogue agent |
| MAGMA [jiang2026magma] | 2026 | Relations | Four graph views | Multi-session | Policy graph traversal | Graph insertion | Not specified | Memory QA agent |
| Memory-R1 [yan2026memoryr1] | 2026 | Managed records | External store | Multi-session | Learned selection | RL ADD/UPDATE | RL DELETE | Memory manager + answer agent |
| AgeMem [yu2026agemem] | 2026 | Working + long-term | Hybrid | Long task | Policy action | Policy action | Policy discard | Unified agent |
| LedgerAgent [uddin2026ledgeragent] | 2026 | Task/policy state | Typed ledger | Session | Ledger lookup | Tool-observation update | State revision | Customer-service tool agent |
| U-Fold [su2026ufold] | 2026 | Active context + tool log | Summary + log | Session | Intent-aware folding | Every turn | Archive retained | User-centric tool agent |
| AMA [huang2026ama] | 2026 | Shared memory | Hierarchical | Multi-session | Adaptive routing | Refresher repair | Outdated-entry removal | Multi-agent manager |

The table compares mechanisms, not benchmark scores. Paper-reported scores are not directly comparable because backbones, context budgets, judges, task mixes and online/offline memory construction vary [hu2026graphs]. `papers.csv` records the remaining architectures and the paper-level evidence available from sources.

## 6. Historical Development: 2021–2026

- **2021:** No paper met the architecture-focused inclusion threshold. This is a coverage fact for this search, not evidence that no relevant work exists.
- **2022:** Multi-session dialogue became an explicit benchmarked problem [xu2022goldfish].
- **2023:** ReAct established a task-local action trace; Generative Agents added observation, reflection and planning; Reflexion stored lessons; MemGPT formalized tiered context [yao2023react; park2023generative; shinn2023reflexion; packer2023memgpt].
- **2024:** Persistent user profiles, transferable skills and experiential insights gained clearer form; LoCoMo exposed long-range dialogue failures [zhong2024memorybank; wang2024voyager; zhao2024expel; maharana2024locomo].
- **2025:** Workflows, linked note networks, temporally organized dialogue, dynamic profiles and explicit belief state broadened the design space; LongMemEval made updates and abstention central [wang2025awm; xu2025amem; ong2025theanine; li2025ldagent; kim2025coex; wu2025longmemeval].
- **2026:** Systems increasingly formalized write/read/delete control, repair, provenance, structured state and action-based evaluation [yu2026agemem; yan2026memoryr1; cao2026reme; banerjee2026apex; uddin2026ledgeragent; hu2026memoryagentbench; shen2026mem2act; he2026memoryarena].

## 7. Major Architectural Patterns

The recurring pattern is a **write–organize–read–act–feedback loop**. Writes can be raw append (Generative Agents), selective extraction (Mem0), recurrence-triggered (RecMem), utility-scored (ReMe), or learned (Memory-R1/AgeMem) [park2023generative; chhikara2025mem0; dai2026recmem; cao2026reme; yan2026memoryr1; yu2026agemem]. Organization ranges from flat text/embeddings to temporal event graphs and semantic scenes [ong2025theanine; banerjee2026apex; hu2026evermemos]. Reads can be similarity-ranked, timeline-assembled, graph-traversed, policy-selected or controlled by a context manager [zhang2025associa; jiang2026magma; wan2026compass]. Feedback may revise content, retrieval policy, or the agent's future procedure [tan2025rmm; ouyang2026reasoningbank].

A second pattern is **dual representation**: preserve high-fidelity episodes while keeping a compact active view. This resolves part of the tension between cheap context and future recoverability [packer2023memgpt; su2026ufold; dai2026recmem]. A third is **explicit state outside the transcript**: a belief state or typed ledger supports constraints and recovery more directly than rereading a conversation [kim2025coex; uddin2026ledgeragent]. A fourth is **separation of facts from opinions and historical values from current values**, visible in temporal event stores and Hindsight's network split [banerjee2026apex; latimer2026hindsight].

## 8. Research Gaps and Answers to the Review Questions

1. **What counts as memory?** Any retained, later addressable information that can affect another decision; a transient hidden state alone counts only if carried across calls.
2. **Why is it different from more context?** Context is the current input; memory has persistence plus write, selection and update policies [packer2023memgpt; wu2025longmemeval].
3. **When does context management become memory?** When prior information is deliberately retained and later reconstructed across calls or sessions, as in tiered paging and U-Fold's archive/view split [packer2023memgpt; su2026ufold].
4. **What state is maintained?** Plans, subgoals, tool outputs, environment beliefs, task constraints, user profiles, skill indices, and role/workflow status [kim2025coex; li2025ldagent; uddin2026ledgeragent; hong2024metagpt].
5. **Where does state live?** Often implicitly in the transcript; more robust alternatives include a symbolic belief store, a typed ledger, or a workflow artifact [yao2023react; kim2025coex; uddin2026ledgeragent].
6. **What is remembered?** Heuristic salience and summaries remain common; recurrence, prediction error, feedback, utility and RL provide newer selection signals [park2023generative; dai2026recmem; ma2026nemori; cao2026reme; yu2026agemem].
7. **What is retrieved?** Most systems rank relevant text; timeline, graph and policy methods constrain retrieval by time, relation, intent or action need [ong2025theanine; jiang2026magma; su2026ufold].
8. **How are incorrect memories updated?** Explicit UPDATE/DELETE actions, targeted refresh, or append-only correction followed by read-time reconciliation [yan2026memoryr1; huang2026ama; banerjee2026apex].
9. **How do agents forget?** FIFO/decay, utility pruning, learned deletion or query-time supersession. Forgetting is usually heuristic, and historical evidence may need retention [zhong2024memorybank; cao2026reme; ong2025theanine; yan2026memoryr1].
10. **How is consolidation done?** Summarization and reflection, temporal hierarchy, recurring-pattern extraction or semantic scene formation [park2023generative; li2026timem; dai2026recmem; hu2026evermemos].
11. **How are conflicts resolved?** Few systems specify this; APEX-MEM resolves temporal facts at read time and AMA invokes a judge/refresher [banerjee2026apex; huang2026ama].
12. **How is time handled?** Timelines, timestamped event graphs and temporal query expansion; these are more defensible than assuming semantic similarity preserves chronology [ong2025theanine; banerjee2026apex; wu2025longmemeval].
13. **How are days to months supported?** External persistence plus user/entity identity, event linkage and state update; benchmarks simulate this, but real deployments remain scarce [ong2025theanine; uddin2026memora].
14. **What happens on restart?** Most papers do not test recovery. BMAM explicitly reports an export–clear–restore test; durable ledger/state serialization is still thinly evaluated [li2026bmam].
15. **How is user memory separated?** Dedicated profile stores or persona modules distinguish user attributes from general/agent knowledge, but identity and privacy handling vary [li2025ldagent; kang2025memoryos; hu2026evermemos].
16. **How do agents share memory?** Role handoffs and manager roles are common; formal consistency, access control and conflict ownership are less developed [hong2024metagpt; qian2024chatdev; huang2026ama].
17. **What dominates 2023–2026?** Retrieval-backed text stores, summaries/reflection, workflow/skill reuse, and increasingly graph/structured temporal stores and explicit memory actions.
18. **Which rely on vectors?** Mem0, A-Mem, MemoryOS and many dialogue systems use embeddings or vector ranking, often with additional structure [chhikara2025mem0; xu2025amem; kang2025memoryos].
19. **Which use structured state?** CoEx, LedgerAgent, APEX-MEM, MAGMA and Text2Mem are clear examples, though they structure different things [kim2025coex; uddin2026ledgeragent; banerjee2026apex; jiang2026magma; wang2026text2mem].
20. **Which learn management?** Memory-R1, AgeMem, Fine-Mem and RMM train or adapt aspects of management; most other write/read/delete rules remain hand-designed [yan2026memoryr1; yu2026agemem; ma2026finemem; tan2025rmm].

The most urgent gaps are **write correctness and provenance**, **temporal supersession without destroying history**, **memory-guided action**, **state serialization and restart**, **cross-agent consistency/access control**, and **causal evaluation of memory components**. HaluMem shows that errors arise during extraction and updating; Xiong et al. show they can be replayed; Memora penalizes stale memory; MemoryArena and Mem2ActBench expose the gap between answering from memory and acting from memory [chen2025halumem; xiong2026experience; uddin2026memora; he2026memoryarena; shen2026mem2act]. There is little evidence yet about weeks-long production reliability or privacy-preserving deletion across all copies of a memory.

## 9. Most Relevant Papers for Our Paper

For a paper on task-structured procedural memory and agent handoffs, the closest comparisons are **AWM, ReMe, ReasoningBank, ExpeL, Voyager, and MAGNET**: they ask what reusable experience should be stored and how it is selected on a new task [wang2025awm; cao2026reme; ouyang2026reasoningbank; zhao2024expel; wang2024voyager; sun2026magnet]. **CoEx, LedgerAgent and Text2Mem** clarify that a procedure should be distinguished from the current task state and from the operation that mutates memory [kim2025coex; uddin2026ledgeragent; wang2026text2mem]. **MemoryArena, Mem2ActBench, AppWorld-facing ReMe results and AWM's WebArena results** motivate measuring future task execution instead of recall alone [he2026memoryarena; shen2026mem2act; cao2026reme; wang2025awm]. **Xiong et al. and Zhao et al.** are essential counterpoints because they show that reuse can propagate errors or constrain search [xiong2026experience; zhao2026demystify].

The manuscript currently has a Related Work placeholder and a separate `references.bib` with preprint entries for ReasoningBank and ReMe. The canonical entries here use their verified ICLR 2026 and Findings of ACL 2026 versions. The new bibliography is separate so existing manuscript citation keys are not silently changed.
