## 5 Post-Training

### 5.1 Post-Training Pipeline

In this release, we refrain from introducing novel post-training algorithms. The overall recipe follows the standard paradigm of supervised fine-tuning (SFT) followed by reinforcement learning (RL) and on-policy distillation (OPD; [Gu et al., 2024](#bib.bib51); [Lu and Lab, 2025](#bib.bib22)), without algorithmic modifications beyond well-established practices. Instead, our efforts are concentrated almost entirely on *what* the model is trained on rather than *how* it is optimized: we invest in large-scale, automated pipelines for data synthesis and environment construction. Concretely, the pipeline (i) synthesizes diverse, verifiable training tasks together with their reference solutions and reward signals, (ii) procedurally constructs and scales interactive agent environments in which trajectories can be collected and evaluated at low cost, and (iii) applies rigorous filtering, deduplication, and difficulty calibration to ensure data quality and curriculum balance. We find that, under a fixed and unremarkable optimization procedure, systematic improvements in the scale, diversity, and verifiability of synthesized data and environments account for essentially all of the observed gains. This observation echoes a broader lesson: at the current stage, the marginal return of engineering the data and environment pipeline substantially exceeds that of algorithmic novelty in post-training.

#### 5.1.1 Large-Scale Agent Task Synthesis

Tasks serve as the fundamental fuel for agent learning. However, constructing high-quality training tasks has traditionally required substantial manual effort. We observe that the model is already beginning to exhibit the ability to construct its own training tasks, though this capability remains far from perfect. Recognizing this potential, we have invested considerable effort in strengthening the model's task-construction and quality-verification abilities.

We formalize each task as a triplet (problem, environment, verification system) and evaluate its quality along two dimensions: difficulty---ensuring the task is non-trivial---and correctness---guaranteeing that no critical flaws exist among the three components. Using difficulty and correctness as reward signals, we iteratively train the model to construct better tasks. We also monitor RL tasks across their full lifecycle. Whenever a task is used in a new RL run, the resulting trajectories provide fresh evidence for quality re-auditing. Under this general framework, we have built dedicated training environment production pipelines for two core scenarios: general agents and coding agents.

General Agent. For general agents, we encourage internal employees and external partners to incorporate our latest model into their routine workflows and, on a voluntary basis, return interaction data and feedback. Based on the interfaces observed in the returned data, we construct a large set of mocked tools that reproduce the interfaces and behaviors of real-world tools and systems, including their input formats, output structures, API schemas, and behavioral constraints, covering both commonly used SaaS and enterprise applications as well as more specialized business back-end systems. In parallel, we collect negative feedback and model failure cases submitted by internal employees at scale, and incorporate them into the pipeline to generate both single-turn and multi-turn agent environments grounded in real workflows. By reconstructing the relevant tool context, user interaction patterns, and failure conditions, the pipeline enables systematic replay of failures and targeted reinforcement learning against observed model weaknesses.

Coding Agent. Coding agent training environments are built from two sources: 1) coding-agent sessions from internal employees and external partners, filtered to retain highly complex tasks or tasks on which model performance is poor, then deduplicated by trajectory; and 2) public GitHub repositories that meet a star-count threshold. Environment construction is carried out collaboratively by multiple specialized agents. First, an agent determines whether the project can be built and fully run inside a container and whether it can be automatically verified; if so, it selects a specific turn or commit as the task starting point, designs several sufficiently complex implementation directions, and produces concrete evaluation points, including both fail-to-pass and pass-to-pass points, along with a construction report, fetching external resources from the web as needed. Next, a separate agent sets up dependencies, the initial working directory, test code, and task descriptions in an isolated container, performs self-testing, removes any traces that could leak the task solution, and packages the environment as a new image layer. Then, multiple distinct agents attempt the task, and an independent quality-inspection agent reviews the environment together with the solving agents' trajectories, checking for environment issues, factual errors, mismatches between evaluation points and task descriptions, and hackability risks. If the inspection does not pass, a repair agent fixes all identified errors, adjusts evaluation points that are too easy or too difficult, and the task re-enters verification.

Through these pipelines, we can automatically and batch-produce RL training data that is correct, discriminative, and controllable in length and difficulty. Whether through the faithful reconstruction of real workflows in general agent environments or the precise construction of coding tasks in coding agent environments, both ultimately feed into a unified training system, driving iterative improvement of model capabilities under continuous quality monitoring.

#### 5.1.2 RL in Synthesized Tasks

CAPTION:: Figure 7: Performance improves on various code agent benchmarks as RL training scales in the Minimal mode of DeepSeek Harness. Further extending maximal context length to 1M tokens continues to improve performance on extremely long-horizon tasks, e.g., Terminal-Bench v3.0.

CAPTION:: Figure 8: Performance improves with cumulative RL steps when jointly training across multiple versions of Claude Code (left) and across heterogeneous scaffolds, including OpenCode, Pi, and DeepSeek Harness in Standard and PTC modes (right). Performance is evaluated on DeepSWE v1.1. Lighter curves show the evaluation of individual scaffold versions or scaffolds.

We use large-scale asynchronous RL in synthesized tasks to improve model performance and shape its behavior in complicated scenarios. We scale RL runs in two dimensions, i.e., training compute and the number of scaffolds. As shown in Figure [7](#S5.F7) and Figure [8](#S5.F8), performance continues to improve as we increase compute with cumulative RL steps, whether scaling within a single scaffold, jointly across variants of the same scaffold, or across heterogeneous scaffolds.

For RL training across diverse scaffolds, we decouple agent rollout execution into an agent sandbox and a worker container. The sandbox runs the scaffold and its tools, while the worker provides a scaffold-agnostic control layer that orchestrates the rollout, normalizes heterogeneous interactions into a common trajectory schema, and communicates with the trainer. Both run on DSec (Section [5.1.3](#S5.SS1.SSS3)), outside the preemptible GPU training pool, separating long-lived rollouts from fine-grained training scheduling. During trainer preemption, rollout execution can be suspended and offloaded while preserving its full state for later resumption and releasing CPU and GPU resources. This design enables stable and efficient RL across heterogeneous scaffolds without modifying the underlying algorithms.

To extend effective RL compute beyond a single training run, we use model merging to reinitialize successive RL runs. Specifically, we merge checkpoints from runs across different scaffolds or configurations, combining improvements acquired along different optimization paths. In Figure [7](#S5.F7) and Figure [8](#S5.F8), disconnected curve segments reflect successive RL runs after model reinitialization. This yields further gains in both task performance and token efficiency, providing a simple and practical way to aggregate parallel RL compute and continue scaling across successive runs.

#### 5.1.3 Running Agents at Massive Scale: DSec

As we transitioned from DeepSeek-V3 to V4, the rapidly growing number and diversity of agentic training environments motivated us to build DeepSeek Elastic Compute (DSec), a production-grade sandbox platform for large-scale agentic training and evaluation. Its initial design addressed heterogeneous execution environments, scalable image distribution, multiple isolation backends, high-density resource management, command trajectory logging, and preemption-safe resumption.

V4.1 training further increased demand to millions of concurrent sandbox instances spanning diverse harnesses, platforms, code repositories, software dependencies, and task-specific services. At this scale, the primary bottlenecks shifted toward datacenter scalability, workload isolation, per-node compute density, and the containment of misbehavior by increasingly capable agents. We briefly describe our key design aspects below.

Horizontally Scaling Compute at Scale. DSec scales through two complementary mechanisms: sharding and scheduling with relaxed consistency. To accommodate a large number of machines, we partition compute nodes into multiple shards (so-called scale units). Such sharding also allows us to reduce blast radius by isolating workloads from different experiments, preventing a single memory-intensive task from exhausting resources shared by unrelated work.

Instead of using off-the-shelf orchestrators like Kubernetes, DSec employs a custom placement engine to schedule the large number of sandboxes, by trading strong global consistency for scalability. This design is based on a key observation: agentic sandbox placements only demand eventual consistency as long as each compute node enforces local safety constraints. To do so, the placement engine deploys in multiple independent replicas, without synchronized coordination. Each replica predicts resource availability from recent measurements and makes good-enough placement decisions. To compensate for this loss in consistency, each node is responsible for validating the final placement decisions, enforcing a hard admission constraint that rejects new placements if it exceeds a local warning threshold. This design as a whole allows DSec to scale to millions of containers without bottlenecking on central coordination.

Running Sandboxes at High Density. At the node level, we use hardware-supported sub-NUMA partitioning and bind each worker VM to an individual NUMA domain. Containers run within these worker VMs, with their CPU and memory allocations confined to the VM's local NUMA resources. This setup balances aggressive memory overcommitment against Linux kernel lock contention, while localizing memory pressure and runtime failures. Under comparable workload configurations, it increases the supported density from roughly 1,000 to more than 2,500 concurrent live containers per physical node before measurable end-to-end degradation appears.

Such high-density deployment can nevertheless distort time-sensitive evaluations through interference from background workloads. DSec therefore introduces a latency-sensitive (LS) execution class. We apply SCHED_IDLE to non-LS tasks to minimize their scheduling priority, and use core scheduling to ensure only tasks of the same priority class execute simultaneously on sibling hyperthreads to eliminate interference.

Mitigation of Misbehaving Agents. During RL training, we frequently observe agents attempting to perform reward hacking or unintentionally crashing the environment. In certain attempts, our agents exploited recently disclosed vulnerabilities, including permission issues from the XFS driver, illegal memory access in AppArmor, leaking answers from package mirror services, and so on. Agents have also been notorious for deleting critical binaries, breaking system files, or even removing the filesystem. We use per-sandbox AppArmor profiles and fine-grained eBPF-based network policies to prevent such attempts. If an agent crashes its environment, we treat the crash as a failed trajectory and report a \"repercussion\" signal to the RL framework.

#### 5.1.4 Controllable Reasoning Effort in RL

Alongside advances in model architecture and hardware, the number of output tokens is another key determinant of serving cost and, consequently, of the cost--quality trade-off in real-world applications. We therefore introduce a scalar effort level $b$ as an explicit conditioning signal during reinforcement-learning. This mechanism is applied to both single-turn reasoning and multi-turn agentic tasks. Specifically, we prepend the following instruction to the system prompt:

Reasoning Effort: {effort} (range 1\--100; higher values request more thorough reasoning)

Here, $b\in\{1,\ldots,100\}$ denotes the requested effort level.

For each training prompt $x$, we sample $M_{b}$ responses at each effort level $b\in\mathcal{B}$:

  -- ------------------------------------------------------------------------------------------- -- -----
     $$z_{b,j}\sim\pi_{\theta}(\cdot\mid x,b),\qquad b\in\mathcal{B},\quad j=1,\ldots,M_{b}.$$      (8)
  -- ------------------------------------------------------------------------------------------- -- -----

Here, $j$ indexes the responses sampled at effort level $b$. Responses sharing the same $(x,b)$ form a subgroup, within which rewards are mean-centered to compute group-relative advantages. Thus, responses from different effort levels are not directly compared. Instead, effort-dependent behavior is induced within each subgroup by making the length component of the reward depend on $b$. Specifically, we add the length-penalty term $r_{b,j}^{\mathrm{len}}$ to the reward of response $z_{b,j}$:

  -- ----------------------------------------------------------------------------------------------------- -- -----
     $$r_{b,j}^{\mathrm{len}}=-\min\left\{C_{\max},\;k(b)\frac{\ell_{b,j}}{L_{\mathrm{norm}}}\right\},$$      (9)
  -- ----------------------------------------------------------------------------------------------------- -- -----

where $\ell_{b,j}$ is the number of reasoning tokens, $L_{\mathrm{norm}}$ is a reference length, and $C_{\max}$ caps the maximum deduction applied to a trajectory. The token-penalty coefficient decreases exponentially with the requested effort:

  -- ------------------------------------------------------------------------------------------------ -- ------
     $$k(b)=k_{0}\exp\left(-\frac{b-b_{\min}}{\tau}\right),\qquad\tau=\lambda\overline{\Delta b},$$      (10)
  -- ------------------------------------------------------------------------------------------------ -- ------

where $k_{0}$ is the basic penalty coefficient at different effort levels, $b_{min}$ is the minimum value of $\mathcal{B}$, $\overline{\Delta b}$ is the average spacing between training effort levels, and $\lambda$ controls the rate of penalty decay. Increasing $b$ by $\tau$ multiplies the penalty coefficient by $e^{-1}$. The parameter $k_{0}$ controls the overall pressure toward shorter reasoning, whereas a smaller $\tau$ causes the penalty to decay more rapidly and tends to produce greater behavioral separation between effort levels. Appendix [C](#A3) provides a marginal-utility motivation for the exponential parameterization of $k(b)$.

At deployment time, the scalar $b$ provides a flexible control interface over the model's reasoning strength. By varying $b$, a single model checkpoint can move between different operating regimes along the learned cost--quality frontier, adapting to different latency, token-budget, and solution-quality requirements. Although training uses only a finite set of effort levels, intermediate values can be used at deployment to elicit interpolated reasoning behaviors, providing a fine-grained and efficient mechanism for test-time resource allocation.

In our production deployment launched in September 2026, the public API exposes three preset reasoning-effort tiers---max, high, and low---which map directly onto this scalar interface. As summarized in Table [2](#S5.T2), the three tiers correspond to effort values of $b=100$, $b=75$, and $b=50$, respectively, so that API users select an operating point on the learned cost--quality frontier without any change to the model weights or decoding configuration.

CAPTION:: Table 2: Mapping between the public API reasoning-effort tiers and the underlying scalar effort values $b$.

  API tier   Effort value $b$
  ---------- ------------------
  max        100
  high       75
  low        50

### 5.2 Asynchronous Post-training Infrastructure

The long-tail problem during the rollout phase of RL for LLMs has consistently been a major bottleneck for training efficiency. To address this, we extend our post-training infrastructure to allow asynchronous generation of samples ([Zeng et al., 2026](#bib.bib68); [Team et al., 2026b](#bib.bib55)), which significantly mitigates the long-tail issue in the rollout phase by maintaining a sufficiently high level of concurrency. Asynchronous training is now enabled for nearly all our RL and OPD tasks, and rollout efficiency has improved substantially.

#### 5.2.1 Overall Workflow

We colocate rollout and training on the same physical devices and time-share their execution, eliminating the need to manually tune resource allocation between the two phases. Each task specifies an upper bound on the number of in-flight samples, and the system maintains this bound throughout the rollout phase.

We evaluated three dispatch granularities for maintaining the target rollout concurrency. Our final approach is sample-level dispatch: once the number of newly completed samples reaches the GRPO group size assigned to the next prompt, we dispatch that prompt regardless of which groups produced those completions. This helps maintain a steady rollout concurrency throughout training. Before settling on this dispatch strategy, we experimented with two alternative dispatch granularities. In the first attempt, we dispatched several extra batches at the beginning and supplemented a full batch after each training iteration; however, this caused severe oscillations in training metrics, indicating that batch-level granularity was too coarse. In the second attempt, we switched to prompt-level dispatch, where a new prompt was dispatched after one GRPO group finishes, but found that it stalled easily on long-tail samples within a GRPO group, making it difficult to smoothly maintain the target rollout concurrency.

Once enough training samples have accumulated, training preempts ongoing rollouts. During training, we use concatenated routing-replay: for samples that span multiple checkpoints, we concatenate the expert routing produced at each rollout segment rather than discarding and recomputing the routing information with new checkpoints.

#### 5.2.2 Mitigating Length Bias and Off-Policy Effects

Asynchronous generation, while effectively improving rollout efficiency, introduces two side effects that can degrade training quality. First, it creates a length-distribution bias, especially in the early training stage, because shorter sequences tend to complete first and thus dominate the initial training batches. Second, it inevitably produces off-policy samples, i.e., samples whose tokens are partially or entirely generated by earlier checkpoints. These two issues require different handling strategies.

To address the length bias, we employ two mechanisms. First, the dispatcher can limit concurrency on a per-dataset basis, which helps regulate the proportion of each dataset in the steady-state training batch, indirectly mitigating the length skew by controlling the sources of incoming samples. Second, we support discarding early-returned short samples to smooth the transition into the steady-state length distribution, and prevent the model from overfitting to overly short sequences.

For the off-policy issue, we implement two additional mechanisms. First, by tuning the logic that controls sample dispatching and the waiting condition for training samples, we can bound the maximum off-policy ratio, ensuring that the training data does not deviate excessively from the current model. Second, during training, we add a loss masking scheme that eliminates the contribution of tokens with excessive staleness, thereby mitigating the adverse impact of stale samples on gradient updates.

#### 5.2.3 Performance Optimization

In our asynchronous RL framework, the rollout phase is periodically interrupted to switch to updated policy checkpoints. We aim to make this process seamless: interruptions should be near-instantaneous, and interrupted rollouts should resume as if never stopped.

To stop rollouts promptly, we support token-level interruption: generation can be halted at any token boundary. Once sufficient training data has been collected, all in-flight samples stop almost immediately, allowing the system to enter the training phase without delay.

To preserve rollout progress across checkpoint switches, rollout states such as KV cache and expert routing are persisted at token granularity during generation. Upon resumption with a new checkpoint, the persisted states are directly reused, eliminating the cost of re-prefilling and allowing interrupted samples to continue exactly where they left off. Since this requires retaining the states of all in-flight samples, we perform sample-grained garbage collection, releasing each sample's states as soon as it completes.

Beyond checkpoint switching, the same fast-interruption and seamless-resumption machinery allows the training jobs to respond promptly to cluster scheduling preemption signals without losing progress, thereby improving overall cluster utilization.

#### 5.2.4 Large-Scale On-Policy Distillation

As the last stage of post-training, the final full-vocabulary OPD task is trained on datasets from all domains using over 40 teacher models. It also adopts asynchronous generation to improve rollout efficiency. Due to differences in training procedures across domains, the best teacher for each domain may come from a different stage of model development. Moreover, the teacher models may differ architecturally from one another and from the student. Our post-training infrastructure readily accommodates this setting, supporting full-vocabulary OPD with an effectively unbounded number of architecturally heterogeneous teachers, and efficient switching among them at negligible cost ([DeepSeek-AI, 2026b](#bib.bib34)).

The OPD stage also requires dynamic reconfiguration during training. We continuously track model capabilities and may adjust the training recipe accordingly, including the dataset mixture, per-dataset concurrency limits, and active teachers. Such changes are straightforward in synchronous training where rollout batches provide explicit configuration boundaries. In the asynchronous setting, however, samples generated under different configurations may coexist in flight. Our infrastructure supports consistent transitions between configurations without disrupting rollout or training.

### 5.3 Evaluation

#### 5.3.1 Evaluation Setup

Our post-training evaluation focuses primarily on reasoning and agentic capabilities, while knowledge-intensive performance is largely determined by pretraining and reported in Table [1](#S4.T1). For reasoning, we evaluate on GPQA Diamond ([Rein et al., 2023](#bib.bib26)), Humanity's Last Exam ([Phan et al., 2025](#bib.bib25)), Codeforces (internal benchmark), and MathArena Apex ([Dekoninck et al., 2025](#bib.bib92)), using temperature and top-$p$ of 1.0. For agentic capabilities, we evaluate across four categories:

- •
  Code agent: Terminal-Bench 2.1 ([Merrill et al., 2026](#bib.bib21)), Terminal-Bench 3.0 ([Marten et al., 2026b](#bib.bib65)), Terminal-Bench 4.0 ([Marten et al., 2026a](#bib.bib75)), DeepSWE v1.1 ([DataCurve, 2026](#bib.bib15)), ProgramBench ([Yang et al., 2026](#bib.bib43)), NL2Repo-Bench ([Ding et al., 2025](#bib.bib74)).
- •
  Cyber security: SEC-Bench Pro version 260505 ([Lee et al., 2026](#bib.bib17)), CyberGym ([Wang et al., 2026c](#bib.bib20)), and ExploitGym ([Wang et al., 2026b](#bib.bib16)).
- •
  General agent: the public evaluation set of AutomationBench v1.0.6 ([Shepard and Salimans, 2026](#bib.bib19)), Agents' Last Exam ([Sun et al., 2026a](#bib.bib18)) (ALE-CLI).
- •
  Visual agent: Chartography ([Garre et al., 2026](#bib.bib78)), BabyVision ([Chen et al., 2026](#bib.bib76)), main set of ZeroBench ([Roberts et al., 2025](#bib.bib77)).

For code agents, we evaluate DeepSeek-V4.1-Flash using the Minimal mode of DeepSeek Harness with a 1M-token context window, temperature set to 1.0, and top-p set to 0.95. To align with official setup requirements, we employ the mini-SWE harness for DeepSWE v1.1. For SEC-Bench Pro, we utilize the Claude Code harness specifically for its session compact design. For visual agent tasks, we evaluate using the Claude Code harness with a 512k-token context window, temperature set to 1.0, and top-p set to 0.95. Agents' Last Exam and AutomationBench are evaluated with their official scaffolds. Model performance with other coding scaffolds is reported in Table [4](#S5.T4).

To mitigate reward hacking in coding agent evaluations, we restrict internet access and strip Git histories from the environment. Additionally, we automatically purge transient build and package caches across diverse environments, including Go module caches (go/mod), node modules dependency artifacts, compiled .jar files, and Python pycache directories. Despite these precautions, we still observe instances of exploit-seeking behavior during testing---such as decompiling core Ubuntu Linux packages to uncover vulnerabilities in CyberGym. As models grow increasingly capable, standard evaluation infrastructure (e.g., Docker containers and validation scripts) becomes more susceptible to model gaming. We urge the broader research community to prioritize detecting and mitigating these behaviors when designing next-generation benchmarks.

#### 5.3.2 Evaluation Results

CAPTION:: Table 3: Comparison between DeepSeek-V4.1-Flash with closed/open source models. ${\dagger}$ denotes text-only subset of HLE. The best results are highlighted in bold; the second-best results are underlined.

+--------------+----------------------------------+--------+-------------+------+---------+-----------+-------------+----------------+
|              | Benchmark (Metric)               | Opus-5 | GPT-5.6 Sol | K3   | GLM-5.3 | DS-V4-Pro | DS-V4-Flash | DS-V4.1-Flash  |
+--------------+                                  +--------+-------------+------+---------+-----------+-------------+----------------+
|              |                                  | Max    | Max         | Max  | Max     | Max       | Max         | Max            |
+--------------+----------------------------------+--------+-------------+------+---------+-----------+-------------+----------------+
| Reasoning    | GPQA Diamond (Pass@1)            | 93.4   | 94.1        | 92.9 | 88.1    | 92.4      | 89.9        | 90.9           |
|              +----------------------------------+--------+-------------+------+---------+-----------+-------------+----------------+
|              | HLE (Pass@1)                     | 56.3   | 44.5        | 43.5 | 42.0^†^ | 42.7^†^   | 37.8^†^     | 36.8 (39.1^†^) |
|              +----------------------------------+--------+-------------+------+---------+-----------+-------------+----------------+
|              | Codeforces (Rating)              | -      | -           | -    | -       | 3348      | 3289        | 3471           |
|              +----------------------------------+--------+-------------+------+---------+-----------+-------------+----------------+
|              | MathArena Apex (Pass@1)          | -      | -           | 65.6 | -       | 65.3      | 58.6        | 65.6           |
+--------------+----------------------------------+--------+-------------+------+---------+-----------+-------------+----------------+
| Agentic      | Terminal-Bench 2.1 (Pass@1)      | 89.1   | 88.8        | 88.3 | 88.2    | 87.9      | 82.7        | 90.6           |
|              +----------------------------------+--------+-------------+------+---------+-----------+-------------+----------------+
|              | Terminal-Bench 3.0 (Pass@1)      | 43.3   | 34.4        | 17.7 | 28.3    | 11.8      | 7.6         | 30.0           |
|              +----------------------------------+--------+-------------+------+---------+-----------+-------------+----------------+
|              | Terminal-Bench 4.0 (Pass@1)      | 51.8   | 39.9        | 12.6 | 37.9    | 12.4      | 7.0         | 31.2           |
|              +----------------------------------+--------+-------------+------+---------+-----------+-------------+----------------+
|              | DeepSWE v1.1 (Resolved)          | 74.0   | 73.0        | 67.5 | 66.9    | 62.7      | 54.4        | 74.2           |
|              +----------------------------------+--------+-------------+------+---------+-----------+-------------+----------------+
|              | ProgramBench (Almost@1)          | 37.0   | 23.0        | 17.5 | 19.0    | 15.5      | -           | 20.3           |
|              +----------------------------------+--------+-------------+------+---------+-----------+-------------+----------------+
|              | NL2Repo-Bench (Score)            | 75.3   | 56.8        | 58.0 | 58.0    | 61.5      | 54.2        | 65.4           |
|              +----------------------------------+--------+-------------+------+---------+-----------+-------------+----------------+
|              | CyberGym (Pass@1)                | -      | 84.5        | 80.0 | 84.5    | 83.3      | 76.7        | 88.1           |
|              +----------------------------------+--------+-------------+------+---------+-----------+-------------+----------------+
|              | SEC-Bench Pro (Pass@1)           | -      | 74.3        | -    | -       | 56.4      | 30.9        | 62.8           |
|              +----------------------------------+--------+-------------+------+---------+-----------+-------------+----------------+
|              | ExploitGym (Pass@1)              | 22.1   | 33.7        | -    | 15.0    | 5.4       | 1.8         | 15.3           |
|              +----------------------------------+--------+-------------+------+---------+-----------+-------------+----------------+
|              | HLE w/ tools (Pass@1)            | 63.6   | -           | 59.8 | 62.5    | 60.0      | 51.5        | 63.9           |
|              +----------------------------------+--------+-------------+------+---------+-----------+-------------+----------------+
|              | Automation-Bench (Pass@1)        | 50.3   | 45.8        | 46.7 | 48.8    | 43.2      | 37.7        | 54.8           |
|              +----------------------------------+--------+-------------+------+---------+-----------+-------------+----------------+
|              | Agents' Last Exam(Pass@1)        | 28.6   | 26.7        | 27.6 | 28.5    | 25.7      | 25.2        | 31.8           |
|              +----------------------------------+--------+-------------+------+---------+-----------+-------------+----------------+
|              | Chartography w/ tools (Pass@1)   | 84.0   | 79.9        | 68.1 | -       | -         | -           | 78.9           |
|              +----------------------------------+--------+-------------+------+---------+-----------+-------------+----------------+
|              | BabyVision w/ tools (Pass@1)     | 94.1   | 88.9        | 85.7 | -       | -         | -           | 89.6           |
|              +----------------------------------+--------+-------------+------+---------+-----------+-------------+----------------+
|              | ZeroBench-main w/ tools (Pass@5) | 52.0   | 53.0        | 41.0 | -       | -         | -           | 49.0           |
+--------------+----------------------------------+--------+-------------+------+---------+-----------+-------------+----------------+

As detailed in Table [3](#S5.T3), DeepSeek-V4.1-Flash exhibits significant performance upgrades across both reasoning and agentic benchmarks over its predecessor, DeepSeek-V4-Flash, while matching or outperforming top-tier open-source and proprietary models.

In core reasoning tasks, DeepSeek-V4.1-Flash achieves a Codeforces rating of 3471, surpassing both DeepSeek-V4-Flash (3289) and DeepSeek-V4-Pro (3348). On MathArena Apex, it obtains a 65.6% Pass@1 accuracy, fully matching the top-performing open-source baseline Kimi-K3 (65.6%) and DeepSeek-V4-Pro (65.3%). Furthermore, its GPQA Diamond score reaches 90.9%, showing steady improvements over DeepSeek-V4-Flash (89.9%).

The performance gains are even more pronounced across agentic tasks. Notably, on DeepSWE v1.1, DeepSeek-V4.1-Flash reaches 74.2% pass rate, marking a substantial jump from DeepSeek-V4-Flash (54.4%) and surpassing leading proprietary models including Opus-5 (74.0%) and GPT-5.6 Sol (73.0%). On Terminal-Bench 2.1, it achieves 90.6%, outperforming Opus-5 (89.1%) and GLM-5.3 (88.2%). Similarly, on Automation-Bench (54.8%) and Agents' Last Exam (31.8%), DeepSeek-V4.1-Flash establishes leading scores over both open-source counterparts and top closed-source systems. Despite its compact nature, DeepSeek-V4.1-Flash demonstrates state-of-the-art agentic capabilities, substantially closing the gap with frontier closed-source models while establishing clear advantages among open-source alternatives.

On cyber-security tasks, DeepSeek-V4.1-Flash establishes a new state of the art among open-source models. Given the dual-use nature of these capabilities, we encourage the community to apply them responsibly, such as for defensive security research and vulnerability remediation. In the domain of visual agent tasks, DeepSeek-V4.1-Flash demonstrates robust capabilities, particularly in scenarios requiring visual reasoning and the analysis of complex professional charts. While it outperforms the leading open-source model Kimi-K3, we acknowledge that a measurable gap still remains when benchmarked against the leading closed-source alternatives.

Beyond peak performance, DeepSeek-V4.1-Flash exposes a reasoning-effort setting that allows users to trade inference cost for accuracy in a controllable manner. As illustrated in Figure [9](#S5.F9), both accuracy and output length increase steadily with the effort level across reasoning and agentic benchmarks alike. Raising the effort from 25 to 100 improves the average Pass@1 on eight reasoning-intensive benchmarks from 67.1% to 76.3%, on DeepSWE v1.1 from 66.0% to 74.2%, and on Terminal-Bench 2.1 from 82.4% to 90.6%, at the cost of roughly $2.5\times$ more output tokens. Notably, the effort control learned on single-response reasoning transfers faithfully to long-horizon agentic trajectories, where it governs the total amount of exploration and verification across turns. The gains are front-loaded: the 60--80 range already recovers most of the accuracy of the maximum setting at less than half of its token budget, whereas the final step to effort 100 lengthens agent trajectories by $1.6$--$1.8\times$ for only marginal improvements. The maximum tier is thus best reserved for the most challenging tasks, while moderate effort levels offer a favorable cost--performance balance for everyday agentic use.

#### 5.3.3 Performance across reasoning efforts

Figure [9](#S5.F9) reports performance and average response length across a range of budget values. As the reasoning effort increases, the model produces progressively longer reasoning traces and accuracy improves monotonically on reasoning-intensive benchmarks and software engineering tasks, with the largest gains concentrated in the low-to-mid budget range and diminishing returns beyond. Although our RL involves only a limited number of effort levels, the use of scalar efforts achieves flexible, interpolated control of response length within a specific range. This allows practitioners to trade off quality against latency and token cost along a smooth continuum: latency-sensitive applications can operate at low effort with modest accuracy degradation, while difficult tasks can invoke high effort to recover the model's full reasoning capability. In our public API service, we expose three preset effort levels that map onto this scale: low, high, and max correspond to effort values of 50, 75, and 100, respectively.

CAPTION:: Figure 9: Performance and output length as a function of reasoning effort. Each panel plots Pass@1 (solid, left axis) and mean output tokens per response (dashed, right axis) as the reasoning-effort value is varied from 25 to 100; The results of reasoning-intensive benchmarks are averaged over eight benchmarks (AIME 2026, Apex 2025 Shortlist, GPQA Diamond, HLE, IMO-AnswerBench, LiveCodeBench, MathArena-Apex, SimpleQA-Verified). DeepSWE v1.1 is evaluated based on mini-SWE and Terminal-Bench v2.1 is evaluated based on DeepSeek Harness (Minimal).

#### 5.3.4 Performance across agent scaffolds

In practice, a model is rarely deployed within a single fixed agent framework; different scaffolds vary in their system prompts, tool definitions, context management strategies, and interaction protocols, and a model that overfits to one particular harness may degrade substantially when placed in another. To assess the robustness of our model to such variation, our comparison covers eight configurations from six scaffold families: Claude Code ([Anthropic, 2026](#bib.bib96)), Codex ([OpenAI, 2026](#bib.bib97)), OpenCode ([Anomaly, 2026](#bib.bib98)), Pi ([Zechner, 2026](#bib.bib99)), mini-SWE ([Yang et al., 2024](#bib.bib94)), and DeepSeek Harness (DSH) ([DeepSeek-AI, 2026a](#bib.bib95)) in Minimal, Standard, and PTC modes. For each scaffold, we keep the model checkpoint, decoding configuration, and task set identical, and only the surrounding harness, including its native system prompt, tool schema, and turn-taking logic, is changed. Table [4](#S5.T4) reports performance at Max reasoning effort (100) on DeepSWE v1.1 and Terminal-Bench v2.1.

CAPTION:: Table 4: Performance across agent scaffolds at Max reasoning effort.

+------------------------------+-------------+-------+----------+------+----------+--------+--------+--------+
| Benchmark (Metric)           | Claude Code | Codex | OpenCode | Pi   | mini-SWE | DeepSeek Harness         |
+------------------------------+-------------+-------+----------+------+----------+--------+--------+--------+
| Minimal                      | Standard    | PTC   |          |      |          |        |        |        |
+------------------------------+-------------+-------+----------+------+----------+--------+--------+--------+
| DeepSWE v1.1 (Resolved)      | 69.8        | 65.6  | 65.5     | 66.2 | 74.2     | 72.6   | 70.5   | 67.6   |
+------------------------------+-------------+-------+----------+------+----------+--------+--------+--------+
| Terminal-Bench v2.1 (Pass@1) | 88.0        | 84.1  | 85.0     | 86.1 | 90.3     | 90.6   | 85.8   | 85.8   |
+------------------------------+-------------+-------+----------+------+----------+--------+--------+--------+

Note. All scaffolds use $N=8$ samples per task on DeepSWE v1.1 and $N=3$ on Terminal-Bench v2.1. Runs use Linux containers, temperature 1.0, top-p 0.95, a 1M-token context window, and max_steps=500 model-generation rounds per agent on both benchmarks. Terminal-Bench v2.1 is evaluated without network access. We evaluate four Claude Code versions, with per-version results and their average reported in Appendix Table [5](#A2.T5); this table reports v2.1.251. More scaffold-specific configurations are detailed in Appendix [B.1](#A2.SS1).

The model's agentic capabilities transfer well across scaffold families with different prompts and tool interfaces, rather than depending on conventions specific to a particular harness. Its performance remains robust as the surrounding interaction protocol and tool abstractions change, indicating that its agentic behavior is not tightly coupled to a single scaffold design. This robustness is consistent with the diversity of environments, tool schemas, and interaction formats in our synthesized training data (Section [5](#S5)), which is designed to encourage generalization across agent scaffolds.

#### 5.3.5 Multi-Agent

CAPTION:: Figure 10: Test-time compute scaling for single-agent and multi-agent configurations on ProgramBench (Almost@1) and FrontierSWE v2 (Mean@5) as functions of the per-rollout wall-clock deadline.

To explore multi-agent collaboration on complex tasks, we conduct preliminary experiments with DeepSeek-V4.1-Flash using DeepSeek Harness's Agent Team mode.

Multi-Agent Harnesses. We use DeepSeek Harness in Agent Team mode, where a lead agent can asynchronously create named, persistent teammates by default through spawn_teammate. Each receives a delegated task and starts in either fresh mode without lead history or fork mode with a one-time snapshot of the lead's completed turns. All agents share one repository checkout, making edits immediately visible to one another.

Agents communicate through a durable peer mailbox. A message sent through send_message reaches a running teammate at its next step boundary, starts a new turn for an idle teammate, or resumes an inactive teammate. Across each teammate's task lifecycle, the lead monitors runtime status with list_agents and waits for status, mailbox, or shared-task changes with wait_agent. Task ownership, dependencies, and advisory write scopes are maintained on a shared task board (using the four team_task\_\* tools with revision checks on updates). When intervention is needed, only the lead can interrupt a teammate's current turn through interrupt_agent. Once the required work is complete, the lead reviews and tests the combined changes and produces the final response.

Training. We train Agent Team mode with an RL reward combining task performance, a collaboration bonus that encourages delegation and inter-agent communication, and a derived-latency penalty that promotes efficient coordination. Derived latency is computed by representing execution events and their collaboration dependencies as a directed acyclic graph (DAG), assigning costs from token counts at fixed prefill/decode rates plus measured tool-execution time, and taking the length of the critical path. This encourages useful parallelism while penalizing unnecessary sequential work and synchronization, with reduced sensitivity to serving-side batching and queuing delays.

Performance. We construct a high-confidence subset of ProgramBench ([Yang et al., 2026](#bib.bib43)) by retaining only tasks for which the reference solution achieves a pass rate of at least 95% on the hidden test suite. This filtering procedure leaves 172 \"golden\" tasks. We also evaluate on FrontierSWE v2 ([Kondra et al., 2026](#bib.bib44)), a larger and more challenging successor to FrontierSWE that uses a substantially improved methodology. We construct a no-GPU subset from the currently public tasks by excluding tasks that require GPU access. We evaluate single-agent and multi-agent configurations on both benchmarks under explicit per-rollout wall-clock deadlines. The results reported here are preliminary: we compare the strongest observed multi-agent configurations with the strongest available single-agent baselines. On ProgramBench, we run up to three rollouts per task, corresponding to 516 planned rollouts for each configuration. We report Almost@1, which measures the fraction of individual rollouts achieving a score of at least 0.95. On FrontierSWE v2, we report Mean@5. ProgramBench deadlines range from 1 to 12 hours, while FrontierSWE v2 is evaluated at deadlines ranging from 1 to 20 hours. At each deadline, metrics are computed from the outputs available when the deadline is reached.

As shown in Figure [10](#S5.F10), multi-agent configurations outperform their single-agent counterparts at every deadline on both benchmarks. On ProgramBench, Almost@1 increases from 13.59% at 1 hour to a peak of 30.04% at 8 hours for the multi-agent configuration, compared with 12.79% and 20.39% for the single-agent configuration. On FrontierSWE v2, Mean@5 increases from 13.50% at 1 hour to 32.90% at 20 hours for the multi-agent configuration, compared with an increase from 10.50% to 28.20% for the single-agent configuration.

## 6 Conclusion, Limitations, and Future Directions

In this work, we introduce DeepSeek-V4.1-Flash, a multimodal Mixture-of-Experts (MoE) model with support for contexts of up to one million tokens. Through joint optimization of model architecture, cache precision, and deployment strategy, DeepSeek-V4.1-Flash pushes the limits of KV cache compression. Its Causal Encoder-Decoder (CED) architecture enables the model to activate only 8B parameters per token during prefill, compared with 16B during decode, improving cost efficiency for input-heavy agentic workloads. At equal sequence lengths, cross-layer KV cache reuse in Compressed Sparse Attention 2 (CSA2) and FP4 KV caching reduce its global KV cache footprint (always in HBM) to 890 bytes per token, roughly 1/4 of the corresponding footprint of DeepSeek-V4-Flash. SWA Bounded Replay further reduces its persistent KV cache footprint (always on SSD or in host memory) to roughly 1/8 of that of DeepSeek-V4-Flash. These reductions alleviate HBM and SSD capacity pressure while the model delivers substantially better overall performance than DeepSeek-V4-Flash. Despite possessing a significantly smaller parameter footprint than contemporary open-source models such as GLM-5.3 and Kimi-K3, DeepSeek-V4.1 achieves comparable---and in several tasks, superior---performance across key benchmarks.

Although DeepSeek-V4.1-Flash substantially simplifies several architectural components relative to DeepSeek-V4-Flash, the newly introduced architectural changes also create robustness boundaries that have yet to be fully characterized. Our internal evaluations cover a diverse range of test cases and boundary conditions, and we have not observed any systematic degradation in model capabilities in the evaluated settings. Nevertheless, no finite test suite can cover every extreme input and deployment condition. Potential selection errors in CSA2 and approximate state reconstruction in SWA Bounded Replay may still cause capability degradation in untested boundary cases. Going forward, we will continue to expand our stress-testing and evaluation stack, with particular attention to sparse retrieval over long contexts and SWA state reconstruction at cache-resumption boundaries. We will also monitor real-world workloads, systematically characterize potential failure modes and robustness boundaries, and further improve model robustness under extreme conditions.

As AI models achieve remarkable performance capabilities, standard evaluation benchmarks have increasingly reached saturation. While DeepSeek-V4.1-Flash demonstrates performance that closely approaches top-tier models like Fable-5 and GPT-6 Astra---offering a highly comparable user experience in daily applications---a performance gap remains on the most challenging tasks. Although benchmark scores show a narrow margin, this parity does not imply that the model matches the frontier capabilities of leading closed-source systems on complex, high-difficulty reasoning and edge cases.

Consequently, we will continuously update our evaluation protocols to ensure rigorous assessment of state-of-the-art reasoning boundaries. Alongside continued efforts to reduce model costs, we believe that further advances in model intelligence will depend on the coordinated scaling of data, model capacity, and RL. With DeepSeek-V4.1-Flash as a new starting point, we will continue to explore the limits of model capabilities and systematically address key challenges in large-scale data synthesis and RL scaling. We will also actively integrate model--harness co-design, enabling the joint system to evolve and be optimized together. By advancing cost reduction and capability scaling in tandem, we hope to make highly capable agents more accessible and easier to deploy, further lowering the barriers to adopting AI technologies across a broader range of industries and scenarios.

