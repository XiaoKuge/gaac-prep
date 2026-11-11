## 3 General Infrastructures

### 3.1 Training Infrastructure

#### 3.1.1 Multimodal Training Infrastructure

##### Communication-Computation Overlap in Contrastive Learning.

The vision encoder is first optimized with a contrastive objective before being fine-tuned with a generative next-token prediction loss. In the contrastive phase, the loss is computed over a full batch of text and vision pairs, so the features of both modalities must be all-gathered across data-parallel ranks, incurring substantial communication. Because the gradient of the text features depends only on the gathered visual features---and, symmetrically, the gradient of the visual features depends only on the gathered text features---each all-gather can be overlapped with the forward or backward pass instead of stalling the pipeline:

  -- ------------------------------------------------------------------------------------------------------------------------------------------------- --
     $$\text{Forward}(V)\;\rightarrow\;\bigl(\text{Forward}(T)\;\|\;\text{AllGather}(V)\bigr)\;\rightarrow\;\nabla_{\text{Text}}\\                     
     \;\rightarrow\;\bigl(\text{Backward}(T)\;\|\;\text{AllGather}(T)\bigr)\;\rightarrow\;\nabla_{\text{Vision}}\;\rightarrow\;\text{Backward}(V),$$   
  -- ------------------------------------------------------------------------------------------------------------------------------------------------- --

where $V$ and $T$ denote the visual and text features, $(A\,\|\,C)$ denotes the overlap of computation $A$ with communication $C$, and $\nabla$ denotes the gradient computation. In this schedule, the visual features are gathered during the text forward pass and the text features during the text backward pass, so that both all-gathers are hidden entirely behind useful computation.

##### End-to-End Parallelism.

To handle the model and data heterogeneity between the vision encoder and the LLM ([Zhang et al., 2025b](#bib.bib56)), we adopt the disaggregated encoder design used in recent training systems ([Team et al., 2025](#bib.bib54); [Team et al., 2026b](#bib.bib55)). The vision encoder is replicated outside the LLM parameter tree, and each training step is divided into three phases: vision encoder forward, LLM forward/backward, and vision encoder backward. This separation prevents interference between vision encoder and LLM computation. Load-balanced vision processing is confined to the first and last phases, while the LLM phase remains free of vision computation and preserves the parallel strategy of text-only training.

##### Long-Sequence Multimodal Training Optimization.

DeepSeek-V4.1-Flash is trained on sequences of up to one million tokens, with a substantial share of training occurring at ultra-long sequence lengths in both the pre-training and post-training stages. At these sequence lengths, multimodal samples create heavy I/O, CPU, and memory bottlenecks.

- •
  Balanced image sharding. During pre-training, a single ultra-long, image-dense sequence can exhaust one host's I/O, CPU, and memory during loading, so the images of each sequence are sharded across the CP ranks with load balancing, and each image is loaded exactly once. With images read once, loading stays hidden behind compute whenever

    -- ---------------------------------------------------------------------------------------------------------------------------------------------------------- --
       $$\frac{N\times\rho}{B_{\mathrm{IO}}}<\frac{N\times C}{B_{\mathrm{GPU}}}\quad\Longleftrightarrow\quad\rho<\frac{B_{\mathrm{IO}}}{B_{\mathrm{GPU}}}\,C,$$   
    -- ---------------------------------------------------------------------------------------------------------------------------------------------------------- --

  where $N$ is the token count, $\rho$ the raw bytes per token, $C$ the per-token compute, and $B_{\mathrm{IO}}$, $B_{\mathrm{GPU}}$ the file-system and GPU bandwidths. Since $N$ cancels, the criterion involves only per-token quantities ($\rho$ and $C$), independent of sequence length and cluster size; $\rho$ is set by the vision-module configuration (e.g., the resolution cap or the spatial downsample). Storage throughput therefore becomes a bottleneck only for small models with low per-token compute, as in ablations, while production-scale models remain compute-bound.
- •
  Incremental image transfer. Besides the balanced sharding above, the reinforcement-learning rollout transfers images to the inference engine only incrementally, and caches the engine's CPU-side decoding and preprocessing outputs on a distributed file system for reuse across rollouts and subsequent training.

#### 3.1.2 Attention Sharing Training for CSA2

In Section [2.3](#S2.SS3), we introduce CSA2, an attention sharing method with three modes, some of which involve sharing one or more of the main KV, the indexer K, and the Top-K indices across multiple layers. Supporting CSA2 in large-scale distributed training requires additional coordination beyond the attention computation itself. In particular, layers that share attention components may be placed on different pipeline stages, making direct module reuse incompatible with conventional stage-local execution. Therefore, we adopt several designs to support CSA2 training.

Shadow indexers address this issue by placing a lightweight executable replica on each participating stage while retaining a single logical owner for the shared parameters. The owner remains responsible for optimization and checkpointing, whereas parameter synchronization and gradient aggregation keep the shadow replicas consistent throughout training. This design preserves the original model semantics without requiring the pipeline scheduler to treat shared layers as a special execution unit.

Pipeline payload extensions provide the intermediate representations and sparse routing information required by downstream consumers when the source and consumer layers cross a pipeline boundary. These states are incorporated into the existing point-to-point communication path and partitioned consistently with context parallelism, avoiding unnecessary replication while maintaining the corresponding gradient flow.

Micro-batch-level shared-state management tracks the states associated with concurrently active pipeline micro-batches and coordinates their lifetimes across forward execution, activation recomputation, and backward propagation. Shared states are retained until their final consumer has completed and are then released promptly to limit additional memory overhead. The same runtime abstraction also handles stage placement and source--consumer relationships, allowing the attention implementation to access shared states without depending on the physical pipeline layout.

Together with lightweight adaptations to the optimizer, checkpointing, warm-up, and computation-graph tracing workflows, these mechanisms enable CSA2 to operate transparently under the existing distributed training interface and pipeline schedules.

#### 3.1.3 Engram

Engram embedding tables are partitioned by row across dedicated process groups of engram parallel size. The group size controls the trade-off between per-device memory usage and the communication scope of embedding lookups. Optimizer states are further sharded across replicas of each table partition. Engram lookup indices depend solely on the input token sequence. Embedding prefetch is therefore initiated for the entire local batch before each pipeline stage begins processing microbatches for the current training step, minimizing interference with pipeline execution. Embedding gradients are buffered during backward and returned to their owning ranks after the backbone backward pass. For efficient integration with multimodal training, embedding prefetch and gradient transfers are scheduled to overlap with the vision encoder's forward and backward computation. Embeddings are stored and fetched in FP8, with the retrieved values and scaling factors passed directly to the following GEMM. For Engram table updates, Sinkhorn normalization maintains row and column scaling vectors across iterations to avoid repeated writes of the full normalized matrix. Row normalization and the accumulation of partial column statistics are fused into a single kernel to further reduce memory traffic. During RL rollouts, Engram embedding tables remain resident in GPU memory. This placement reduces host memory pressure and helps avoid out-of-memory failures caused by host memory fragmentation.

### 3.2 Inference System

DeepSeek-V4.1-Flash is designed with inference efficiency as a first-class concern. Although its architecture is conceptually complex, the resulting inference kernel flow is remarkably concise. Through reasonable kernel fusion, we encapsulate the intricate operations and keep hardware resources fully pipelined inside a small number of fused kernels---including the fused-RoPE-attention-RoPE-cast kernel in FlashMLA ([Li and Liu, 2025](#bib.bib38)), the Mega-Gate, Mega-mHC, and Mega-MoE kernels in DeepGEMM ([Zhao et al., 2025](#bib.bib37)), the kernels in TileKernels ([Wang et al., 2026a](#bib.bib39)), and the TopK kernel in DeepSelect ([Qian et al., 2026](#bib.bib40)). As a result, the vast majority of Transformer layers---those whose CSA2 operates in Reuse Mode---execute with only 15 kernels during prefill and 11 during decode, thereby achieving both high-throughput and low-latency inference.

At the deployment level, we adopt Encoder--Prefill--Decode (EPD) disaggregation, enabling vision encoding, prefill, and decoding to scale independently and overlap in execution.

#### 3.2.1 Persistent KV Cache Management

Under identical workloads, V4.1 reduces the persistent KV cache footprint to 1/8 of that of V4. Two multiplicative factors account for this reduction: the persistent KV cache no longer stores SWA KV, which almost halves its size, and the global KV retained in it is further compressed to 1/4 of V4's footprint through architectural and precision optimizations.

In the V4 deployment, SWA KV accounts for nearly half of the persistent KV cache capacity. Within this cache, global KV and SWA KV are managed independently, governed by an LRU eviction policy. Global KV is stored in its entirety, and upon a hit, the complete prefix is reused. In contrast, SWA KV is cached at two specific points---the end of the prompt and the end of the output---to facilitate regeneration and multi-turn sessions; a hit allows computation to resume from that cached position. To maintain a high hit rate, we configured a sufficiently large persistent KV cache on SSD such that, under typical workloads, both types of KV remain resident for over 72 hours. Despite retaining only $n_{\mathrm{win}}$ KV entries at designated positions, the uncompressed SWA KV cache still incurs a substantial storage overhead, especially in multi-turn conversations with short turns.

Persistently storing SWA KV is both costly and ineffective, because its access pattern does not match the persistent KV cache's long retention policy. Unlike global KV, which exhibits long-tail reuse, SWA KV is reused only within a narrow, minute-scale window inside an active session and becomes dead once the session ends or the next turn begins. The V4 technical report proposed Zero SWA Caching, which avoids the storage overhead by recomputing missing SWA KV. Exact recovery, however, requires a full forward pass over $L\times n_{\mathrm{win}}$ tokens, whose cost proved prohibitive in production deployments.

V4.1 therefore revises persistent KV cache management as follows:

1.  1.
    SWA KV is no longer cached in the persistent KV cache and instead stored in a distributed memory pool provisioned from 10% of the host DRAM on each machine. Although this pool is far smaller in aggregate capacity, its short TTL (only minutes) allows expired entries to be recycled immediately for new sessions; under real-world workloads, this high turnover suffices to serve the vast majority of concurrent active sessions. Global KV remains in the persistent KV cache with a guaranteed lifetime of at least 72 hours.
2.  2.
    Evicting SWA KV inevitably causes misses, which stay affordable thanks to a lightweight fallback, Encoder SWA Bounded Replay (detailed in Section [3.2.2](#S3.SS2.SSS2)). For the inevitable but infrequent requests that hit global KV but miss SWA KV, it recovers the missing state by recomputing only $n_{\mathrm{win}}$ tokens instead of a full $L\times n_{\mathrm{win}}$-token forward pass. This bounded replay is the cornerstone of the design: it turns a catastrophic miss into a graceful, inexpensive degradation, thereby justifying the removal of SWA KV from the persistent KV cache.

#### 3.2.2 SWA Bounded Replay

Since SWA dependencies accumulate across layers, exactly reconstructing the SWA KV of $L$ layers would require replaying $L\times n_{\mathrm{win}}$ tokens. SWA Bounded Replay instead replays only the most recent $n_{\mathrm{win}}$ tokens and truncates SWA to the replay segment, accepting approximate states: for a replay starting at position $s$, a query at position $i$ attends to SWA keys in $\bigl[\max(s,\,i-W+1),\,i\bigr]$.

##### Encoder SWA Bounded Replay.

Encoder SWA Bounded Replay makes prefix caching depend only on global KV, allowing SWA KV to be removed from the persistent KV cache.

When the encoder SWA KV is missing, we replay the last $n_{\mathrm{win}}$ tokens of the cached prefix and process them together with the uncached suffix. The replayed tokens regenerate only SWA KV, reusing the cached global KV without recomputation or overwriting, while the uncached suffix generates both global KV and SWA KV.

By design, the replayed prefix state is approximate, so the global KV and SWA KV computed for the uncached suffix depend on the cache-hit position and are not mathematically identical across positions. Encouragingly, our experimental evidence confirms that this bounded replay strategy barely compromises response quality.

##### Decoder SWA Bounded Replay.

Decoder SWA Bounded Replay bounds the decoder forward pass to $n_{\mathrm{win}}$ tokens, nearly halving total prefill computation.

Under CED, decoder global KV is projected from the final encoder hidden states. The only obstacle to ending prefill at the encoder is decoder SWA KV, which is generated from each decoder layer's own hidden states and is needed by the first decode steps. Since we never cache decoder SWA KV, exactly reconstructing it requires running the $\frac{L}{2}$ decoder layers over the last $\frac{L}{2}\times n_{\mathrm{win}}$ prompt tokens, which is expensive when a short uncached suffix follows a long cached prefix. Therefore, we also apply the bounded replay strategy to this scenario: at every prefill, we replay the last $n_{\mathrm{win}}$ tokens of the prompt, feed their encoder outputs through the decoder layers under the same SWA truncation, and use the resulting decoder SWA KV only for decoding, not for prefix caching.

By design, the reconstructed decoder SWA KV is not mathematically equivalent to that from a full decoder forward pass. Also, we find that this strategy has only a negligible impact on response quality. For added safety, we additionally simulate the same replay during post-training for train-aware adaptation.

## 4 Pre-Training

### 4.1 Data Construction

##### Text Data Curation

In pursuit of higher intelligence, we go beyond the general, sample-level quality reflected by small-scale data experiments and focus more on the holistic interactions among diverse corpora that offer unique information gains. We adopt a more systematic and standardized data construction pipeline to improve data quality and optimize the data mixture. Specifically, based on more comprehensive evaluations, a scaling ladder over model parameters and training data is carefully designed to guide large-scale training runs. We filter out model-generated content with limited information gain, including outputs from less capable models and low-quality machine-translated text. We regard such content as implicit duplication, as it largely reformulates existing information and may become detrimental over long training horizons. We also explore model-in-the-loop data iteration approaches as a foundation for future large-scale synthetic data. In addition, we involve more domain experts to construct fine-grained data quality evaluation dimensions. Compared with the previous version, the new corpus incorporates more recent code from newly released open-source repositories, commits, libraries, and emerging frameworks to cover a broader range of programming languages and better reflects contemporary real-world software engineering scenarios.

##### Multimodal Data Curation

Our multimodal pre-training dataset primarily comprises three types of data: image-text pairs, interleaved image-text data, and domain-specific data. Operating on the premise that raw web data naturally provides rich multimodal knowledge, we refrained from large-scale data synthesis; instead, we prioritized cleaning and utilizing the data in its native form to achieve the most direct and scalable visual knowledge compression during pre-training. During the initial data collection, we found that our crawling system was overly biased toward text-centric web content; we therefore re-bootstrapped it from Common Crawl to improve its coverage of multimodal sources. For image-text data, we extract images together with their associated alt text from webpages, filter them by applying an image-text relevance threshold, and deduplicate them based on image semantics. For interleaved data, we build this subset predominantly from webpages and PDFs. Processing large-scale multimodal corpora usually incurs higher CPU and disk storage costs than processing text-only corpora, which motivated us to organize the interleaved-data construction into progressively more expensive stages. Before image retrieval, we apply heuristic and statistical filtering, deduplication, and quality models to select high-value documents. The surviving documents are then assembled into interleaved image--text sequences, where filtering and deduplication are applied again in an image-aware manner. Finally, we employ SmolVLM ([Marafioti et al., 2025](#bib.bib52)) to conduct strict quality scoring on the image-text content, thereby extracting high-quality interleaved data. Documents filtered out during this process are partly recycled into additional image-text pairs via screening and recombination. To compensate for the inherent limitations of web-gathered data, we also incorporate domain-specific datasets to boost the model's capabilities in fine-grained visual perception (e.g., visual grounding and pointing), optical character recognition (OCR), and the acquisition of long-tail knowledge. We also collect extensive image-code pairs and computer-use trajectories to improve multimodal agentic understanding.

##### Data Integration and Deduplication

As our text-only and multimodal data were processed through distinct pipelines, we constructed the final training corpus as the union of both data sources. For overlapping samples, we replace the text-only versions with their multimodal counterparts and use the larger epoch count of the two configurations. After this substitution, the resulting corpus uses a $7{:}1$ token ratio of text-only to multimodal data. We minimize sample overlap during pre-training and context extension by jointly prefetching and assigning training samples. Ultra-long documents are deterministically pre-split before mixing to ensure a uniform distribution of training tokens across data shards and training steps. We further enhance our best-fit packing algorithm, achieving a padding rate of at most $10^{-4}$.

### 4.2 Pre-Training Setups

#### 4.2.1 Model Setups

We set the number of Transformer layers to 40 and the hidden dimension $d$ to 5120. We adopt a Causal Encoder-Decoder architecture, with 20 layers in the encoder and 20 layers in the decoder. For the first two layers, we use pure sliding window attention. The remaining 18 encoder layers use CSA2 with a compression rate of $m=2$. These layers are divided into three identically configured groups of six layers. In each group, the first layer operates in Full Mode, and the remaining five layers operate in Reuse Mode. The 20 decoder layers use CSA2 with a compression rate of $m=1$. These layers are divided into five groups of four layers. In the first group, the first layer operates in Full Mode, and the remaining three layers operate in Reuse Mode. The remaining four groups share the same configuration: the first layer operates in Reindex Mode, and the remaining three layers operate in Reuse Mode. For all CSA2 layers, we set the number of indexer query heads to 32, the indexer head dimension to 128, and the number of KV entries selected for sparse attention (i.e., attention top-k) to 512. We set the number of query heads to 64, the head dimension to 512, and the query compression dimension to 1280. For the Hierarchical Sparse Indexer, we select a maximum of 2,048 blocks with 8 positions, yielding up to 16,384 candidate positions in total. The number of output projection groups is set to 8, and the dimension of each intermediate attention output is set to 1024. For the additional branch of sliding window attention, the window size $n_{\mathrm{win}}$ is set to 128. We employ MoE layers in all Transformer blocks, using SwiGLU activation function with clamping ([OpenAI, 2025](#bib.bib41)) at a threshold of 10. Each MoE layer consists of 1 shared expert and 384 routed experts, where the intermediate hidden dimension of each expert is 2304. Among the routed experts, 6 experts will be activated for each token. As for mHC, the expansion factor is set to 4, and the number of Sinkhorn-Knopp iterations is set to 20. For the vision encoder, we set its number of layers to 32, the hidden dimension to 1024, the number of attention heads to 16, and the image patch size to 14. The vision MLP projector has 2 layers with a hidden dimension of 5120. Under this configuration, DeepSeek-V4.1-Flash comprises 552B backbone parameters, with 8B activated per token during prefill and 16B during decode.

#### 4.2.2 Training Setups

We employ the Muon optimizer ([Jordan et al., 2024](#bib.bib31); [Liu et al., 2025](#bib.bib32)) for the parameters of linear transformations, use AdamW optimizer ([Loshchilov and Hutter, 2019](#bib.bib60)) for the weights of all RMSNorm modules and other non-matrix parameters, and use Sinkhorn-balanced update for all embeddings and prediction head. For AdamW, we set its hyper-parameters to $\beta_{1}=0.9$, $\beta_{2}=0.95$, $\varepsilon=10^{-20}$, and $\mathrm{weight\_decay}=0.1$. For Muon, we set the momentum to $0.95$ and the weight decay to $0.1$, and rescale the RMS of each update matrix to $0.18$ for reutilization of the AdamW learning rate. For Sinkhorn-balanced updating, we use the same momentum coefficient and learning-rate correction factor as for Muon and set $K=11$, $\tau=10^{-3}$, $\varepsilon=10^{-20}$. Following ([Cheng et al., 2026b](#bib.bib50)), the learning rate of Engram is scaled by $5\times$. We train DeepSeek-V4.1-Flash on 45T tokens of multimodal data with no instability. We keep the batch size fixed at 100.6 million tokens throughout training. The learning rate is linearly warmed up over the first 2000 steps and then maintained at $2.6\times 10^{-4}$ until 28T tokens. Between 28T and 40T tokens, we decay the learning rate to $2.6\times 10^{-5}$ following a cosine schedule. We keep the learning rate at this value from 40T to 45T tokens. We train the model from scratch with sparse attention at a sequence length of 64K and extend the sequence length to 1M at 34T tokens. For auxiliary-loss-free load balancing, we set the bias update speed to 0.001 for both image and text tokens, while retaining a small sequence-level balance loss with a loss weight of 0.0001 to avoid extreme imbalance within single sequences. Similar to DeepSeek-V4, we employ sample-level attention masking during pre-training.

##### Vision Encoder Training.

Our DeepSeek-ViT undergoes a separate training stage before being integrated with the language backbone. The training pipeline consists of two stages: contrastive pre-training and autoregressive fine-tuning. During contrastive pretraining, we optimize the model using the sigmoid contrastive loss introduced by SigLIP ([Zhai et al., 2023](#bib.bib59)) on approximately 47B image-text pairs sourced from alt-text data. To efficiently learn visual representations from such massive datasets, we restrict the maximum input resolution to $224\times 224$ pixels by downscaling larger images while preserving their aspect ratios. Although using higher resolutions in this phase yields notable gains, empirical results show that these benefits contribute little to the final model. Because the subsequent autoregressive stage specifically handles high-resolution extrapolation, scaling up resolutions during contrastive pretraining significantly increases computational overhead without much overall improvement. In the autoregressive fine-tuning stage, we connect the vision encoder to a 4B MoE LLM and train on 236B tokens across datasets including image captions, alt text, charts, and OCR, using a next-token prediction objective. This stage aims to enhance the encoder's ability to model fine-grained visual features. We therefore constrain the input resolution between $544\times 544$ and $1344\times 1344$ pixels by proportionally scaling out-of-bound images. After this stage, we discard the LLM and retain only the optimized vision encoder for the subsequent pre-training pipeline, where the same input-resolution policy is maintained.

### 4.3 Evaluations

#### 4.3.1 Evaluation Benchmarks

We compare DeepSeek-V4.1-Flash-Base with its predecessor models DeepSeek-V4-Flash-Base and DeepSeek-V4-Pro-Base. We report benchmarks spanning five key dimensions: world knowledge, language understanding and reasoning, coding and mathematics, long context, and multimodal abilities.

World knowledge benchmarks include AGIEval ([Zhong et al., 2023](#bib.bib11)), MMLU-Pro ([Wang et al., 2024b](#bib.bib29)), C-Eval ([Huang et al., 2023](#bib.bib9)), MultiLoKo ([Hupkes and Bogoychev, 2025](#bib.bib48)), SimpleQA-Verified ([Haas et al., 2025](#bib.bib45)) and SuperGPQA ([Du et al., 2025](#bib.bib46))，

Language understanding and reasoning benchmarks include BigBench Hard (BBH) ([Suzgun et al., 2022](#bib.bib7)), BigBench Extra Hard (BBEH) ([Kazemi et al., 2025](#bib.bib57)), DROP ([Dua et al., 2019](#bib.bib8)) and HellaSwag ([Zellers et al., 2019](#bib.bib13)),

Coding and mathematical benchmarks include BigCodeBench ([Zhuo et al., 2025](#bib.bib49)), HumanEval ([Chen et al., 2021](#bib.bib12)), GSM8K ([Cobbe et al., 2021](#bib.bib10)), MATH ([Hendrycks et al., 2021](#bib.bib14)) and MGSM ([Shi et al., 2023](#bib.bib30)),

Long context benchmark includes LongBench-V2 ([Bai et al., 2025](#bib.bib47)).

Multimodal benchmarks include MMMU-Pro ([Yue et al., 2025](#bib.bib85)), DocVQA ([Mathew et al., 2021](#bib.bib87)), CVBench ([Tong et al., 2024](#bib.bib86)) and RefCOCO/RefCOCO+/RefCOCO-g ([Kazemzadeh et al., 2014](#bib.bib88); [Nagaraja et al., 2016](#bib.bib89); [Mao et al., 2016](#bib.bib90); [Yu et al., 2016](#bib.bib91)).

CAPTION:: Table 1: Comparison among DeepSeek-V4-Flash-Base, DeepSeek-V4-Pro-Base, and DeepSeek-V4.1-Flash-Base. All models are evaluated in our internal framework and share the same evaluation setting. Scores with a gap not exceeding 0.3 are considered to be at the same level. The highest score in each row is in bold font, and the second is underlined.

+---------------+-------------------------+----------+-------------------+-----------------+---------------------+
|               | Benchmark (Metric)      | \# Shots | DeepSeek-V4-Flash | DeepSeek-V4-Pro | DeepSeek-V4.1-Flash |
+---------------+                         |          +-------------------+-----------------+---------------------+
|               |                         |          | Base              | Base            | Base                |
+---------------+-------------------------+----------+-------------------+-----------------+---------------------+
|               | Architecture            | -        | MoE               | MoE             | MoE                 |
+---------------+-------------------------+----------+-------------------+-----------------+---------------------+
|               | \# Activated Params     | -        | 13B               | 49B             | 8B/16B              |
+---------------+-------------------------+----------+-------------------+-----------------+---------------------+
|               | \# Backbone Params      | -        | 284B              | 1.6T            | 552B                |
+---------------+-------------------------+----------+-------------------+-----------------+---------------------+
| World Knowl.  | AGIEval (EM)            | 3-5-shot | 83.9              | 84.4            | 83.4                |
|               +-------------------------+----------+-------------------+-----------------+---------------------+
|               | MMLU-Pro (EM)           | 5-shot   | 68.3              | 73.5            | 74.1                |
|               +-------------------------+----------+-------------------+-----------------+---------------------+
|               | C-Eval (EM)             | 5-shot   | 92.1              | 93.1            | 92.1                |
|               +-------------------------+----------+-------------------+-----------------+---------------------+
|               | MultiLoKo (LLM-Judge)   | 5-shot   | 42.6              | 50.9            | 45.5                |
|               +-------------------------+----------+-------------------+-----------------+---------------------+
|               | Simple-QA verified (EM) | 25-shot  | 30.1              | 55.2            | 42.3                |
|               +-------------------------+----------+-------------------+-----------------+---------------------+
|               | SuperGPQA (EM)          | 5-shot   | 46.5              | 53.9            | 53.1                |
+---------------+-------------------------+----------+-------------------+-----------------+---------------------+
| Lang. & Reas. | BBH (EM)                | 3-shot   | 86.9              | 87.5            | 86.1                |
|               +-------------------------+----------+-------------------+-----------------+---------------------+
|               | BBEH (EM)               | 1-shot   | 25.4              | 29.8            | 27.2                |
|               +-------------------------+----------+-------------------+-----------------+---------------------+
|               | DROP (F1)               | 1-shot   | 88.6              | 88.7            | 87.9                |
|               +-------------------------+----------+-------------------+-----------------+---------------------+
|               | HellaSwag (EM)          | 0-shot   | 85.7              | 88.0            | 87.2                |
+---------------+-------------------------+----------+-------------------+-----------------+---------------------+
| Code & Math   | BigCodeBench (Pass@1)   | 3-shot   | 56.8              | 59.2            | 60.6                |
|               +-------------------------+----------+-------------------+-----------------+---------------------+
|               | HumanEval (Pass@1)      | 0-shot   | 69.5              | 76.8            | 79.4                |
|               +-------------------------+----------+-------------------+-----------------+---------------------+
|               | GSM8K (EM)              | 8-shot   | 90.8              | 92.6            | 93.0                |
|               +-------------------------+----------+-------------------+-----------------+---------------------+
|               | MATH (EM)               | 4-shot   | 57.4              | 64.5            | 61.1                |
|               +-------------------------+----------+-------------------+-----------------+---------------------+
|               | MGSM (EM)               | 8-shot   | 85.7              | 84.4            | 80.2                |
+---------------+-------------------------+----------+-------------------+-----------------+---------------------+
| Long Context  | LongBench-V2 (EM)       | 1-shot   | 44.7              | 51.5            | 45.2                |
+---------------+-------------------------+----------+-------------------+-----------------+---------------------+
| Multimodal    | MMMU-Pro (EM)           | 4-shot   | -                 | -               | 56.5                |
|               +-------------------------+----------+-------------------+-----------------+---------------------+
|               | CVBench (EM)            | 4-shot   | -                 | -               | 77.9                |
|               +-------------------------+----------+-------------------+-----------------+---------------------+
|               | DocVQA (LLM-Judge)      | 4-shot   | -                 | -               | 95.6                |
|               +-------------------------+----------+-------------------+-----------------+---------------------+
|               | RefCOCO-avg (Acc@0.5)   | 0-shot   | -                 | -               | 86.0                |
+---------------+-------------------------+----------+-------------------+-----------------+---------------------+

#### 4.3.2 Evaluation Results

In Table [1](#S4.T1), we provide a detailed comparison of the base models for DeepSeek-V4-Flash, DeepSeek-V4-Pro and DeepSeek-V4.1-Flash, all evaluated under our internal evaluation framework using strictly controlled and reproducible settings. Compared with DeepSeek-V4-Flash-Base and DeepSeek-V4-Pro-Base, our latest base model reveals a compelling efficiency gain. DeepSeek-V4.1-Flash activates a substantially smaller number of parameters than DeepSeek-V4-Pro-Base and occupies a heavily-reduced KV cache, yet its performance is fully on par with its predecessors. These results also reflect the substantial improvements we made to our pre-training data curation pipeline. In this version, we introduce native multimodal training and validate its effectiveness through corresponding multimodal evaluations. Trained on a more diverse and multimodal corpus, DeepSeek-V4.1-Flash achieves world knowledge and comprehension capabilities comparable to those of DeepSeek-V4-Pro. In reasoning and coding benchmarks, DeepSeek-V4.1-Flash shows consistent progress, reaching performance close to or better than DeepSeek-V4-Pro across multiple benchmarks.

To further assess the model's capabilities in real-world R&D scenarios, we additionally perform perplexity tests on dedicated internal corpora. Since perplexity tests are impossible through model APIs, we mainly focus on our own pretrained base models. For corpus selection, a separate evaluation set is collected based on our daily development, including internal documentation, proprietary code repositories and academic materials, which targets reasoning, attribution and problem-solving on complex scientific problems and frontier research. The results are shown in Figure [6](#S4.F6), where we report the bits-per-byte(BPB) of different models, with lower values indicating better performance.

![Refer to caption](2609.19969v1/figures/pretrain_inhouse_ppl.png)

CAPTION:: Figure 6: Bits-per-bytes (BPB) comparison of DeepSeek-V4-Flash-Base, DeepSeek-V4-Pro-Base and DeepSeek-V4.1-Flash-Base on our held-out evaluation sets. DeepSeek-V4.1-Flash-Base achieves lowest BPB on all tasks and demonstrates greater potential to serve as a strong base-model.

