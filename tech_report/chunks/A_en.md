\\reportnumber

001

# DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression

DeepSeek-AI\
research@deepseek.com

###### Abstract

The widespread adoption of long-horizon agents has made model workloads increasingly input-heavy. Although prior work has substantially reduced the cost of long-context computation, prefill remains computationally expensive, and large KV caches continue to strain HBM and SSD capacity and data-transfer bandwidth. Together, these compute, storage, and bandwidth demands constitute the primary bottleneck to further lowering deployment costs. To address this challenge, we introduce DeepSeek-V4.1-Flash, a multimodal Mixture-of-Experts (MoE) model with 552B backbone parameters and support for contexts of up to one million tokens. With its Causal Encoder-Decoder (CED) architecture, the model activates 16B parameters per token during decode but only 8B parameters during prefill, substantially improving cost efficiency for agentic workloads. To push the limits of KV cache compression, DeepSeek-V4.1-Flash combines cross-layer KV cache reuse in Compressed Sparse Attention 2 (CSA2) with FP4 KV caching. These designs reduce its global KV cache footprint (always in HBM) to 890 bytes per token, roughly 1/4 of the corresponding footprint of DeepSeek-V4-Flash. Further, through a dedicated deployment optimization known as SWA Bounded Replay, DeepSeek-V4.1-Flash reduces its persistent KV cache footprint (always on SSD or in host memory) to roughly 1/8 of that of DeepSeek-V4-Flash. Despite its much smaller KV cache footprint, the model delivers substantially better performance than the baseline. In addition, we streamline the DeepSeek-V4 architecture and introduce several efficient architectural extensions. We pretrain DeepSeek-V4.1-Flash on a multimodal corpus comprising 45T tokens and conduct comprehensive post-training, yielding strong performance across diverse text-based and multimodal agentic scenarios. Model checkpoints are available at [https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).


CAPTION:: Figure 1: (a) Performance of DeepSeek-V4.1-Flash and its counterparts on agentic benchmarks. (b) Global KV cache size per token (in bytes) across generations of DeepSeek models, highlighting DeepSeek's sustained efforts to reduce context memory requirements. DeepSeek-V4.1-Flash achieves approximately 4-fold and 437-fold reductions in per-token global KV cache size relative to DeepSeek-V4-Flash and DeepSeek-V1, respectively.

## 1 Introduction

Applications of long-horizon agents have expanded rapidly in recent years, making ultra-long-context processing an increasingly important model workload. Supporting such workloads requires not only efficient long-sequence processing, but also the persistent storage, reuse, and transfer of large KV caches. KV cache management has therefore become a foundational capability for model deployment, while introducing substantial challenges across computation, storage, and communication. Prior advances in sparse attention ([DeepSeek-AI, 2025](#bib.bib36); [DeepSeek-AI, 2026b](#bib.bib34)) have significantly reduced the computational cost of long-sequence processing, making persistent storage and data movement increasingly prominent bottlenecks.

Specifically, DeepSeek-V4 ([DeepSeek-AI, 2026b](#bib.bib34)) combines a global attention branch spanning the full context with local Sliding-Window Attention (SWA). The global branch maintains global KV, comprising main KV and indexer K, while SWA maintains local KV states. For a fixed window size, SWA KV storage is bounded independently of sequence length. For sufficiently long sequences, global KV therefore dominates the runtime KV footprint, which is constrained by HBM capacity. In addition, certain KV are persisted for prefix reuse, referred to as persistent KV caches, which are constrained by SSD and host memory capacity. I/O and interconnect bandwidth also limit cache migration and loading. Together, these constraints limit serving throughput, increase deployment costs, and ultimately hinder the deployment and adoption of agents over longer task horizons and across broader application scenarios.

Further reducing the KV cache footprint is therefore critical to alleviating storage and communication bottlenecks and lowering the cost of long-context serving. To this end, we develop DeepSeek-V4.1-Flash, a multimodal Mixture-of-Experts (MoE) model designed for more aggressive KV cache compression. DeepSeek-V4.1-Flash has 552B backbone parameters, natively supports multimodal inputs, and accommodates contexts of up to one million tokens. We adopt a Causal Encoder-Decoder (CED) architecture, in which decoder global KV is projected from the final encoder hidden states. This design enables the model to activate 8B parameters per token during prefill and 16B during decode, which is particularly cost-effective for input-heavy agentic scenarios. Despite being considerably larger than DeepSeek-V4-Flash, DeepSeek-V4.1-Flash requires only approximately 1/4 as much runtime KV cache storage and 1/8 as much persistent KV cache storage at the same sequence length. Moreover, DeepSeek-V4.1-Flash delivers better overall performance than DeepSeek-V4-Flash.

This level of KV cache compression is achieved through joint optimizations in model architecture, cache precision, and deployment strategy. Conceptually, DeepSeek-V4 can be viewed as an SWA-based local-processing backbone augmented with compressed global context. This perspective motivates us to focus on simplifying the global branch while largely preserving the local attention design. At the architectural level, we design Compressed Sparse Attention 2 (CSA2), which applies cross-layer reuse to global KV (including main KV and indexer K) and Top-K indices to substantially reduce KV cache storage. CSA2 has three statically assigned modes: Full, Reindex, and Reuse. Full Mode generates global KV and performs indexing. Reindex Mode reuses the global KV from a preceding layer, and uses its own indexer Q to rescore the shared indexer K and select fresh Top-K indices. Reuse Mode reuses both global KV and the Top-K indices in a preceding layer, and directly performs sparse attention. In all three modes, each layer retains its own global Q and SWA KV. Sharing global KV and indexer K reduces duplicated cache storage. In addition, different from DeepSeek-V4 that employs the Compressed Sparse Attention (CSA)--Heavily Compressed Attention (HCA) hybrid architecture, DeepSeek-V4.1-Flash uses pure CSA2. At the cache-precision level, we use FP4 global KV caches during training with only marginal performance degradation. Together, CSA2 and FP4 KV caching reduce global KV cache storage to approximately 1/4 of that of DeepSeek-V4-Flash, as shown in Figure [1](#S0.F1)(b). At the deployment level, DeepSeek-V4.1-Flash, like DeepSeek-V4, uses Sliding-Window Attention (SWA) in every layer. In DeepSeek-V4, we use a hybrid strategy to balance the storage cost of persisting SWA KV caches against the computation required for exact reconstruction. Exact reconstruction requires replaying the most recent $L\times n_{\mathrm{win}}$ tokens, where $L$ is the number of layers and $n_{\mathrm{win}}$ is the SWA window size. In DeepSeek-V4.1-Flash, we introduce SWA Bounded Replay, which approximately reconstructs the required SWA KV states by replaying only the most recent $n_{\mathrm{win}}$ tokens. Our experiments show that this incurs only negligible performance degradation. This finding establishes a new storage--computation trade-off, allowing us to avoid persisting SWA KV cache to SSD while incurring a small amount of prefill recomputation. With SWA Bounded Replay, the persistent KV cache footprint is further reduced to approximately 1/8 of that of DeepSeek-V4-Flash. Together, these optimizations greatly ease pressure on HBM and SSD capacity, reduce deployment costs, and pave the way for deployment at a larger scale.

![Refer to caption](2609.19969v1/figures/decode_flops_curves.png)

CAPTION:: Figure 2: Single-token Decode FLOPs versus context length across generations of DeepSeek models. We account for compute precision by weighting BF16, FP8, and FP4 operations by $1$, $0.5$, and $0.25$, respectively. DeepSeek-V4.1-Flash maintains nearly constant Decode FLOPs as context length increases, substantially reducing the computational cost of long-context scenarios.

Complementing CED and CSA2, we further streamline the original DeepSeek-V4 architecture. Additionally, we upgrade the original mHC ([Xie et al., 2026](#bib.bib42)) design to Single-Pass mHC, with an accompanying Mega-mHC deployment kernel that halves activation memory traffic relative to the original four-kernel implementation. Furthermore, we integrate the Engram ([Cheng et al., 2026b](#bib.bib50)) conditional memory module to strengthen model capabilities. We also introduce the DSpark ([Cheng et al., 2026a](#bib.bib1)) speculative decoding architecture to improve decoding efficiency through semi-autoregressive draft generation and confidence-scheduled verification. With all the architectural improvements combined, the single-token Decode FLOPs of DeepSeek-V4.1-Flash remain nearly constant across context lengths. Figure [2](#S1.F2) shows that extending the context length 256-fold, from 4K to 1M, increases its Decode FLOPs by only 1/4, significantly less than the growth observed for DeepSeek-V4-Flash.

To fully realize the KV cache compression benefits of these architectural designs and further improve training and inference efficiency, we systematically co-optimize the training infrastructure and inference system for DeepSeek-V4.1-Flash, ensuring efficient and scalable large-scale multimodal training and long-context deployment. Training infrastructure supports disaggregated vision-encoder execution, balanced image sharding for long sequences, and cross-stage shared-state management for attention reuse. The inference system implements Encoder and Decoder SWA Bounded Replay paths. Further optimizations include communication--computation overlap, sharded Engram embedding tables, and inference kernel fusion. In particular, each CSA2 Reuse Mode layer executes with only 15 kernels during prefill and 11 during decode. We also separate long-lived global KV storage from short-lived encoder SWA KV in host memory, using bounded replay to approximately reconstruct missing encoder SWA states.

During pre-training, we train DeepSeek-V4.1-Flash on a large-scale multimodal corpus comprising 45T tokens. Sparse attention is trained from scratch at a sequence length of 64K, without any dense attention warmup stages. After pre-training, the model possesses native multimodal capabilities and supports contexts of up to one million tokens. In our evaluations, DeepSeek-V4.1-Flash-Base achieves world knowledge, reasoning and coding abilities comparable to DeepSeek-V4-Pro-Base, and delivers 5%--10% improvements on held-out evaluations, using only 1/3 total parameters and 1/4 activated parameters. Together, these results highlight its strong parameter efficiency and reflect improvements in training data quality for real-world deployment.

Building on this base model, we conduct post-training to elicit its reasoning and agentic capabilities. In contrast to the architectural innovations described above, our post-training introduces no algorithmic innovation: the recipe follows the standard paradigm of supervised fine-tuning (SFT) followed by reinforcement learning (RL) and on-policy distillation (OPD), without any modification beyond well-established practice used in DeepSeek-V4 development ([DeepSeek-AI, 2026b](#bib.bib34)). All substantive changes lie instead in the data pipeline. We develop large-scale automated pipelines for data synthesis and environment construction, and progressively scale the data, tasks, and rollouts employed during RL, thereby extending the model's capabilities across textual, multimodal, and agentic domains. Figure [1](#S0.F1)(a) summarizes DeepSeek-V4.1-Flash's performance on core agentic benchmarks. Our evaluation shows that, despite its compact size, DeepSeek-V4.1-Flash exhibits a distinctive capability profile:

- •
  Reasoning. The model delivers strong reasoning ability, sustaining high accuracy on reasoning-intensive benchmarks such as mathematics and competitive programming, showing comparable performance with top open-source models, such as Kimi-K3([Team et al., 2026a](#bib.bib67)) and DeepSeek-V4-Pro.
- •
  Agent. DeepSeek-V4.1-Flash achieves performance on par with closed-source frontier models across standard agentic benchmarks like Terminal-Bench 2.1 ([Merrill et al., 2026](#bib.bib21)), DeepSWE v1.1 ([DataCurve, 2026](#bib.bib15)), and AutomationBench ([Shepard and Salimans, 2026](#bib.bib19)). It has proven fully capable of handling everyday coding tasks and white-collar workflows. However, a gap with giant models remains on science-oriented agentic tasks, such as Terminal-Bench 4.0 ([Marten et al., 2026a](#bib.bib75)), that require expert-level domain knowledge.
- •
  Multimodal. Within the multimodal domain, the model surpasses top-tier open-source competitors like Kimi-K3 specifically on benchmarks evaluating visual reasoning and the interpretation of professional charts. Beyond formal metrics, it also exhibits practical utility in real-world visual agentic workflows, such as frontend development and office automation, where it can utilize rendered screen captures for visual inspection and self-correction. Nevertheless, we acknowledge that a distinct overall performance gap remains when compared to giant closed-source systems.

These results indicate that DeepSeek-V4.1-Flash can already match closed-source frontier models on the vast majority of benchmarks, and is capable of completing over 95% of real-world tasks. Meanwhile, its small activation footprint yields low inference latency and serving cost. We therefore believe that DeepSeek-V4.1-Flash offers a favorable trade-off between capability and efficiency, and can serve as a fast, affordable assistant supporting the daily work of a broad population of users. In summary, DeepSeek-V4.1-Flash simultaneously improves model intelligence and inference efficiency while reducing deployment costs. It substantially lowers the cost barrier to deploying long-horizon agents at scale and creates new opportunities for their adoption across a broader range of scenarios. DeepSeek-V4.1-Flash also serves as a new starting point for our continued scaling efforts. Building on this foundation, we will pursue the joint scaling of model architecture, pre-training, and post-training to further explore the frontier of model intelligence.

CAPTION:: Figure 3: Overall architecture of DeepSeek-V4.1-Flash. The 40-layer network is divided into a causal encoder and a decoder, each with 20 layers. All feed-forward layers use standard DeepSeekMoE. The first two encoder layers use sliding window attention (SWA); the rest use Compressed Sparse Attention 2 (CSA2), with CSA2(ratio, mode) specifying the compression ratio and mode. The model also uses Single-Pass mHC, Engram, DSpark, and a Hierarchical Sparse Indexer.

## 2 Architecture

### 2.1 Overview

DeepSeek-V4.1-Flash is a multimodal mixture-of-experts (MoE) Transformer that takes images and text as input and generates text autoregressively. Its language backbone comprises 40 causal Transformer layers, organized into a 20-layer causal encoder followed by a 20-layer decoder. Each layer incorporates both global attention and sliding window attention (SWA), except for the first two layers, which use SWA only. A vision encoder and an MLP projector convert images into visual embeddings that are processed jointly with text embeddings, with multimodal data incorporated from the start of language-model pre-training. Overall, DeepSeek-V4.1-Flash has 552B backbone parameters and 196B Engram parameters, activating 8B parameters per token during prefill and 16B during decode. Figure [3](#S1.F3) illustrates the overall architecture of DeepSeek-V4.1-Flash.

The Causal Encoder--Decoder (CED) architecture and Compressed Sparse Attention 2 (CSA2) address complementary costs of long-context inference. CED constructs the decoder's global key-value (KV) cache from encoder outputs, allowing most prompt tokens to bypass full decoder computation while retaining layer-local sliding-window attention. This nearly halves prefill computation, lowering the cost of processing new or uncached inputs in agentic workloads with growing contexts. CSA2 shares global KV across layers to reduce cache storage and reuses sparse selections to reduce indexing work. In the decoder, a Hierarchical Sparse Indexer restricts later indexers to a candidate pool selected by an earlier indexer, further reducing the number of entries scored per query.

We retain the shared and fine-grained routed experts of DeepSeekMoE ([Dai et al., 2024](#bib.bib24)), and introduce modality-specific load balancing ([Wang et al., 2024a](#bib.bib28)) for image and text tokens. Single-Pass mHC ([Xie et al., 2026](#bib.bib42)) revises residual-stream mixing to enable more efficient kernel fusion, and Engram ([Cheng et al., 2026b](#bib.bib50)) adds sparsely accessed conditional memory. We omit the MTP module during backbone pre-training and use DSpark ([Cheng et al., 2026a](#bib.bib1)) for speculative decoding. We train DSpark separately after the backbone pre-training stage. Additionally, we compress the main KV cache to FP4 to further reduce storage overhead. The following sections describe these components and the corresponding optimization changes.

#### 2.1.1 Multimodal Architecture

The multimodal input pathway comprises a vision encoder and an MLP projector. For each input image, the vision encoder produces a spatial grid of visual features. A $3\times 3$ pixel-unshuffle operation then rearranges each local neighborhood along the channel dimension, reducing the spatial resolution before the MLP projector maps the features to the hidden dimension of the language backbone. Finally, the resulting visual embeddings are inserted at the corresponding image-token positions in the input embedding sequence and processed jointly with text embeddings by the language backbone.

##### DeepSeek-ViT

We train a vision encoder named DeepSeek-ViT from scratch to natively process images at varying resolutions. We build DeepSeek-ViT on the Vision Transformer ([Dosovitskiy et al., 2021](#bib.bib79)) architecture with several modifications. To accommodate inputs of arbitrary resolutions, we replace standard absolute positional embeddings with 2D-RoPE. To align the ViT more closely with LLM design principles, we replace the patch embedding layer's convolution with a linear projection to ensure compatibility with the Muon optimizer. We also adopt RMSNorm ([Zhang and Sennrich, 2019](#bib.bib80)) for normalization and SwiGLU ([Shazeer, 2020](#bib.bib23)) as the activation function. Before feeding visual features into the LLM, we apply a pixel-unshuffle operation with $3\times 3$ downsampling to reduce the visual token count by a factor of nine, effectively supporting input resolutions up to approximately $1344\times 1344$ pixels.

##### Multimodal Auxiliary-loss-free Load Balancing for MoEs

Image and text tokens exhibit distinct representation distributions and may induce different expert-routing preferences in MoEs. Balancing their aggregate load may therefore obscure modality-specific imbalance. To address this issue, we extend auxiliary-loss-free load balancing ([Wang et al., 2024a](#bib.bib28)) by maintaining separate expert-wise correction biases for text and image tokens. During routing, each token uses the correction biases associated with its modality for expert selection, while the original routing scores are retained for weighting the selected expert outputs. After each training step, the two sets of biases are updated independently according to their respective expert loads. This design balances expert utilization within each modality and contributes to stable and efficient multimodal training.

### 2.2 Causal Encoder-Decoder (CED)

In agentic workflows, frequent tool calls generate extensive prefill requests, imposing severe computational overhead when KV caches miss. To alleviate this prefill bottleneck, we propose the Causal Encoder-Decoder (CED) architecture, inspired by YOCO ([Sun et al., 2024](#bib.bib53)). YOCO reduces prefill computation by allowing the upper half of the layers to directly share the KV cache generated by the lower half. Building upon this concept, CED introduces a series of structural improvements to enhance both the overall KV cache capacity and the computational depth of KV generation. Consequently, CED successfully reduces nearly half of the prefill computation while maintaining performance comparable to the baseline.

For global attention, CED treats the bottom $L/2$ layers of the Transformer as the causal encoder. For the upper half layers (i.e., the decoder, $l>L/2$), the KV entries are not derived from their respective hidden states $H_{l}$. Instead, they are projected directly from the hidden state of the $(L/2)$-th layer, $H_{L/2}$, using layer-dependent projection weights ($W_{l}^{KV}$ and $W_{l}^{Z}$):

  -- ------------------------------------------------------------------------------- -- -----
     $$C_{l}=H_{L/2}W_{l}^{KV},\quad Z_{l}=H_{L/2}W_{l}^{Z},\quad l>\frac{L}{2},$$      (1)
  -- ------------------------------------------------------------------------------- -- -----

where $C$ and $Z$ represent the KV entries and their corresponding compression weights, respectively. This design allows CED to compute only the first half of the layers during the prefill phase, acquiring the upper-layer global KV cache with minimal computational cost.

For sliding window attention (SWA), CED maintains the conventional layer-wise computation across all layers. Specifically, for any layer $l$, the local keys and values are derived directly from the current layer's hidden state $H_{l}$. This design effectively increases the computational depth of local KV generation. However, maintaining this layer-wise computation necessitates an SWA replay process. During the prefill phase, computing the SWA KV cache for the decoder requires processing an additional $n_{\mathrm{win}}\times L/2$ tokens (where $n_{\mathrm{win}}$ denotes the window size). For multi-turn interactions with short prompts per turn, this computational overhead in the decoder becomes non-negligible. Fortunately, prior work ([Chen et al., 2025](#bib.bib58)) has shown that the actual effective receptive field of SWA is much smaller than the theoretical $n_{\mathrm{win}}\times L/2$. Motivated by this observation, we introduce Decoder SWA Bounded Replay, which only prefills the last $n_{\mathrm{win}}$ tokens of the prompt for the SWA computation, thereby significantly reducing the computational cost. Further details are provided in Section [3.2.2](#S3.SS2.SSS2).

Overall, for a sequence length $N\gg n_{\mathrm{win}}$, CED reduces the prefill complexity from $\mathcal{O}(NL)$ to $\mathcal{O}(NL/2+n_{\mathrm{win}}\times L/2)\approx\mathcal{O}(NL/2)$, effectively halving the overall computation.

CAPTION:: Figure 4: Three operating modes of CSA2. The modes differ in how they obtain main KV, indexer K, and Top-K indices. Green blocks indicate quantities computed in the current layer; yellow blocks indicate main KV and indexer K reused from the most recent Full Mode layer; while red blocks indicate Top-K indices reused from the most recent index-producing (Full or Reindex Mode) layer. All three modes compute main Q and SWA KV in the current layer.

### 2.3 Compressed Sparse Attention 2 (CSA2)

Serving long contexts requires controlling both KV cache storage and attention computation. These costs can be reduced along three multiplicative dimensions: the entry size, where GQA ([Ainslie et al., 2023](#bib.bib3)) reduces the number of KV heads and MLA ([DeepSeek-AI, 2024a](#bib.bib27)) shares a small latent across heads; the sequence dimension, where every $m$ tokens are compressed into one entry, like CSA and HCA in DeepSeek-V4 ([DeepSeek-AI, 2026b](#bib.bib34)); and the layer dimension, where some layers reuse the caches ([Brandon et al., 2024](#bib.bib73)) and selections of other layers instead of keeping their own, or are replaced altogether by more efficient layers. Prior work has shown that compression along the layer dimension is effective: IndexCache ([Bai et al., 2026](#bib.bib69)) reuses Top-K indices across layers to cut indexer computation; YOIO ([Sun et al., 2026b](#bib.bib70)) computes the sparse routing once and shares it across all layers; and HySparse ([Gao et al., 2026](#bib.bib71)) lets sparse layers reuse the KV cache of dense layers. However, index reuse alone saves no main KV storage, network-wide routing sharing limits performance, and hybrid designs still retain full attention layers; more importantly, none of these methods covers all three multiplicative dimensions.

CSA2 exploits the three dimensions jointly: it shares main KV and indexer K across layers and allows layers to reuse Top-K indices, with cache sharing and index reuse decoupled. It combines these reuse strategies with a simplified compressor and a Hierarchical Sparse Indexer that narrows the search domain of subsequent indexing layers in the Decoder.

Similar to CSA, CSA2 includes a lightweight indexer that scores the main KV entries using indexer Q and indexer K and selects the Top-K entries for each query. Each Q attends to the selected entries together with the layer-local sliding-window KV (SWA KV). CSA2 also includes the uncompressed main KV setting as a special case with a compression ratio of 1. Meanwhile, CSA2 simplifies both the compressor and the indexer. In CSA, a compression ratio of $m$ produces each main KV entry from $2m$ original KV cache entries, with overlapping source entries for adjacent compressed entries. It also includes absolute positional embedding to encode the positions of these $2m$ entries during compression. CSA2 removes this overlap and absolute positional embedding. In addition, CSA2 obtains indexer K by projecting main KV entries, replacing CSA's separate compression path from hidden states. Both designs simplify the implementation and increase the training efficiency.

Sections [2.3.1](#S2.SS3.SSS1) and [2.3.2](#S2.SS3.SSS2) describe the cross-layer reuse strategies and the Hierarchical Sparse Indexer, respectively.

#### 2.3.1 Cross-Layer KV and Index Reuse

Each CSA2 layer is statically assigned one of three modes: Full, Reindex, or Reuse. In all three modes, the layer computes its own query and SWA KV and uses them together with the selected main KV entries to produce a new attention output. The modes differ in how they obtain main KV, indexer K, and Top-K indices. Figure [4](#S2.F4) illustrates the three modes.

Full Mode. The layer computes its own main KV and indexer Q, projects indexer K from that main KV, and runs the indexer to produce fresh Top-K indices. It therefore executes the complete CSA2 computation path and has the same component responsibilities as a complete CSA layer in DeepSeek-V4.

Reindex Mode. The layer reuses the most recent available *main KV* from a preceding layer together with its corresponding *indexer K*. The indexer computes its own query, rescores the reused keys, and produces fresh Top-K indices. This allows the sparse selection to change across layers while main KV and indexer K remain shared.

Reuse Mode. The layer reuses the most recent available *main KV* and the latest *Top-K indices* computed against that main KV by a preceding layer in Full or Reindex Mode. It performs attention using this selection without computing indexer Q or evaluating index scores.

Sharing main KV and indexer K reduces cache storage, while reusing Top-K indices avoids additional indexer computation. Reindex Mode preserves cache sharing while allowing the selected entries to change across layers. When CSA2 is combined with CED, the decoder layer assigned to Full Mode computes its own global KV from the hidden state of the $(L/2)$-th layer, i.e. the last layer of the causal encoder. The Reindex and Reuse Modes are unchanged.

CAPTION:: Figure 5: Hierarchical Sparse Indexer. Each square represents a position; green squares mark selected indices, and blue rectangles mark blocks selected based on their maximum indexer scores. The decoder's first CSA2 layer in Full mode selects its own Top-512 indices and builds a shared candidate pool from the selected blocks for subsequent layers. CSA2 layers in Reindex mode then select their Top-512 indices from this pool.

#### 2.3.2 Hierarchical Sparse Indexer

Cross-layer index reuse reduces the number of indexer evaluations, but the remaining indexers still score the full causally visible context. For extremely long contexts, this cost remains a major computational bottleneck. Prior work introduced indexer sparsity by scoring and pruning pooled block representations before token-level indexing ([Xu et al., 2026b](#bib.bib72)). We find that in the decoder, information from shallower indexers can naturally be used to restrict the candidates considered by deeper indexers without adding any extra state. We therefore introduce the Hierarchical Sparse Indexer, which is used only in the decoder of CED to reduce this repeated scoring during decode. For each query, the first layer assigned to Full Mode constructs a candidate pool that later re-indexing layers use as their search domain. For a fixed candidate-pool size, this changes the per-query cost of deeper indexers from linear in context length to constant. The mechanism is training-aware and introduced in post-training: the candidate restriction is applied identically during training and inference, so deeper indexers are optimized under the same search domain they use at inference. Figure [5](#S2.F5) illustrates this process.

This first Full Mode layer scores all causally visible main KV positions and produces the Top-K indices for its own attention. It also performs blockwise candidate selection: each block is assigned the maximum index score among its positions, and the blocks with the highest scores are selected. It then collects the positions covered by the selected blocks into a candidate pool larger than the final Top-K set. For example, selecting 2,048 blocks with 8 positions each yields 16,384 candidate positions. This pool defines where later indexers search; the final Top-K selection determines which main KV entries each layer reads.

Subsequent layers in Reindex Mode score only the candidate positions for the corresponding query and select their own Top-K entries within that pool. Layers in Reuse Mode perform no new indexing and use the latest Top-K indices computed against the main KV they reuse. Thus, the candidate pool is shared across indexing layers, while their final selections can differ.

For a fixed candidate-pool size, the number of positions scored per query by each subsequent indexer is bounded independently of context length. The first Full Mode layer still scans the entire causally visible range. Hierarchical indexing therefore reduces the cost of later indexer evaluations while retaining the initial full-range pass.

### 2.4 Efficient Architectural Extensions

#### 2.4.1 Single-Pass mHC

In DeepSeek-V4, we introduced mHC ([Xie et al., 2026](#bib.bib42)), which maintains $n$ residual streams between adjacent Transformer blocks. For each token, we denote these streams by $X_{l}\in\mathbb{R}^{n\times d}$, where $l$ is the block index and $d$ is the hidden dimension. The streams are updated as follows:

  -- --------------------------------------------------------------------------------------------------------- -- -----
     $$X_{l+1}=B_{l}X_{l}+C_{l}\,\mathcal{F}_{l}(A_{l}X_{l}),\qquad(A_{l},B_{l},C_{l})=\mathcal{H}(X_{l}),$$      (2)
  -- --------------------------------------------------------------------------------------------------------- -- -----

where $A_{l}\in\mathbb{R}^{1\times n}$, $C_{l}\in\mathbb{R}^{n\times 1}$ and $B_{l}\in\mathbb{R}^{n\times n}$ are token-wise coefficients predicted from $X_{l}$. The coefficient predictor $\mathcal{H}$ includes normalization and projection.

Ideally, the residual transformation between two blocks is a single map from $(X_{l-1},Y_{l-1})$ to $(X_{l},\hat{X}_{l})$, where $\hat{X}_{l}=A_{l}X_{l}$ is the current block input and $Y_{l-1}=\mathcal{F}_{l-1}(\hat{X}_{l-1})$ is the previous block output. Such a map requires $(n+1)d$ reads and $(n+1)d$ writes, giving a lower bound of $(2n+2)d$ on activation memory traffic. In practice, DeepSeek-V4 uses a multi-pass implementation of ([2](#S2.E2)), with three kernels that execute sequentially due to data dependencies:

  -- ------------------------------------ ----------------------------------------------- -- --------------------------------------- -- -----
     $\displaystyle X_{l}$                $\displaystyle=B_{l-1}X_{l-1}+C_{l-1}Y_{l-1}$      Residual update, contraction over $n$      (3)
     $\displaystyle(A_{l},B_{l},C_{l})$   $\displaystyle=\mathcal{H}(X_{l})$                 Coefficients, contraction over $nd$        (4)
     $\displaystyle\hat{X}_{l}$           $\displaystyle=A_{l}X_{l}$                         Input mixing, contraction over $n$         (5)
  -- ------------------------------------ ----------------------------------------------- -- --------------------------------------- -- -----

The three kernels read $(n+1)d$, $nd$ and $nd$ values respectively and write $(n+1)d$ in total. Including the pre-norm in $\mathcal{F}_{l}$, the total activation memory traffic is $(4n+4)d$, twice the lower bound.

In $\mathcal{H}$, the normalization weights are folded into the projection weights offline, and the RMS division is applied after the projection. Two of the three stages can therefore share one traversal of the residual: the residual update does not require a reduction across the hidden dimension, so each tile of $X_{l}$ can be computed and immediately used to accumulate the projection outputs and the sum of squares needed to compute the RMS. Input mixing cannot be fused into this pass because $A_{l}$ is not available until the reduction over all hidden tiles is complete. It therefore requires a second read of $X_{l}$. This second pass can also incorporate input pre-norm. This two-pass implementation would require $(3n+2)d$ activation reads and writes in total, one additional read of $X_{l}$ compared with the lower bound.

We therefore introduce Single-Pass mHC, which shifts the input-mixing coefficients by one block, i.e. every block consumes the mixing coefficients produced by the previous one, so that the above dependency disappears:

  -- ----------------------------------------------------------------------------------------------------------- -- -----
     $$X_{l+1}=B_{l}X_{l}+C_{l}\,\mathcal{F}_{l}(A_{l-1}X_{l}),\qquad(A_{l},B_{l},C_{l})=\mathcal{H}(X_{l}).$$      (6)
  -- ----------------------------------------------------------------------------------------------------------- -- -----

Input mixing now uses $A_{l-1}$ instead of $A_{l}$, so it no longer depends on the coefficients computed from $X_{l}$. Each tile of $X_{l}$ can therefore be used immediately for both input mixing and coefficient prediction, without waiting for the full reduction. Empirically, this shift incurs negligible performance degradation.

For pre-training, we keep the existing multi-kernel implementation, since the shift only changes which mixing coefficients each block applies. For deployment, we fuse residual update, input mixing, and coefficient prediction into a single kernel, Mega-mHC. The kernel implements mHC with $(3n+2)d$ activation reads and writes and Single-Pass mHC with $(2n+2)d$ activation reads and writes. Mega-mHC processes $X_{l}$ in tiles along the hidden dimension. Each tile is used to compute the mixed input and to accumulate the quantities needed to predict $(A_{l},B_{l},C_{l})$ for the next block. The kernel also incorporates input pre-norm and FP8 conversion. The residual is thereby read once and written once, attaining the $(n+1)d$ reads and $(n+1)d$ writes of the ideal map and halving the activation memory traffic of our original implementation.

#### 2.4.2 Engram

We augment DeepSeek-V4.1-Flash with Engram ([Cheng et al., 2026c](#bib.bib2)), the conditional memory module introduced in our previous work to decouple memorization from computation. We follow the original Engram design---tokenizer compression, multi-head hashing, context-aware gating, and multi-branch integration---with two modifications. First, we omit the short causal convolution because its performance gains do not justify the added complexity in our inference stack. Second, we optimize the Engram embedding with momentum-based update followed by Sinkhorn balancing, as detailed in Section [2.5](#S2.SS5).

We allocate 196B Engram parameters evenly across two modules. Each module uses $N$-gram orders $\{2,3,4\}$, with 8 hash heads and a total embedding dimension of 2048 per order. Each head indexes a table of approximately 16M entries, with table sizes chosen to be distinct primes. Both the embedding tables and the key/value projections use FP8 precision. The modules are placed at layers 1 and 14 (zero-indexed) to balance memory usage across training pipeline stages. During inference, deterministic addressing enables embeddings to be prefetched from host memory via background RDMA transfers, with prefetching for the first module overlapping computation in the first Transformer block. Further implementation details for Engram training and inference are discussed in Section [3.1.3](#S3.SS1.SSS3).

#### 2.4.3 DSpark

We equip DeepSeek-V4.1-Flash with DSpark ([Cheng et al., 2026a](#bib.bib1)), a speculative decoding module that combines semi-autoregressive drafting with confidence-scheduled verification. The drafter comprises three Transformer blocks with a sliding attention window of 128 tokens. A single forward pass through these blocks computes base logits for five draft positions in parallel, while a lightweight Markov head models dependencies among the draft tokens. A confidence head predicts per-position conditional acceptance probabilities, which are used to estimate prefix survival probabilities. The scheduler combines these estimates with profiled engine throughput curves to dynamically select the verification length for each request, aiming to maximize expected system-wide token throughput under the current system load.

Unlike the MTP module in DeepSeek-V3 ([DeepSeek-AI, 2024b](#bib.bib33)), which is trained jointly with the backbone throughout pre-training, DSpark is introduced in a dedicated stage after pre-training. In this stage, we train only DSpark while keeping the backbone frozen. During post-training, we continue to train DSpark alongside the backbone, without propagating gradients from the DSpark objective into the backbone. This keeps DSpark aligned with the evolving policy, enabling it to accelerate both online serving and rollout generation for RL and OPD.

#### 2.4.4 FP4 Main KV Cache

Long-context agent workloads require large per-request KV caches, increasing serving costs. DeepSeek-V4 already uses quantization-aware training (QAT) ([Jacob et al., 2018](#bib.bib5)) for FP4 indexer queries and keys, accelerating index computation and reducing the indexer cache size. We adopt the OCP-standard MXFP4 format ([Rouhani et al., 2023](#bib.bib4)) to support as many hardware platforms as possible, despite the higher accuracy of alternative formats in our experiments. We now extend QAT to the main KV cache, where FP4 reduces storage rather than accelerates matrix multiplication. Dequantizing cached values before attention allows us to use a more accurate format without requiring native matrix-multiplication support for that format, preserving compatibility across hardware platforms.

Among the approximately four-bit formats evaluated, we select E2M1 with one E4M3 scale per 16 channels, following NVFP4 ([Alvarez et al., 2025](#bib.bib6)) but omitting its second-level global scale to balance accuracy and simplicity. Omitting this scale leaves ample dynamic range for the main KV cache: the format supports magnitudes up to $448\times 6=2688$, far above the cache's magnitude bound. In DeepSeek-V4.1-Flash, the largest trained RMSNorm weight magnitude is approximately $1$. After RMS normalization, the L2 norm of the $512$-channel KV latent is at most approximately $\sqrt{512}$. RoPE preserves this norm, so the maximum absolute value across channels after rotation is also bounded by approximately $\sqrt{512}\approx 22.6$. Besides, the maximum magnitude observed during training is around $10$. Therefore, omitting the global scale causes no measurable decrease in accuracy and simplifies the cache layout.

To enable FP4 main KV cache storage in DeepSeek-V4.1-Flash, we introduce QAT during post-training. The non-RoPE and RoPE components use the same quantization format. We quantize the cache after RoPE: quantizing before RoPE yields only a marginal accuracy improvement in our experiments and would introduce additional overhead during decoding. We retain FP8 for the SWA KV cache due to its sensitivity to quantization. Compared with the FP8 main KV cache in DeepSeek-V4, this format nearly halves the storage footprint, both in HBM and when offloaded to SSD.

### 2.5 Optimization

Building upon the optimization configuration used in DeepSeek-V4, we make some new modifications to better align with the architectural design.

First, we use head-wise Muon, where Query weights are split by head before applying the Muon update. Here, we briefly discuss the motivation for such a design. By viewing Muon as a preconditioned gradient descent, vanilla Muon uses one preconditioner for all heads, whereas head-wise Muon provides different preconditioners for different heads. This design can better handle the heterogeneity across attention heads [Zhang et al., 2024](#bib.bib64); [Zhang, 2026](#bib.bib66), Section 3. As a result, we observe that head-wise Muon outperforms vanilla Muon. The empirical advantage of head-wise Muon is also validated in GLM 5 ([Zeng et al., 2026](#bib.bib68)) and Kimi-K3 ([Team et al., 2026a](#bib.bib67)).

Second, applying Adam to the newly introduced Engram parameters substantially increases the optimizer-state memory footprint. To reduce memory usage during training, we instead optimize the Engram embedding tables, token embedding, and prediction head using a momentum-based update followed by Sinkhorn balancing. Sinkhorn balancing has previously been applied to linear-layer weight matrices in SinkGD ([Scetbon et al., 2025](#bib.bib61)); here, we extend it to these large parameter matrices. Like Muon, this approach requires only a momentum buffer while empirically outperforming Adam.

##### Basic Configurations.

We retain AdamW ([Loshchilov and Hutter, 2019](#bib.bib60)) for normalization-layer weights and other non-matrix parameters, including biases and scaling factors. We use Muon ([Jordan et al., 2024](#bib.bib31)) for the weight matrices of linear transformations in the language-model backbone, the Engram projection layers, and the vision-language projector. We use head-wise Muon for Query and Key weights. We apply decoupled weight decay and Nesterov momentum to Muon ([Nesterov, 1983](#bib.bib35); [Liu et al., 2025](#bib.bib32)); normalization-layer weights are also subject to weight decay, whereas biases and scaling factors are not. The Sinkhorn-balanced update also uses Nesterov momentum but does not apply weight decay. During pre-training, we keep the vision encoder frozen until the learning-rate decay stage, while its final normalization layer and the vision--language projector remain trainable. At the onset of learning-rate decay, we unfreeze the vision encoder and optimize it jointly with the LLM with a smaller learning rate.

##### Sinkhorn-Balanced Updates for Engram / Embedding / Prediction Head.

The complete procedure is summarized in Algorithm [1](#alg1). At a high level, it follows the same workflow as Muon, with Sinkhorn balancing taking the place of Newton--Schulz orthogonalization. We denote the larger matrix dimension by $m$, which corresponds to the vocabulary size for embedding tables and prediction heads, and denote the hidden dimension by $n$.

CAPTION:: Algorithm 1 Momentum update with Sinkhorn balancing

1: Weight $W_{t}\in\mathbb{R}^{m\times n}$, gradient $G_{t}$, momentum $M_{t-1}$, momentum coefficient $\beta$, base learning rate $\eta_{t}$, learning-rate correction $\gamma$, numerical constants $\varepsilon$ and $\tau$, and an odd number of normalization steps $K$

2: $M_{t}\leftarrow\beta M_{t-1}+(1-\beta)G_{t}$

3: $\widehat{G}_{t}\leftarrow\beta M_{t}+(1-\beta)G_{t}$ $\triangleright$ Nesterov momentum

4: $\rho_{i}\leftarrow\lVert\widehat{G}_{t,i,:}\rVert_{2}$ and $\bar{\rho}\leftarrow\frac{1}{m}\sum_{i=1}^{m}\rho_{i}$

5: $U^{(0)}\leftarrow\widehat{G}_{t}$

6: Set $U^{(0)}_{i,:}\leftarrow 0$ if $\rho_{i}\leqslant\tau\bar{\rho}$ $\triangleright$ Mask near-zero rows

7: for $k=1,\ldots,K$ do

8:   if $k$ is odd then

9:    for all $i=1,\ldots,m$ do

10:      $U^{(k)}_{i,:}\leftarrow U^{(k-1)}_{i,:}/(\lVert U^{(k-1)}_{i,:}\rVert_{2}+\varepsilon)$

11:    end for

12:   else

13:    for all $j=1,\ldots,n$ do

14:      $U^{(k)}_{:,j}\leftarrow U^{(k-1)}_{:,j}/(\lVert U^{(k-1)}_{:,j}\rVert_{2}+\varepsilon)$

15:    end for

16:   end if

17: end for

18: $\Delta_{t}\leftarrow\sqrt{n}\,U^{(K)}$ $\triangleright$ Convert unit row $\ell_{2}$ norm to unit row RMS

19: $\widetilde{\eta}_{t}\leftarrow\gamma\eta_{t}$ $\triangleright$ Match the update magnitude of Adam

20: $W_{t+1}\leftarrow W_{t}-\widetilde{\eta}_{t}\Delta_{t}$

Given the Nesterov momentum update $\widehat{G}_{t}$, Sinkhorn balancing finds diagonal scaling matrices $D_{r}$ and $D_{c}$ such that

  -- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----
     $$\Delta_{t}=\sqrt{n}\,U^{(K)}=\sqrt{n}\,D_{r}\widehat{G}_{t}D_{c},\qquad\frac{1}{n}\sum_{j=1}^{n}(\Delta_{t})_{ij}^{2}\approx 1,\qquad\frac{1}{m}\sum_{i=1}^{m}(\Delta_{t})_{ij}^{2}\approx 1,$$      (7)
  -- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- -- -----

Thus, the procedure approximately equalizes the row-wise and column-wise RMS of the update matrix. Here, one row corresponds to one token index or n-gram identity; and one column encodes one hidden feature. Sinkhorn balancing exploits this token--feature structure by normalizing along both rows and columns. For numerical stability, rows satisfying $\rho_{i}\leqslant\tau\bar{\rho}$ are masked. The factor $\sqrt{n}$ converts unit row $\ell_{2}$ norm into unit row-wise RMS. Separately, we adjust the effective learning rate as $\widetilde{\eta}_{t}=\gamma\eta_{t}$ to match the update magnitude of Adam. We set $\gamma=0.18$, which is close to the factor $0.2$ used in Moonlight ([Liu et al., 2025](#bib.bib32)).

More broadly, Sinkhorn balancing is closely related to optimizers that exploit matrix or tensor axis structure ([Shazeer and Stern, 2018](#bib.bib62); [Zhang et al., 2025a](#bib.bib63); [Wen et al., 2025](#bib.bib83); [Glentis et al., 2025](#bib.bib93); [Deng et al., 2026](#bib.bib81); [Yuan et al., 2026](#bib.bib82); [Xu et al., 2026a](#bib.bib84)). For example, Adafactor ([Shazeer and Stern, 2018](#bib.bib62)) conducts row- and column-wise normalization in a different manner, and Adam-mini ([Zhang et al., 2025a](#bib.bib63)) uses an alternative row-wise normalization for embedding tables and prediction head. These normalization strategies may differ in optimization performance and communication overhead. We leave more detailed investigation as a future direction.

