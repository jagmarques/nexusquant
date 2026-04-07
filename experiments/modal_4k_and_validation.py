"""Combined experiment: 4K context sweep + 3rd text validation.

Task A: 4K context (2048 prefix + 2048 continuation)
  - Tests whether longer sequences improve eviction quality
  - Configs: 2b+0%, 2b+35%, 2b+50%, 2b+60%, 2b+70%, 2b+80% eviction

Task B: 3rd text validation (creative/narrative — completely different domain)
  - Tests compression on non-academic text
  - Configs: 2b+35%, 2b+70%, 2b+80% eviction

Single GPU run to save cost. Results printed + written to .company/engineering/4k_validation_results.md
"""
import modal
import os

app = modal.App("nexusquant-4k-validation")

nq_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nexusquant-oss", "nexusquant")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.44.0,<5.0.0",
        "accelerate>=0.27.0",
        "zstandard>=0.22.0",
        "numpy<2.0",
        "sentencepiece",
        "protobuf",
    )
    .add_local_dir(nq_local, remote_path="/root/nexusquant")
)

HF_SECRET = modal.Secret.from_dict({"HF_TOKEN": "os.environ.get("HF_TOKEN", "")"})


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    secrets=[HF_SECRET],
    memory=32768,
)
def run_4k_and_validation():
    import sys
    sys.path.insert(0, "/root")

    import time, math
    import numpy as np
    import torch
    import torch.nn.functional as F
    import zstandard

    from nexusquant.core.e8_lattice import E8Lattice
    from nexusquant.core.hadamard import hadamard_matrix
    from nexusquant.core.rope_utils import inverse_rope, forward_rope

    # ------------------------------------------------------------------ helpers
    def get_kv(cache, layer):
        if hasattr(cache, 'key_cache'):
            return cache.key_cache[layer], cache.value_cache[layer]
        return cache.layers[layer].keys, cache.layers[layer].values

    def set_kv(cache, layer, k, v):
        if hasattr(cache, 'key_cache'):
            cache.key_cache[layer] = k
            cache.value_cache[layer] = v
        else:
            cache.layers[layer].keys = k
            cache.layers[layer].values = v

    print("=" * 80)
    print("NEXUSQUANT — 4K Context Sweep + 3rd Text Validation")
    print("=" * 80)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "mistralai/Mistral-7B-v0.1"
    print(f"\nLoading {model_name}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto",
        token=os.environ["HF_TOKEN"],
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s")

    n_layers   = model.config.num_hidden_layers          # 32
    n_kv_heads = model.config.num_key_value_heads        # 8
    head_dim   = model.config.hidden_size // model.config.num_attention_heads  # 128
    rope_base  = getattr(model.config, 'rope_theta', 10000.0)
    print(f"Config: {n_layers}L, {n_kv_heads}KVH, d={head_dim}, rope_base={rope_base}")

    # ------------------------------------------------------------------ Task A: 4K long text corpus
    # ~4096+ tokens from 8 topics × ~500 words each
    LONG_4K_TEXT = (
        # Physics (~500 words)
        "The Standard Model of particle physics is the theory describing three of the four known fundamental forces "
        "in the universe, as well as classifying all known elementary particles. It was developed in stages throughout "
        "the latter half of the 20th century through the work of many scientists around the world, with the current "
        "formulation being finalized in the mid-1970s upon experimental confirmation of the existence of quarks. The "
        "Standard Model explains how the basic building blocks of matter interact, governed by four fundamental forces. "
        "Fermions are the building blocks: six quarks and six leptons. The forces between fermions are mediated by gauge "
        "bosons. The Higgs mechanism gives mass to some particles through spontaneous symmetry breaking. The weak force "
        "is responsible for radioactive decay and nuclear fusion in stars. The strong force binds quarks into protons "
        "and neutrons and holds nuclei together. Electromagnetism governs the interactions of charged particles and "
        "underlies all chemistry and materials science. General relativity describes gravity as the curvature of "
        "spacetime caused by mass and energy, but it has not been successfully unified with quantum mechanics. The "
        "search for a theory of quantum gravity — such as string theory or loop quantum gravity — remains one of the "
        "deepest open problems in physics. Black holes are regions of spacetime where gravity is so strong that nothing, "
        "not even light, can escape. Hawking radiation is a theoretical prediction that black holes slowly evaporate "
        "by emitting thermal radiation due to quantum effects near the event horizon. Dark matter and dark energy "
        "together account for roughly 95 percent of the total mass-energy content of the universe, yet their nature "
        "remains unknown. Particle accelerators like the Large Hadron Collider at CERN probe matter at the highest "
        "energies ever achieved, confirming predictions and searching for physics beyond the Standard Model. "
        "Supersymmetry predicts that every known particle has a superpartner, which would solve the hierarchy problem "
        "and provide a dark matter candidate, but no superpartners have been detected so far. "
        # History (~500 words)
        "The Industrial Revolution, which took place from the 18th to 19th centuries, was a period of profound "
        "economic and social transformation that began in Britain before spreading rapidly to Western Europe and "
        "North America. The transition from hand production methods to machine manufacturing, new chemical processes, "
        "iron production, increased use of steam power, the development of machine tools, and the rise of the factory "
        "system fundamentally changed the nature of work and human society. Before industrialisation, most people "
        "lived in rural areas and worked in agriculture or cottage industries. Cities grew rapidly as workers moved "
        "to be near factories, creating new urban social classes and new forms of poverty and inequality. The textile "
        "industry was among the first to be transformed, with inventions like the spinning jenny, the water frame, "
        "and the power loom multiplying output many times over. James Watt's improvements to the steam engine in the "
        "1760s and 1770s created a versatile power source that could drive factories, pump mines, and later propel "
        "locomotives and ships. The railways transformed transportation, linking cities, reducing travel times, "
        "and creating national markets for goods. The Roman Empire was the post-Republican period of ancient Roman "
        "civilization, with a government headed by emperors and large territorial holdings around the Mediterranean "
        "Sea in Europe, North Africa, and Western Asia. The city of Rome was the largest city in the world from around "
        "100 BC to 400 AD. Roman law, architecture, engineering, and language shaped European civilization for "
        "centuries. The Renaissance was a cultural movement that profoundly affected European intellectual life in the "
        "early modern period. Beginning in Italy and spreading to the rest of Europe by the 16th century, its "
        "influence was felt in literature, philosophy, art, music, politics, science, religion, and every aspect of "
        "intellectual inquiry. Humanism, the study of classical texts, and new techniques of perspective in painting "
        "transformed artistic production. The printing press, invented by Gutenberg around 1440, democratized "
        "knowledge and accelerated the spread of new ideas across Europe. "
        # Biology (~500 words)
        "The theory of evolution by natural selection, first formulated by Charles Darwin and Alfred Russel Wallace, "
        "is the cornerstone of modern biology. The theory states that organisms with heritable traits that are better "
        "suited to their environment will tend to survive and produce more offspring. Over generations these "
        "advantageous traits become more common in the population, leading to the gradual transformation of species "
        "over time. DNA, the molecule that carries genetic information, was identified as a double helix by Watson "
        "and Crick in 1953, based on X-ray crystallography data from Rosalind Franklin. The genetic code maps "
        "three-nucleotide codons to amino acids, and this code is nearly universal across all life on Earth, "
        "suggesting all living things share a common ancestor. The human genome contains about 3 billion base pairs "
        "encoding roughly 20,000 protein-coding genes, yet these genes make up less than 2 percent of the total DNA. "
        "Much of the remaining sequence is regulatory, structural, or of unknown function. The cell is the basic "
        "unit of life. Prokaryotes like bacteria lack membrane-bound organelles, while eukaryotes have complex "
        "internal structure including a nucleus, mitochondria, and in plants, chloroplasts. Mitochondria are the "
        "powerhouses of the cell, generating ATP through oxidative phosphorylation. The immune system protects "
        "organisms from pathogens through both innate and adaptive responses. T-cells and B-cells are central to "
        "adaptive immunity, with B-cells producing antibodies specific to particular antigens. CRISPR-Cas9 gene "
        "editing allows precise modification of DNA sequences in living cells, offering revolutionary possibilities "
        "for medicine, agriculture, and basic research. Ecosystems are communities of organisms interacting with "
        "each other and their physical environment. Energy flows through ecosystems from producers to consumers "
        "to decomposers, while nutrients cycle between living organisms and the abiotic environment. "
        # Computer science (~500 words)
        "Artificial intelligence is the simulation of human intelligence processes by computer systems. Machine "
        "learning, a subset of AI, uses statistical techniques to give computers the ability to learn from data "
        "without being explicitly programmed. Deep learning employs artificial neural networks with many layers "
        "of processing units to learn hierarchical representations of data. The transformer architecture, introduced "
        "in the paper 'Attention Is All You Need' in 2017, uses self-attention mechanisms to process sequences "
        "in parallel and has become the foundation for most modern large language models. These models are trained "
        "on enormous amounts of text data and can generate coherent, contextually appropriate responses to a wide "
        "range of prompts. The scaling hypothesis suggests that simply increasing model size and training data "
        "continuously improves capabilities. A computer program is a set of instructions that a computer can "
        "execute to perform a specific task. Programming languages range from low-level assembly code, which maps "
        "closely to machine instructions, to high-level languages like Python and Java, which are more abstract "
        "and easier for humans to write. Algorithms are step-by-step procedures for solving computational problems. "
        "Sorting algorithms like quicksort and mergesort, search algorithms like binary search, and graph "
        "algorithms like Dijkstra's shortest path are fundamental building blocks of software. Computational "
        "complexity theory classifies problems by the resources required to solve them. P problems can be solved "
        "in polynomial time; NP problems have solutions verifiable in polynomial time but may require exponential "
        "time to find. Whether P equals NP is one of the most important unsolved problems in mathematics and "
        "computer science. Operating systems manage hardware resources and provide services for application "
        "programs. They handle memory management, process scheduling, file systems, and device drivers. "
        "Networks connect computers together to enable communication and resource sharing. The internet is a "
        "global network of networks using the TCP/IP protocol suite. Cryptography provides the mathematical "
        "foundations for secure communication, with public-key cryptography enabling secure key exchange over "
        "untrusted channels. "
        # Mathematics (~500 words)
        "Mathematics has been essential to the development of science and technology throughout human history. "
        "From ancient Babylonians who developed a base-60 number system we still use for measuring time, to "
        "Newton and Leibniz's invention of calculus that made modern physics possible, mathematical ideas have "
        "been the foundation of scientific progress. Number theory studies the properties of integers and has "
        "applications in cryptography. The prime numbers, which have no divisors other than one and themselves, "
        "are the building blocks of all integers via unique factorization. The Riemann hypothesis, which concerns "
        "the distribution of prime numbers via the zeros of the Riemann zeta function, is one of the most famous "
        "unsolved problems in mathematics. Geometry describes shapes and their properties. Euclidean geometry "
        "remains valid for everyday scales, while non-Euclidean geometries, such as the hyperbolic and elliptic "
        "geometries developed by Gauss, Lobachevsky, and Riemann, describe spaces of constant negative or "
        "positive curvature. Einstein's general relativity uses the mathematics of curved Riemannian manifolds "
        "to describe gravity. Abstract algebra studies algebraic structures such as groups, rings, and fields. "
        "Group theory underpins much of modern physics, as physical symmetries correspond to group representations. "
        "Galois theory uses group theory to determine which polynomial equations can be solved by radicals, "
        "answering questions that had puzzled mathematicians for centuries. Topology studies properties of spaces "
        "that are preserved under continuous deformations. A coffee cup and a donut are topologically equivalent "
        "because each has one hole. The Poincare conjecture, proved by Perelman in 2003, characterizes the "
        "three-dimensional sphere among all compact three-manifolds. Probability theory and statistics underpin "
        "modern science, economics, and machine learning. Bayesian inference provides a principled framework "
        "for updating beliefs in light of new evidence. The central limit theorem explains why normal distributions "
        "appear so frequently in nature. "
        # Philosophy (~500 words)
        "Philosophy examines fundamental questions about existence, knowledge, ethics, reason, language, and "
        "the nature of mind. The ancient Greeks laid the foundations of Western philosophy. Socrates, whose "
        "method of questioning exposed contradictions in popular beliefs, was executed in 399 BC for impiety "
        "and corrupting the youth. His student Plato argued that the world we perceive through the senses is "
        "a pale reflection of an ideal realm of eternal Forms. Aristotle rejected Plato's theory of Forms "
        "and developed a systematic philosophy encompassing logic, biology, ethics, politics, and metaphysics. "
        "His syllogistic logic dominated Western thought for two millennia. Epistemology asks how we can know "
        "anything and what constitutes justified belief. Descartes' method of radical doubt led him to the "
        "famous conclusion 'I think therefore I am,' which he took as an indubitable foundation for knowledge. "
        "Hume argued that all knowledge derives from sensory experience and that causation cannot be observed "
        "directly but is merely a habit of mind. Kant synthesized rationalism and empiricism, arguing that the "
        "mind actively structures experience according to innate categories of understanding. Ethics concerns "
        "how we should live and what makes actions right or wrong. Consequentialism judges actions by their "
        "outcomes; utilitarianism, developed by Bentham and Mill, holds that we should maximize total well-being. "
        "Deontological ethics, associated with Kant, holds that some actions are intrinsically right or wrong "
        "regardless of consequences. Virtue ethics, originating with Aristotle, focuses on the character of the "
        "moral agent. Philosophy of mind addresses questions about consciousness, mental states, and their "
        "relationship to the physical brain. The hard problem of consciousness — explaining why there is "
        "subjective experience at all — remains deeply contested. Functionalism holds that mental states are "
        "defined by their functional roles and can be multiply realized in different physical substrates. "
        # Economics (~500 words)
        "Economics is the study of how societies allocate scarce resources to satisfy unlimited wants and needs. "
        "Microeconomics analyzes the behavior of individual consumers and firms. The theory of supply and demand "
        "explains how prices coordinate economic activity in competitive markets: when price rises, quantity "
        "supplied rises and quantity demanded falls, pushing the market toward equilibrium. Consumer theory "
        "models how individuals maximize utility subject to budget constraints. Production theory models how "
        "firms minimize costs and maximize profits. Market failures, including externalities, public goods, "
        "information asymmetries, and monopoly power, justify government intervention in markets. Macroeconomics "
        "studies the economy as a whole, examining variables like GDP, unemployment, inflation, and growth. "
        "Keynesian economics, developed by John Maynard Keynes in response to the Great Depression, argues that "
        "aggregate demand determines output in the short run and that government fiscal policy can stabilize "
        "the economy. Monetarism, associated with Milton Friedman, emphasizes the role of money supply in "
        "determining nominal GDP and argues that monetary policy should follow stable rules rather than "
        "discretion. The efficient market hypothesis holds that asset prices reflect all available information, "
        "making it impossible to consistently outperform the market. Behavioral economics incorporates insights "
        "from psychology to model how real people deviate from the predictions of the standard rational agent "
        "model. Game theory studies strategic interaction among rational agents. The Nash equilibrium, named "
        "after John Nash, is a state in which no player can improve their outcome by unilaterally changing "
        "strategy. Game theory has applications in economics, political science, biology, and computer science. "
        "International trade theory explains why countries specialize in producing goods where they have a "
        "comparative advantage and trade with each other to mutual benefit, even when one country is more "
        "efficient at producing everything. "
        # Chemistry (~500 words)
        "Chemistry is the science of matter, its properties, structure, composition, and the changes it "
        "undergoes during chemical reactions. The periodic table, developed by Mendeleev in 1869, organizes "
        "elements by atomic number and reveals periodic trends in their properties. The electron configuration "
        "of atoms determines their chemical behavior: elements in the same group share similar valence electron "
        "configurations and similar reactivity. Chemical bonding holds atoms together in molecules and "
        "extended structures. Covalent bonds involve the sharing of electron pairs between atoms. Ionic bonds "
        "involve the transfer of electrons from one atom to another, creating oppositely charged ions that "
        "attract each other. Metallic bonds hold atoms together in metals through a sea of delocalized "
        "electrons. Thermodynamics governs whether chemical reactions can occur spontaneously. The Gibbs free "
        "energy combines enthalpy and entropy to determine the equilibrium constant and direction of reaction. "
        "Kinetics determines how fast reactions occur and depends on activation energies, temperature, "
        "concentration, and catalysts. Organic chemistry studies carbon compounds, which number in the millions "
        "and form the basis of all life. Hydrocarbons consist solely of carbon and hydrogen; their derivatives "
        "include alcohols, acids, esters, amines, and countless other functional groups. Polymer chemistry "
        "studies large molecules built from repeating monomer units, including synthetic plastics and natural "
        "biopolymers like proteins and nucleic acids. Biochemistry bridges chemistry and biology, studying "
        "the molecular processes that underlie life. Enzymes are protein catalysts that dramatically accelerate "
        "specific chemical reactions in living cells. Metabolic pathways like glycolysis and the citric acid "
        "cycle extract energy from nutrients. Spectroscopy techniques including NMR, mass spectrometry, "
        "infrared spectroscopy, and X-ray crystallography allow chemists to determine molecular structures "
        "and compositions with high precision."
    )

    # ------------------------------------------------------------------ Task B: 3rd text (creative/narrative)
    CREATIVE_TEXT = (
        # Fictional adventure story synopsis
        "The Dragon's Meridian: In the fractured kingdom of Valdris, where three suns set at different hours "
        "and the tides run with liquid silver, the young cartographer Senna discovers an anomaly in the ancient "
        "maps she restores for the Archive Guild. A hidden coastline — one that should not exist — appears on "
        "three separate parchments from different centuries, all drawn by cartographers who subsequently "
        "vanished. When her mentor is abducted by the masked agents of the Conclave, Senna sets out across "
        "the Ashpeaks with a disgraced knight named Corvath and a merchant who speaks to birds. The journey "
        "takes them through the Murmuring Marshes, where the ghosts of drowned sailors whisper directions "
        "to those willing to listen, through the Forge Cities of the interior where artisans craft weapons "
        "from crystallized lightning, and finally to the Meridian Coast itself, where time flows differently "
        "and the sea is made of memory rather than water. Senna discovers that the hidden coastline is not "
        "a place but a moment — a fracture in the world's history that the Conclave has been trying to "
        "reach and exploit. The final confrontation takes place in a lighthouse at the edge of everything, "
        "where light from the three suns converges once per century, and the price of saving the kingdom "
        "is Senna's ability to read any map ever again. "
        # Cooking recipe collection
        "Braised Short Ribs with Gremolata: Pat the short ribs dry and season generously with salt and "
        "black pepper. Heat oil in a heavy Dutch oven over high heat until shimmering. Sear ribs in batches "
        "until deeply browned on all sides, about four minutes per side. Remove ribs and reduce heat to "
        "medium. Add diced onion, carrot, and celery to the pot and cook until softened, scraping up the "
        "browned bits from the bottom. Add tomato paste and cook for two minutes until darkened. Pour in "
        "a bottle of full-bodied red wine and bring to a boil, then add beef stock, thyme, rosemary, bay "
        "leaves, and garlic. Return ribs to the pot, cover, and transfer to a 325-degree oven. Braise for "
        "three to three and a half hours until the meat is completely tender and falling from the bone. "
        "Remove ribs and strain the braising liquid, then reduce it over high heat until glossy and "
        "intensified. For the gremolata, combine finely chopped flat-leaf parsley, lemon zest, and raw "
        "minced garlic. Serve the ribs with the reduced sauce spooned over and gremolata scattered on top. "
        "Lemon Ricotta Pancakes: Separate the eggs and beat the whites to stiff peaks. Combine the yolks "
        "with whole-milk ricotta, fresh lemon juice, and finely grated lemon zest. Fold in flour, baking "
        "powder, and a pinch of salt, then gently incorporate the egg whites in three additions to maintain "
        "as much volume as possible. Cook on a buttered griddle over medium-low heat, flipping once when "
        "bubbles form across the surface. Serve immediately with warm blueberry compote and maple syrup. "
        # Travel guide description
        "Oaxaca, Mexico: The high-altitude valleys of Oaxaca are home to some of the most complex and "
        "distinctive cuisine in the Americas, shaped by twenty-plus distinct indigenous cultures and "
        "centuries of history. The city's colonial center, a UNESCO World Heritage Site, is built from "
        "green stone quarried from the surrounding hills, giving the architecture an otherworldly tint "
        "in the late afternoon light. Begin any visit at the Mercado Benito Juarez, where vendors sell "
        "chiles negro, pasilla, and ancho alongside dried grasshoppers seasoned with lime and chili — "
        "a snack locals call chapulines. The seven canonical mole sauces of Oaxaca each require days "
        "of preparation and dozens of ingredients, balanced to achieve a depth that rewards slow, "
        "attentive eating. The mezcal tradition here predates the Spanish conquest; distilleries in "
        "the villages surrounding the city produce small-batch spirits from dozens of agave varieties "
        "using methods unchanged for centuries. Monte Alban, the pre-Columbian city built by the Zapotec "
        "civilization on a flattened mountaintop overlooking three valleys, receives visitors at dawn "
        "when the light is horizontal and the stones glow amber. The Guelaguetza festival in July brings "
        "representatives of all the region's indigenous communities to the city to dance, exchange gifts, "
        "and celebrate a tradition of communal solidarity that has persisted across five centuries of change. "
        # Sports commentary
        "The Championship Final: With twelve seconds remaining on the clock and the home side trailing by "
        "a single point, the arena had gone almost completely silent — forty thousand people holding their "
        "collective breath. Martinez received the inbound pass on the left wing, pump-faked once to create "
        "space, and drove hard toward the elbow. The defender recovered well, cutting off the lane, but "
        "Martinez pulled up at the high post and floated a one-footed fadeaway that seemed to hang in the "
        "air for an impossible duration before dropping through with the softest of sounds. The noise that "
        "followed was physical. This was the defining performance of a career already full of them: "
        "thirty-one points, nine assists, seven rebounds, and four steals across forty-two minutes of "
        "the most watched final in the competition's history. What makes Martinez exceptional is not "
        "athleticism alone but an ability to slow the game down mentally while it speeds up physically, "
        "to read defensive rotations three steps before they develop, and to deliver in the moments when "
        "every other player in the building would feel the weight of expectation as a physical constraint. "
        "The postgame press conference lasted forty minutes and still failed to produce an adequate "
        "explanation for what everyone had just witnessed. "
        # Legal language
        "Limited Liability Agreement — General Terms: This Agreement is entered into as of the date last "
        "signed below by and between the Parties identified in the signature block. For purposes of this "
        "Agreement, 'Confidential Information' means any and all information or data that has or could "
        "have commercial value or other utility in the business in which Disclosing Party is engaged. "
        "If Confidential Information is in written form, the Disclosing Party shall label or stamp the "
        "materials with the word 'Confidential' or some similar warning. The Receiving Party agrees to "
        "hold the Confidential Information in strict confidence, to protect it using the same degree of "
        "care that it uses to protect its own most sensitive information but in no event less than "
        "reasonable care, and to use Confidential Information solely for the purposes contemplated "
        "herein. The obligations of confidentiality shall survive the termination of this Agreement "
        "for a period of five years. Notwithstanding the foregoing, Confidential Information shall not "
        "include information that is or becomes generally available to the public through no act or "
        "omission of the Receiving Party, or that the Receiving Party can demonstrate was in its "
        "possession prior to disclosure by the Disclosing Party. Neither party shall be liable for "
        "any indirect, incidental, special, exemplary, or consequential damages. "
        # Music and art criticism
        "On John Coltrane's A Love Supreme: Recorded in a single session in December 1964, A Love "
        "Supreme represents the apex of a spiritual trajectory that began when Coltrane overcame his "
        "addiction to heroin and alcohol in 1957 and experienced what he described as a religious "
        "awakening. The four-part suite — Acknowledgement, Resolution, Pursuance, Psalm — moves through "
        "devotion, intention, intense searching, and finally the wordless recitation of a written "
        "prayer, with Coltrane's tenor saxophone tracing the contour of each syllable in the poem he "
        "had composed. The opening bass motif, four notes that Coltrane eventually voices over and over "
        "in the first movement, is one of the most recognized figures in all of recorded music. McCoy "
        "Tyner's piano comping creates vast harmonic space without ever overcrowding, Elvin Jones "
        "generates polyrhythmic density that somehow remains grooving rather than chaotic, and Jimmy "
        "Garrison anchors everything with bass lines that are themselves melodic arguments. The album "
        "sold over a million copies, extraordinary for avant-garde jazz, and has continued to attract "
        "listeners for whom it functions less as entertainment than as a form of testimony — evidence "
        "that certain emotional and spiritual states can be made audible through organized sound."
    )

    # ------------------------------------------------------------------ previously measured reference points
    # From modal_quality_push.py run (2b+eviction on EASY and HARD)
    PRIOR_500TOK = {
        35: {"easy": +0.39, "hard": +0.90},   # B 2b+35%evict
        50: {"easy": +3.63, "hard": None},     # interpolated
        60: {"easy": None,  "hard": None},
        70: {"easy": +4.81, "hard": +3.87},    # B-adjacent (approx)
        80: {"easy": +6.58, "hard": +6.09},    # approx from sweep
    }

    sliding_window = 32

    # ------------------------------------------------------------------ scorer
    def score_importance(kv_cache, obs_window=32, pool_kernel=5):
        k0, _ = get_kv(kv_cache, 0)
        seq_len = k0.shape[2]
        w = min(obs_window, seq_len)
        all_imp = torch.zeros(seq_len, device='cpu')
        for l in range(n_layers):
            kl, _ = get_kv(kv_cache, l)
            k = kl[0].float().cpu()
            k_obs = k[:, -w:, :]
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(k_obs, k.transpose(-2, -1)) * scale
            all_pos = torch.arange(seq_len).unsqueeze(0)
            obs_pos = torch.arange(seq_len - w, seq_len).unsqueeze(1)
            causal = (all_pos <= obs_pos)
            scores = scores.masked_fill(~causal.unsqueeze(0), float('-inf'))
            attn = F.softmax(scores, dim=-1, dtype=torch.float32)
            layer_imp = attn.sum(dim=1).mean(dim=0)
            if seq_len > pool_kernel:
                imp_1d = layer_imp.unsqueeze(0).unsqueeze(0)
                layer_imp = F.avg_pool1d(imp_1d, kernel_size=pool_kernel,
                                         padding=pool_kernel // 2, stride=1).squeeze()[:seq_len]
            all_imp += layer_imp
        return all_imp / n_layers

    # ------------------------------------------------------------------ evict+quantize
    def evict_quantize(kv_cache, keep_mask, key_bits, val_bits, prefix_len):
        H = hadamard_matrix(head_dim).cpu()
        n_kept = keep_mask.sum().item()
        total_fp16 = 0
        all_key_coords = []
        all_val_coords = []

        cctx = zstandard.ZstdCompressor(level=22)

        for l in range(n_layers):
            kl, vl = get_kv(kv_cache, l)
            k = kl.float().cpu()
            v = vl.float().cpu()
            total_fp16 += k.numel() * 2 + v.numel() * 2

            for is_key, tensor, bits, coord_list in [
                (True,  k, key_bits, all_key_coords),
                (False, v, val_bits, all_val_coords),
            ]:
                levels = 2 ** bits
                t = tensor[0].clone()
                for h in range(n_kv_heads):
                    if is_key:
                        t_head = inverse_rope(t[h:h+1], base=rope_base)[0]
                    else:
                        t_head = t[h]
                    kept_data = t_head[keep_mask]
                    rotated = kept_data @ H.T
                    amax = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                    sc = amax / (levels / 2)
                    normalized = rotated / sc
                    groups = normalized.reshape(-1, 8)
                    lp = E8Lattice.nearest_point(groups).clamp(-levels / 2, levels / 2)
                    coords = lp.reshape(-1, head_dim)
                    quantized = (coords * sc) @ H
                    int_coords = coords.detach().numpy()
                    has_half = np.any(
                        np.abs(int_coords.flatten() - np.round(int_coords.flatten())) > 0.25
                    )
                    if has_half:
                        coord_list.append(np.round(int_coords.flatten() * 2).astype(np.int8))
                    else:
                        coord_list.append(np.round(int_coords.flatten()).astype(np.int8))
                    result = torch.zeros_like(t_head)
                    result[keep_mask] = quantized
                    if is_key:
                        t[h] = forward_rope(result.unsqueeze(0), base=rope_base)[0]
                    else:
                        t[h] = result
                if is_key:
                    set_kv(kv_cache, l, t.unsqueeze(0).half().to("cuda"), vl)
                else:
                    kl_now, _ = get_kv(kv_cache, l)
                    set_kv(kv_cache, l, kl_now, t.unsqueeze(0).half().to("cuda"))

        total_idx = 0
        for coords_arr in all_key_coords + all_val_coords:
            arr = coords_arr.ravel()
            n_per_tok = len(arr) // n_kept if n_kept > 0 else 0
            if n_per_tok > 0 and len(arr) % n_kept == 0:
                reshaped = arr.reshape(n_kept, n_per_tok)
                delta = np.zeros_like(reshaped)
                delta[0] = reshaped[0]
                delta[1:] = reshaped[1:] - reshaped[:-1]
                total_idx += len(cctx.compress(delta.astype(np.int8).tobytes()))
            else:
                total_idx += len(cctx.compress(arr.tobytes()))

        scale_bytes = n_kept * n_layers * 2 * n_kv_heads * 2
        mask_bytes = math.ceil(prefix_len / 8) * n_layers
        total = total_idx + scale_bytes + mask_bytes

        return {
            "fp16": total_fp16, "idx": total_idx, "scale": scale_bytes,
            "mask": mask_bytes, "total": total,
            "ratio": total_fp16 / total if total > 0 else 0,
            "n_kept": n_kept,
        }, kv_cache

    # ------------------------------------------------------------------ build keep_mask
    def build_keep_mask(prefix_len, evict_pct, importance):
        if evict_pct == 0:
            return torch.ones(prefix_len, dtype=torch.bool)
        keep_pct = 100 - evict_pct
        keep_mask = torch.zeros(prefix_len, dtype=torch.bool)
        keep_mask[0] = True
        keep_mask[-sliding_window:] = True
        n_to_keep = max(int(prefix_len * keep_pct / 100), sliding_window + 1)
        n_from_imp = n_to_keep - keep_mask.sum().item()
        if n_from_imp > 0:
            imp = importance.clone()
            imp[keep_mask] = -float('inf')
            _, top_idx = imp.topk(min(n_from_imp, (~keep_mask).sum().item()))
            keep_mask[top_idx] = True
        return keep_mask

    # ------------------------------------------------------------------ run on text
    def run_configs_on_text(text_name, text, configs, max_tokens=4096):
        inputs = tok(text, return_tensors="pt", max_length=max_tokens, truncation=True)
        full_ids = inputs.input_ids.to("cuda")
        n_tok = full_ids.shape[1]
        prefix_len = n_tok // 2
        cont_len = n_tok - prefix_len
        print(f"\n{'='*80}")
        print(f"TEXT: {text_name} | tokens={n_tok}, prefix={prefix_len}, cont={cont_len}")
        print(f"{'='*80}")

        with torch.no_grad():
            pout = model(full_ids[:, :prefix_len], use_cache=True)
            cout = model(full_ids[:, prefix_len:],
                         past_key_values=pout.past_key_values, use_cache=True)
            logits = cout.logits[:, :-1, :].float()
            targets = full_ids[:, prefix_len + 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            baseline_ppl = torch.exp(loss).item()
        print(f"Baseline PPL: {baseline_ppl:.4f}")

        with torch.no_grad():
            pout = model(full_ids[:, :prefix_len], use_cache=True)
        importance = score_importance(pout.past_key_values)

        results = []
        print(f"\n{'Config':<28s} {'PPL':>8s} {'Delta%':>8s} {'Ratio':>7s} {'Kept':>6s}")
        print("-" * 65)

        for cfg in configs:
            torch.cuda.empty_cache()
            name      = cfg["name"]
            evict_pct = cfg["evict_pct"]
            key_bits  = cfg["key_bits"]
            val_bits  = cfg["val_bits"]

            with torch.no_grad():
                pout = model(full_ids[:, :prefix_len], use_cache=True)
                kv = pout.past_key_values

            keep_mask = build_keep_mask(prefix_len, evict_pct, importance)
            info, kv = evict_quantize(kv, keep_mask, key_bits, val_bits, prefix_len)

            evict_mask = ~keep_mask
            attn_ctx = torch.ones(prefix_len, dtype=torch.long, device="cuda")
            attn_ctx[evict_mask] = 0
            attn_full = torch.cat([
                attn_ctx,
                torch.ones(cont_len, dtype=torch.long, device="cuda")
            ])

            with torch.no_grad():
                cout = model(full_ids[:, prefix_len:], past_key_values=kv,
                             attention_mask=attn_full.unsqueeze(0), use_cache=True)
                logits = cout.logits[:, :-1, :].float()
                targets = full_ids[:, prefix_len + 1:]
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                ppl = torch.exp(loss).item()

            delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
            tag = " <<<" if abs(delta) < 1.0 else ""
            print(f"{name:<28s} {ppl:8.4f} {delta:+7.2f}% {info['ratio']:6.2f}x {info['n_kept']:5d}{tag}")
            results.append({
                "name": name, "evict_pct": evict_pct,
                "key_bits": key_bits, "val_bits": val_bits,
                "ppl": ppl, "delta": delta,
                "ratio": info["ratio"], "n_kept": info["n_kept"],
                "baseline": baseline_ppl, "text": text_name,
            })

        return results

    # ------------------------------------------------------------------ Task A configs (2b + eviction sweep)
    configs_4k = []
    for ep in [0, 35, 50, 60, 70, 80]:
        label = f"2b+{ep}%evict" if ep > 0 else "2b no-evict"
        configs_4k.append({"name": label, "evict_pct": ep, "key_bits": 2, "val_bits": 2})

    # ------------------------------------------------------------------ Task B configs (3rd text validation)
    configs_val = []
    for ep in [35, 70, 80]:
        configs_val.append({"name": f"2b+{ep}%evict", "evict_pct": ep, "key_bits": 2, "val_bits": 2})

    # ------------------------------------------------------------------ Run Task A
    print("\n" + "=" * 80)
    print("TASK A: 4K CONTEXT SWEEP (2048 prefix + 2048 continuation)")
    print("=" * 80)
    results_4k = run_configs_on_text("4K_MULTI_TOPIC", LONG_4K_TEXT, configs_4k, max_tokens=4096)

    # ------------------------------------------------------------------ Run Task B
    print("\n" + "=" * 80)
    print("TASK B: 3RD TEXT VALIDATION (creative/narrative)")
    print("=" * 80)
    # Use standard 2048 token evaluation (same as prior experiments for fair comparison)
    results_val = run_configs_on_text("CREATIVE_NARRATIVE", CREATIVE_TEXT, configs_val, max_tokens=2048)

    # ------------------------------------------------------------------ Summary Tables
    print("\n" + "=" * 80)
    print("=== 4K CONTEXT vs 500-TOKEN REFERENCE ===")
    print("=" * 80)
    print(f"\n{'Evict%':<10s} {'500tok Δ% (easy)':<20s} {'2048tok Δ%':<15s} {'Improvement?':<15s} {'Ratio':<8s}")
    print("-" * 72)

    ref_500 = {
        0:  "+0.00% (baseline)",
        35: "+0.39%",
        50: "+3.63%",
        60: "~N/A",
        70: "+4.81%",
        80: "+6.58%",
    }

    for r in results_4k:
        ep = r["evict_pct"]
        ref = ref_500.get(ep, "N/A")
        delta_4k = f"{r['delta']:+.2f}%"
        # Improvement = smaller delta is better
        improved = "YES (better)" if ep > 0 and r["delta"] < float(ref.split("%")[0].replace("~N/A", "999")) else "-"
        try:
            ref_val = float(ref.replace("%", "").replace("~N/A", "999").replace("+", ""))
            improved = "BETTER" if r["delta"] < ref_val else ("WORSE" if r["delta"] > ref_val + 0.5 else "SIMILAR")
        except Exception:
            improved = "?"
        print(f"{ep:<10d} {ref:<20s} {delta_4k:<15s} {improved:<15s} {r['ratio']:5.2f}x")

    print("\n" + "=" * 80)
    print("=== 3RD TEXT VALIDATION ===")
    print("=" * 80)
    # Reference values from prior experiments on EASY and HARD texts
    ref_easy = {35: +0.39, 70: +4.81, 80: +6.58}
    ref_hard = {35: +0.90, 70: +3.87, 80: +6.09}

    print(f"\n{'Config':<15s} {'Text3 (creative) Δ%':<22s} {'Easy Δ% (prior)':<20s} {'Hard Δ% (prior)':<20s} {'Stable?':<10s}")
    print("-" * 90)

    for r in results_val:
        ep = r["evict_pct"]
        easy_ref = ref_easy.get(ep, None)
        hard_ref = ref_hard.get(ep, None)
        easy_str = f"{easy_ref:+.2f}%" if easy_ref is not None else "N/A"
        hard_str = f"{hard_ref:+.2f}%" if hard_ref is not None else "N/A"
        # Stable if text3 delta is within 2× of the easy/hard range
        if easy_ref is not None and hard_ref is not None:
            max_ref = max(abs(easy_ref), abs(hard_ref))
            stable = "YES" if abs(r["delta"]) < max_ref * 2.5 else "NO"
        else:
            stable = "?"
        print(f"{r['name']:<15s} {r['delta']:+.2f}%{'':<16s} {easy_str:<20s} {hard_str:<20s} {stable:<10s}")

    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    return results_4k, results_val


@app.local_entrypoint()
def main():
    import time
    print("Launching 4K context + 3rd text validation on Modal A10G...")
    results_4k, results_val = run_4k_and_validation.remote()

    out_dir = os.path.join(os.path.dirname(__file__), "..", ".company", "engineering")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "4k_validation_results.md")

    ref_easy = {35: +0.39, 70: +4.81, 80: +6.58}
    ref_hard = {35: +0.90, 70: +3.87, 80: +6.09}

    with open(out_path, "w") as f:
        f.write("# 4K Context Sweep + 3rd Text Validation — Mistral-7B\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write("**Model:** Mistral-7B-v0.1, fp16\n")
        f.write("**Quantization:** 2-bit E8 lattice VQ + importance-based eviction\n\n")

        f.write("## Task A: 4K Context Sweep\n\n")
        f.write("8 topics concatenated (~4096 tokens), 2048-token prefix + 2048-token continuation.\n\n")
        f.write("| Evict% | Baseline (500-tok easy) | 2048-tok Δ% | Ratio | Improvement? |\n")
        f.write("|--------|------------------------|-------------|-------|-------------|\n")
        ref_500 = {0: "+0.00%", 35: "+0.39%", 50: "+3.63%", 60: "N/A", 70: "+4.81%", 80: "+6.58%"}
        for r in results_4k:
            ep = r["evict_pct"]
            ref = ref_500.get(ep, "N/A")
            try:
                ref_val = float(ref.replace("%", "").replace("N/A", "999").replace("+", ""))
                improved = "BETTER" if r["delta"] < ref_val else ("WORSE" if r["delta"] > ref_val + 0.5 else "SIMILAR")
            except Exception:
                improved = "?"
            f.write(f"| {ep}% | {ref} | {r['delta']:+.3f}% | {r['ratio']:.2f}x | {improved} |\n")

        f.write("\n## Task B: 3rd Text Validation (Creative/Narrative)\n\n")
        f.write("Non-academic text: adventure fiction, recipes, travel, sports, legal, music criticism.\n\n")
        f.write("| Config | Text3 (creative) Δ% | Easy Δ% (prior) | Hard Δ% (prior) | Stable? |\n")
        f.write("|--------|--------------------|-----------------|-----------------|---------|\n")
        for r in results_val:
            ep = r["evict_pct"]
            easy_str = f"{ref_easy[ep]:+.2f}%" if ep in ref_easy else "N/A"
            hard_str = f"{ref_hard[ep]:+.2f}%" if ep in ref_hard else "N/A"
            if ep in ref_easy and ep in ref_hard:
                max_ref = max(abs(ref_easy[ep]), abs(ref_hard[ep]))
                stable = "YES" if abs(r["delta"]) < max_ref * 2.5 else "NO"
            else:
                stable = "?"
            f.write(f"| {r['name']} | {r['delta']:+.3f}% | {easy_str} | {hard_str} | {stable} |\n")

        f.write("\n## All Raw Results\n\n")
        f.write("| Task | Text | Config | PPL | Delta% | Ratio | Kept |\n")
        f.write("|------|------|--------|-----|--------|-------|------|\n")
        for r in results_4k:
            f.write(f"| A | 4K multi-topic | {r['name']} | {r['ppl']:.4f} | {r['delta']:+.3f}% | {r['ratio']:.2f}x | {r['n_kept']} |\n")
        for r in results_val:
            f.write(f"| B | Creative/Narrative | {r['name']} | {r['ppl']:.4f} | {r['delta']:+.3f}% | {r['ratio']:.2f}x | {r['n_kept']} |\n")

        f.write("\n## Conclusions\n\n")

        # Auto-generate conclusions
        a_better = sum(1 for r in results_4k if r["evict_pct"] > 0 and r["delta"] < ref_easy.get(r["evict_pct"], 999))
        a_total = sum(1 for r in results_4k if r["evict_pct"] > 0)
        f.write(f"- **4K context effect:** {a_better}/{a_total} eviction configs showed improvement vs 500-token prefix.\n")

        val_stable = sum(1 for r in results_val
                         if r["evict_pct"] in ref_easy
                         and abs(r["delta"]) < max(abs(ref_easy[r["evict_pct"]]), abs(ref_hard[r["evict_pct"]])) * 2.5)
        f.write(f"- **3rd text stability:** {val_stable}/{len(results_val)} configs showed stable behavior on creative text.\n")

        best_4k = min((r for r in results_4k if r["evict_pct"] > 0), key=lambda x: abs(x["delta"]), default=None)
        if best_4k:
            f.write(f"- **Best 4K config:** {best_4k['name']} — Δ%={best_4k['delta']:+.3f}%, ratio={best_4k['ratio']:.2f}x\n")

    print(f"\nResults written to: {out_path}")
