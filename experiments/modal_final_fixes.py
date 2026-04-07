"""Combined experiment: Fixed scorer 8K rerun + LongBench-style QA eval on Mistral-7B.

Task A: Fixed Scorer 8K Rerun
  - Bug fix: obs_window was hardcoded to 32 (at 2924 prefix = only 1% context visible)
  - Fix: obs_window = max(32, prefix_len // 16)  → at 2924 prefix gives obs_window=182
  - Same 15-paragraph 8K text as modal_llama3_8k.py
  - Configs: 2b + 0/35/50/60/70/80% eviction
  - Comparison vs old broken scorer results

Task B: LongBench-Style QA Eval
  - 5 diverse QA pairs with ~500-1000 token passages
  - Test types: factual recall, reasoning, detail extraction, comparison, numerical
  - Each evaluated with: baseline KV, 35% eviction, 60% eviction
  - Score: MATCH / PARTIAL / WRONG

Results written to .company/engineering/final_fixes_results.md
"""
import modal
import os

app = modal.App("nexusquant-final-fixes")

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

HF_SECRET = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

# ======================================================================
# SHARED TEXT CORPUS — same 15-paragraph text as modal_llama3_8k.py
# ======================================================================
MULTI_TOPIC_TEXT = (
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
    "in the paper Attention Is All You Need in 2017, uses self-attention mechanisms to process sequences "
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
    "famous conclusion I think therefore I am, which he took as an indubitable foundation for knowledge. "
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
    "and compositions with high precision. "
    # Geology (~350 words)
    "Geology is the science that studies the solid Earth, the rocks of which it is composed, and the "
    "processes by which they change over time. The Earth is composed of several layers: the inner core "
    "of solid iron and nickel, the outer core of liquid iron and nickel, the mantle of semi-solid rock, "
    "and the thin outer crust on which we live. The theory of plate tectonics, developed in the mid-20th "
    "century, revolutionized geology by explaining the movement of continents, the formation of mountain "
    "ranges, the occurrence of earthquakes and volcanoes, and the creation of ocean basins. The Earth's "
    "lithosphere is divided into tectonic plates that float on the more fluid asthenosphere below, driven "
    "by convection currents in the mantle. Where plates collide, one may be subducted beneath the other, "
    "creating deep trenches and volcanic arcs. Where plates spread apart, new ocean floor is created by "
    "volcanic activity at mid-ocean ridges. The rock cycle describes how igneous rocks formed from cooling "
    "magma are weathered into sediments, which are compacted into sedimentary rocks, which may be "
    "metamorphosed by heat and pressure, and eventually melted and recycled back into magma. The geologic "
    "time scale divides Earth's 4.5-billion-year history into eons, eras, periods, and epochs, defined "
    "by major events in the fossil record. The Cambrian explosion, around 541 million years ago, saw a "
    "rapid diversification of complex multicellular life. The five major mass extinctions have each "
    "eliminated a large fraction of species, reshaping the trajectory of evolution. Radiometric dating "
    "uses the known decay rates of radioactive isotopes to determine the ages of rocks and minerals with "
    "high precision. Glaciers and ice sheets have shaped much of the Northern Hemisphere's landscape "
    "through repeated advances and retreats during ice ages, carving valleys, depositing moraines, and "
    "leaving behind distinctive landforms visible today. "
    # Psychology (~350 words)
    "Psychology is the scientific study of mind and behavior, examining processes including perception, "
    "cognition, emotion, personality, social interactions, and mental health. Wilhelm Wundt established "
    "the first experimental psychology laboratory in Leipzig in 1879, marking the beginning of psychology "
    "as a formal discipline separate from philosophy. Sigmund Freud developed psychoanalysis, emphasizing "
    "the role of unconscious processes, early childhood experiences, and repressed conflicts in shaping "
    "adult personality and mental illness. Though many of Freud's specific claims have not been supported "
    "by empirical research, his emphasis on unconscious processes and the importance of early experience "
    "has been influential. Behaviorism, pioneered by Watson and Skinner, focused exclusively on observable "
    "behavior and the environmental contingencies that shape it through classical and operant conditioning. "
    "Cognitive psychology, which emerged in the 1950s and 1960s, brought mental processes back into focus, "
    "studying memory, attention, language, problem-solving, and decision-making. The information-processing "
    "model treats the mind as analogous to a computer, encoding, storing, and retrieving information. "
    "Social psychology studies how individuals think, feel, and behave in social contexts, examining "
    "phenomena like conformity, obedience, attitude change, group dynamics, and social identity. Milgram's "
    "obedience experiments and Zimbardo's Stanford Prison Experiment demonstrated disturbing aspects of "
    "human behavior under social pressure. Cognitive neuroscience bridges psychology and neuroscience, "
    "using brain imaging techniques to study the neural correlates of mental processes. Positive psychology, "
    "developed by Seligman, focuses on human flourishing, well-being, character strengths, and resilience "
    "rather than exclusively on mental illness. Clinical psychology applies psychological knowledge to "
    "assessment and treatment of mental disorders, using therapies ranging from cognitive-behavioral "
    "therapy to mindfulness-based interventions. Developmental psychology examines how cognition, emotion, "
    "and social relationships change across the lifespan from infancy through old age. "
    # Linguistics (~350 words)
    "Linguistics is the scientific study of language, encompassing its structure, variation, acquisition, "
    "and use. Human language is unique among animal communication systems in its productivity — the ability "
    "to generate and understand an unlimited number of novel sentences from a finite set of words and rules. "
    "Phonology studies the sound systems of languages; phonemes are the minimal units of sound that "
    "distinguish meaning. Morphology studies word structure and the rules for forming words from roots, "
    "prefixes, and suffixes. Syntax studies how words combine into sentences, with rules that vary across "
    "languages but also show striking universal tendencies. Semantics studies meaning, examining how words "
    "and sentences are interpreted. Pragmatics studies how context influences interpretation, including "
    "implicature, speech acts, and the cooperative principles governing conversation. Historical linguistics "
    "traces how languages change over time and reconstructs ancestral proto-languages from patterns of "
    "shared vocabulary and grammar. The Indo-European language family, which includes most European "
    "languages plus Persian, Hindi, and many others, is descended from a proto-language spoken around "
    "6000 years ago. Sociolinguistics studies how language varies across social groups, regions, and "
    "contexts. All natural languages have dialects, and variation in pronunciation, vocabulary, and "
    "grammar is normal and systematic rather than evidence of degraded or incorrect language. Language "
    "acquisition proceeds through a universal sequence in children, with babbling, first words, two-word "
    "combinations, and rapid vocabulary growth following similar timelines across cultures. Chomsky's "
    "generative grammar proposed that humans possess an innate universal grammar that constrains the "
    "possible structures of natural languages. Computational linguistics applies algorithms and statistical "
    "models to process and generate natural language, underlying technologies like machine translation, "
    "speech recognition, and large language models. "
    # Music Theory (~350 words)
    "Music is the art of organizing sound and silence in time to create structures that communicate "
    "emotional, aesthetic, or intellectual meaning. The physics of sound involves vibrations creating "
    "pressure waves that are perceived by the ear; pitch corresponds to frequency, loudness to amplitude, "
    "and timbre to the complex overtone spectrum. The twelve-tone equal temperament system divides the "
    "octave into twelve equal semitones, enabling instruments to play in any key with equal facility. "
    "Before equal temperament, other tuning systems were used, each with characteristic pure intervals "
    "and characteristic out-of-tune intervals. Western tonal music is organized around scales and keys, "
    "with the major and minor scales providing the vocabulary for harmony and melody. Chords are built "
    "by stacking thirds above a root note; the major, minor, diminished, and augmented triads are the "
    "basic harmonic units. Functional harmony describes how chords progress toward and away from the "
    "tonic, creating tension and resolution. The dominant seventh chord creates strong tension that "
    "resolves to the tonic, and this motion is fundamental to the tonal system. Counterpoint is the art "
    "of combining independent melodic lines according to rules governing their interaction. Bach's fugues "
    "are the supreme examples of contrapuntal technique, building complex structures from a single theme "
    "through inversion, augmentation, diminution, and stretto. Rhythm and meter organize music in time, "
    "with beats grouped into measures and patterns of strong and weak beats creating metrical feel. "
    "Syncopation displaces accents to weak beats, creating rhythmic complexity and propulsive energy. "
    "Twentieth-century music challenged tonal conventions through atonality, serialism, indeterminacy, "
    "minimalism, and electronic means. Schoenberg's twelve-tone method organized all twelve chromatic "
    "pitches into a row, subjecting it to systematic transformations. The global proliferation of "
    "recorded music in the twentieth century created new hybrid forms combining elements from diverse "
    "traditions. "
    # Architecture (~350 words)
    "Architecture is the art and science of designing and constructing buildings and other physical "
    "structures. It must satisfy practical functional requirements while also achieving aesthetic and "
    "cultural meaning. Structural engineering determines how buildings can stand against the forces of "
    "gravity, wind, and seismic activity. The arch, vault, and dome allowed ancient builders to span "
    "large spaces and create interior volumes of great height. The pointed arch and flying buttress of "
    "Gothic architecture enabled walls to be pierced with large windows, flooding interiors with light. "
    "Greek temple architecture developed a refined vocabulary of column orders — Doric, Ionic, and "
    "Corinthian — that has influenced Western building ever since. Roman engineers adapted Greek forms "
    "and combined them with the arch and concrete to create buildings of unprecedented size and "
    "complexity: the Pantheon's unreinforced concrete dome still stands after two thousand years. "
    "The Renaissance revived classical forms and developed a theoretical literature on the principles "
    "of architecture, with Palladio's Four Books of Architecture remaining influential centuries later. "
    "The Industrial Revolution introduced iron and later steel as structural materials, enabling "
    "skyscrapers and long-span structures like railway stations and exhibition halls. The modern movement "
    "in architecture rejected historical ornament in favor of honest expression of materials and "
    "structure, with form following function. Le Corbusier, Mies van der Rohe, and Frank Lloyd Wright "
    "were the dominant figures of 20th-century modernism. Postmodernism reintroduced historical "
    "reference and symbolic content, rejecting the austerity of the international style. Sustainable "
    "architecture addresses the environmental impact of buildings, which account for a large fraction "
    "of global energy use and carbon emissions. Passive design strategies, green roofs, solar panels, "
    "and natural ventilation can dramatically reduce a building's environmental footprint. "
    # Medicine (~350 words)
    "Medicine is the science and practice of diagnosing, treating, and preventing disease. The history "
    "of medicine stretches from ancient herbal remedies and surgical practices to modern molecular "
    "biology and genomics. The germ theory of disease, established in the 19th century through the "
    "work of Pasteur and Koch, demonstrated that specific microorganisms cause specific infectious "
    "diseases, replacing miasma theory and enabling targeted interventions. Vaccination, pioneered by "
    "Jenner's observation that milkmaids infected with cowpox were protected from smallpox, has "
    "eliminated or dramatically reduced many infectious diseases. Antibiotics, beginning with Fleming's "
    "discovery of penicillin in 1928, transformed the treatment of bacterial infections but are now "
    "threatened by the evolution of antibiotic resistance. The discovery of the double helix structure "
    "of DNA opened the molecular era of medicine, enabling understanding of genetic diseases and "
    "the development of genetic therapies. The human genome project, completed in 2003, provided "
    "a reference sequence for the entire human genome, accelerating the identification of disease "
    "genes. Immunology studies the immune system and has enabled the development of immunotherapies "
    "for cancer, autoimmune diseases, and allergies. Monoclonal antibodies, engineered to recognize "
    "specific molecular targets, have become important drugs for cancers and inflammatory diseases. "
    "Medical imaging technologies including X-ray, CT, MRI, and PET scanning allow non-invasive "
    "visualization of anatomy and function. Evidence-based medicine emphasizes systematic evaluation "
    "of clinical evidence through randomized controlled trials and meta-analyses to guide treatment "
    "decisions. Public health addresses disease at the population level through epidemiology, "
    "vaccination programs, sanitation, and behavioral interventions. The COVID-19 pandemic demonstrated "
    "both the power of modern vaccine development — mRNA vaccines were developed in record time — "
    "and the challenges of coordinating global public health responses. Mental health disorders, "
    "including depression, anxiety, schizophrenia, and addiction, represent a major and often "
    "undertreated global burden of disease that deserves far greater research attention and funding. "
)

# ======================================================================
# QA PASSAGES — 5 diverse passage+question pairs (~500-900 tokens each)
# ======================================================================
QA_PAIRS = [
    # 1. Factual recall: "what year did X happen?"
    {
        "id": "QA1_factual_recall",
        "type": "Factual Recall",
        "passage": (
            "The history of computing begins long before the modern digital era. Charles Babbage "
            "conceived the Difference Engine in 1822, a mechanical device designed to tabulate "
            "polynomial functions. He later designed the Analytical Engine, which included an "
            "arithmetic logic unit, conditional branching, and loops — concepts central to modern "
            "computing — though it was never completed in his lifetime. Ada Lovelace wrote what is "
            "often considered the first algorithm intended for such a machine, earning her the title "
            "of the first computer programmer. In the 20th century, Alan Turing published his "
            "landmark paper 'On Computable Numbers' in 1936, introducing the concept of a Turing "
            "machine as a theoretical model of computation that defines what is computationally "
            "possible. Turing's work during World War II at Bletchley Park helped crack the German "
            "Enigma cipher, saving countless lives. The first general-purpose electronic computer, "
            "ENIAC (Electronic Numerical Integrator and Computer), was completed in 1945 at the "
            "University of Pennsylvania. It weighed 30 tons, used 18,000 vacuum tubes, and could "
            "perform 5,000 additions per second. The stored-program concept, formalized by John von "
            "Neumann in 1945, separated programs from hardware and became the foundation of modern "
            "computer architecture. Transistors replaced vacuum tubes in the late 1950s, enabling "
            "smaller, faster, and more reliable computers. The integrated circuit, invented by Jack "
            "Kilby at Texas Instruments and Robert Noyce at Fairchild Semiconductor in 1958 and 1959 "
            "respectively, packed multiple transistors onto a single chip. Gordon Moore observed in "
            "1965 that the number of transistors on integrated circuits doubled approximately every "
            "two years — Moore's Law — which held remarkably true for decades. The microprocessor, "
            "the first complete CPU on a single chip, was introduced by Intel in 1971 with the 4004. "
            "The personal computer revolution began in the late 1970s with machines like the Apple II "
            "(1977) and the IBM PC (1981). Tim Berners-Lee invented the World Wide Web in 1989 at "
            "CERN, creating the hyperlink-based system of documents accessible over the internet that "
            "transformed communication, commerce, and culture globally. The smartphone era began in "
            "earnest in 2007 when Apple introduced the iPhone, combining a phone, music player, and "
            "internet communicator in one pocket-sized device. Cloud computing emerged in the 2000s, "
            "allowing users to access computing resources over the internet rather than owning "
            "dedicated hardware. Today, deep learning models trained on billions of parameters run "
            "on specialized hardware like GPUs and TPUs, powering applications from image recognition "
            "to natural language understanding at a scale unimaginable to the pioneers of computing."
        ),
        "question": "According to the passage, what year was ENIAC completed?",
        "expected_keywords": ["1945"],
    },
    # 2. Reasoning: "why did X lead to Y?"
    {
        "id": "QA2_reasoning",
        "type": "Reasoning",
        "passage": (
            "The Black Death, caused by the bacterium Yersinia pestis, swept across Eurasia between "
            "1347 and 1351, killing an estimated 30 to 60 percent of Europe's population. The plague "
            "arrived in Europe via Genoese trading ships that docked at the Sicilian port of Messina "
            "in October 1347. Sailors aboard the ships were either dead or covered in black boils "
            "oozing blood and pus — the bubonic plague's characteristic symptom. Despite orders to "
            "leave, the disease had already spread ashore. It moved rapidly along trade routes, "
            "reaching France and England by 1348 and Scandinavia and Poland by 1350. The social "
            "consequences were profound and far-reaching. With up to half the labor force dead, "
            "surviving workers found themselves in a position of unprecedented bargaining power. "
            "Lords who had previously extracted labor services from serfs under the feudal system "
            "found themselves competing for agricultural workers, driving wages sharply upward. "
            "Peasants who had been legally bound to the land began to demand wages and freedom of "
            "movement, and many simply left their manors to seek better conditions elsewhere. This "
            "labor scarcity directly accelerated the collapse of serfdom across much of Western "
            "Europe. The economic disruption also spurred technological innovation: labor-saving "
            "devices became more economically attractive when labor was scarce and expensive. The "
            "printing press, which later democratized literacy, and advances in agricultural "
            "machinery were partly driven by the need to do more with fewer hands. The psychological "
            "and cultural impact was equally significant. The Church, which had promised protection "
            "through prayer and had no medical answer for the pandemic, suffered a severe loss of "
            "authority. Flagellant movements arose, with believers publicly whipping themselves to "
            "appease God. Anti-Jewish pogroms swept through Germany and elsewhere as desperate "
            "populations sought scapegoats. The sense that death could arrive at any moment "
            "permeated art and literature, giving rise to the 'Dance of Death' motif and a pervasive "
            "cultural preoccupation with mortality that influenced European thought for generations. "
            "Paradoxically, the survivors often found themselves heirs to the land and property of "
            "the dead, and per-capita wealth in some regions actually rose after the plague, even as "
            "overall population and economic output fell dramatically."
        ),
        "question": (
            "Based on the passage, why did the Black Death lead to the collapse of serfdom "
            "in Western Europe?"
        ),
        "expected_keywords": ["labor", "wages", "bargaining", "serf", "workers"],
    },
    # 3. Detail extraction: "what are the three main causes/features mentioned?"
    {
        "id": "QA3_detail_extraction",
        "type": "Detail Extraction",
        "passage": (
            "Climate change refers to long-term shifts in global temperatures and weather patterns. "
            "While some climate change is natural, since the mid-20th century human activities have "
            "been the primary driver of climate change. The burning of fossil fuels — coal, oil, and "
            "natural gas — releases carbon dioxide and other greenhouse gases into the atmosphere. "
            "These gases trap heat from the sun, causing the average surface temperature of the Earth "
            "to rise, a process known as the greenhouse effect. Deforestation is the second major "
            "human driver: forests absorb carbon dioxide, and when trees are cut or burned, that "
            "stored carbon is released while the land's capacity to absorb future emissions is "
            "permanently reduced. Agricultural practices form the third major source of emissions: "
            "livestock farming produces methane through enteric fermentation in cattle, rice paddies "
            "emit methane during flooding, and nitrogen fertilizers release nitrous oxide — both "
            "potent greenhouse gases with warming potential far exceeding carbon dioxide. Industrial "
            "processes including cement production, steel manufacturing, and chemical synthesis also "
            "release significant quantities of carbon dioxide and other greenhouse gases. The "
            "consequences of rising temperatures are widespread. Sea levels are rising as glaciers "
            "and ice sheets melt and as warmer water expands in volume. Extreme weather events "
            "including hurricanes, droughts, floods, and heat waves are becoming more frequent and "
            "intense. Ocean acidification, caused by the absorption of excess carbon dioxide, "
            "threatens marine ecosystems including coral reefs. Biodiversity is at risk as habitats "
            "shift faster than many species can adapt. The Intergovernmental Panel on Climate Change "
            "(IPCC) has stated that limiting warming to 1.5 degrees Celsius above pre-industrial "
            "levels requires cutting global emissions in half by 2030 and reaching net zero by 2050. "
            "Achieving this requires rapid deployment of renewable energy, electrification of "
            "transport and industry, improved energy efficiency, reforestation, and the development "
            "of carbon capture technologies. International agreements including the Paris Agreement "
            "aim to coordinate national efforts, though the gap between pledges and actions "
            "necessary to meet the 1.5 degree target remains very large."
        ),
        "question": (
            "What are the three main human causes of climate change mentioned in the passage?"
        ),
        "expected_keywords": ["fossil fuels", "deforestation", "agricultural"],
    },
    # 4. Comparison: "how does X differ from Y?"
    {
        "id": "QA4_comparison",
        "type": "Comparison",
        "passage": (
            "The human immune system operates through two broad defensive strategies that work in "
            "concert to protect the body from pathogens. The first is the innate immune system, "
            "which provides an immediate, non-specific response to any perceived threat. When a "
            "pathogen breaches the skin or mucous membranes, innate immune cells such as "
            "macrophages and neutrophils engulf and destroy foreign material through phagocytosis. "
            "Natural killer cells identify and destroy virus-infected cells and cancer cells by "
            "detecting the absence of normal cell-surface markers. The innate response also triggers "
            "inflammation — increased blood flow and permeability that recruits more immune cells "
            "to the site of infection. Fever is another component, as elevated body temperature "
            "inhibits bacterial growth and accelerates immune cell activity. The innate immune "
            "response acts within minutes to hours, providing a crucial first line of defense, but "
            "it lacks the ability to learn or remember specific pathogens. The adaptive immune "
            "system, by contrast, is highly specific and develops over days to weeks after initial "
            "exposure to a pathogen. It operates through lymphocytes: B-cells produce antibodies "
            "that bind to specific antigens on the surface of pathogens, neutralizing them or "
            "marking them for destruction. T-cells coordinate the immune response and directly "
            "kill infected cells. After an infection is cleared, a subset of B-cells and T-cells "
            "persist as memory cells, enabling a much faster and stronger response if the same "
            "pathogen is encountered again. This immunological memory is the basis of vaccination: "
            "exposing the body to a harmless version of a pathogen primes the adaptive immune "
            "system to respond rapidly to future real infections. Unlike the innate system, the "
            "adaptive immune system can distinguish between thousands of different pathogens and "
            "tailor its response precisely. However, this specificity comes at a cost: the adaptive "
            "response is slow to develop on first encounter and can sometimes attack the body's "
            "own tissues, causing autoimmune diseases such as rheumatoid arthritis and type 1 diabetes."
        ),
        "question": (
            "According to the passage, how does the adaptive immune system differ from "
            "the innate immune system?"
        ),
        "expected_keywords": ["specific", "memory", "slow", "B-cells", "T-cells", "antibodies"],
    },
    # 5. Numerical: "what percentage/number is associated with X?"
    {
        "id": "QA5_numerical",
        "type": "Numerical",
        "passage": (
            "The global energy system is undergoing a rapid transformation as the world moves away "
            "from fossil fuels toward renewable energy sources. Solar photovoltaic and wind power "
            "have experienced extraordinary cost reductions over the past decade. The cost of "
            "utility-scale solar power fell by approximately 90 percent between 2010 and 2023, "
            "making it the cheapest source of new electricity generation in most of the world. "
            "Onshore wind power costs fell by roughly 70 percent over the same period. Battery "
            "storage costs, critical for managing the intermittent nature of renewables, declined "
            "by about 97 percent between 1991 and 2022, enabling the electrification of transport "
            "and the integration of variable renewables into the grid. Despite this progress, "
            "fossil fuels still accounted for approximately 80 percent of global primary energy "
            "consumption as of 2022, with coal, oil, and natural gas each playing significant roles. "
            "Electricity represents only about 20 percent of final energy consumption globally; "
            "the remainder is used directly as heat or as fuel in transport, industry, and "
            "agriculture — sectors that are harder to electrify. The International Energy Agency "
            "estimates that to reach net-zero emissions by 2050, the share of clean electricity "
            "in total generation must rise from approximately 30 percent today to over 90 percent "
            "by mid-century. This requires building roughly three times the current installed "
            "capacity of wind and solar by 2030. The transition also demands massive investment "
            "in grid infrastructure, including long-distance transmission lines and smart grids "
            "capable of balancing supply and demand in real time. Nuclear power, which currently "
            "provides about 10 percent of global electricity with essentially zero direct carbon "
            "emissions, is viewed by some analysts as a necessary complement to variable renewables. "
            "Energy efficiency improvements — reducing the energy needed per unit of economic output "
            "— are considered by most experts to be the single largest potential contribution to "
            "meeting climate targets, with the IEA estimating that efficiency measures could deliver "
            "more than 40 percent of the emissions reductions needed by 2040."
        ),
        "question": (
            "According to the passage, by approximately what percentage did the cost of "
            "utility-scale solar power fall between 2010 and 2023?"
        ),
        "expected_keywords": ["90", "90 percent", "ninety"],
    },
]


# ======================================================================
# TASK A: Fixed Scorer 8K Rerun on Mistral-7B
# ======================================================================
@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    secrets=[HF_SECRET],
    memory=32768,
)
def run_fixed_scorer():
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
    print("NEXUSQUANT — TASK A: Fixed Scorer 8K Rerun on Mistral-7B")
    print("=" * 80)
    print("Bug fix: obs_window = max(32, prefix_len // 16)")
    print("Old bug: obs_window=32 hardcoded → at 2924 prefix only 1% of context visible")
    print("Fix:     obs_window=182 at 2924 prefix → 6.25% of context, sees meaningful history")
    print()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "mistralai/Mistral-7B-v0.1"
    print(f"\nLoading {model_id}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, token=os.environ["HF_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        token=os.environ["HF_TOKEN"],
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s")

    n_layers   = model.config.num_hidden_layers   # 32
    n_kv_heads = model.config.num_key_value_heads # 8
    head_dim   = model.config.hidden_size // model.config.num_attention_heads  # 128
    rope_base  = getattr(model.config, 'rope_theta', 10000.0)
    print(f"Config: {n_layers}L, {n_kv_heads}KVH, d={head_dim}, rope_theta={rope_base}")

    sliding_window = 32

    # ------------------------------------------------------------------ FIXED importance scorer
    def score_importance(kv_cache, n_layers, n_kv_heads, head_dim, prefix_len):
        obs_window = max(32, prefix_len // 16)  # FIXED: scales with context
        print(f"  [scorer] prefix_len={prefix_len}, obs_window={obs_window} "
              f"({obs_window/prefix_len*100:.1f}% of context)")
        k0, _ = get_kv(kv_cache, 0)
        seq_len = k0.shape[2]
        w = min(obs_window, seq_len)
        all_imp = torch.zeros(seq_len, device='cpu')
        for l in range(n_layers):
            kl, _ = get_kv(kv_cache, l)
            k = kl[0].float().cpu()   # [n_kv_heads, seq_len, head_dim]
            k_obs = k[:, -w:, :]
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(k_obs, k.transpose(-2, -1)) * scale
            all_pos = torch.arange(seq_len).unsqueeze(0)
            obs_pos = torch.arange(seq_len - w, seq_len).unsqueeze(1)
            causal = (all_pos <= obs_pos)
            scores = scores.masked_fill(~causal.unsqueeze(0), float('-inf'))
            attn = F.softmax(scores, dim=-1, dtype=torch.float32)
            layer_imp = attn.sum(dim=1).mean(dim=0)
            pool_kernel = 5
            if seq_len > pool_kernel:
                imp_1d = layer_imp.unsqueeze(0).unsqueeze(0)
                layer_imp = F.avg_pool1d(imp_1d, kernel_size=pool_kernel,
                                          padding=pool_kernel // 2, stride=1).squeeze()[:seq_len]
            all_imp += layer_imp
        return all_imp / n_layers

    # ------------------------------------------------------------------ evict + 2-bit E8 quantize
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
        mask_bytes  = math.ceil(prefix_len / 8) * n_layers
        total = total_idx + scale_bytes + mask_bytes
        return {
            "fp16": total_fp16, "idx": total_idx, "scale": scale_bytes,
            "mask": mask_bytes, "total": total,
            "ratio": total_fp16 / total if total > 0 else 0,
            "n_kept": n_kept,
        }, kv_cache

    # ------------------------------------------------------------------ build keep mask
    def build_keep_mask(prefix_len, evict_pct, importance):
        if evict_pct == 0:
            return torch.ones(prefix_len, dtype=torch.bool)
        keep_mask = torch.zeros(prefix_len, dtype=torch.bool)
        keep_mask[0] = True
        keep_mask[-sliding_window:] = True
        n_to_keep = max(int(prefix_len * (100 - evict_pct) / 100), sliding_window + 1)
        n_from_imp = n_to_keep - keep_mask.sum().item()
        if n_from_imp > 0:
            imp = importance.clone()
            imp[keep_mask] = -float('inf')
            _, top_idx = imp.topk(min(n_from_imp, (~keep_mask).sum().item()))
            keep_mask[top_idx] = True
        return keep_mask

    # ------------------------------------------------------------------ tokenize (same as old 8K run)
    inputs = tok(MULTI_TOPIC_TEXT, return_tensors="pt", max_length=8192, truncation=True)
    full_ids = inputs.input_ids.to("cuda")
    n_tok = full_ids.shape[1]
    prefix_len = n_tok // 2
    cont_len = n_tok - prefix_len
    print(f"\nText: tokens={n_tok}, prefix={prefix_len}, continuation={cont_len}")

    obs_window_fixed = max(32, prefix_len // 16)
    print(f"FIXED obs_window = max(32, {prefix_len}//16) = {obs_window_fixed} "
          f"(old was 32, now {obs_window_fixed})")

    # ------------------------------------------------------------------ baseline PPL
    with torch.no_grad():
        pout = model(full_ids[:, :prefix_len], use_cache=True)
        cout = model(full_ids[:, prefix_len:], past_key_values=pout.past_key_values, use_cache=True)
        logits = cout.logits[:, :-1, :].float()
        targets = full_ids[:, prefix_len + 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        baseline_ppl = torch.exp(loss).item()
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    # ------------------------------------------------------------------ compute importance once (FIXED)
    with torch.no_grad():
        pout = model(full_ids[:, :prefix_len], use_cache=True)
    importance = score_importance(pout.past_key_values, n_layers, n_kv_heads, head_dim, prefix_len)

    # ------------------------------------------------------------------ eviction sweep
    configs = [
        {"name": "2b no-evict",  "evict_pct": 0},
        {"name": "2b+35%evict",  "evict_pct": 35},
        {"name": "2b+50%evict",  "evict_pct": 50},
        {"name": "2b+60%evict",  "evict_pct": 60},
        {"name": "2b+70%evict",  "evict_pct": 70},
        {"name": "2b+80%evict",  "evict_pct": 80},
    ]

    print(f"\n{'Config':<20s} {'PPL':>8s} {'Delta%':>9s} {'Ratio':>7s} {'Kept':>6s}")
    print("-" * 56)
    results = []

    for cfg in configs:
        torch.cuda.empty_cache()
        evict_pct = cfg["evict_pct"]

        with torch.no_grad():
            pout = model(full_ids[:, :prefix_len], use_cache=True)
            kv = pout.past_key_values

        keep_mask = build_keep_mask(prefix_len, evict_pct, importance)
        info, kv = evict_quantize(kv, keep_mask, key_bits=2, val_bits=2, prefix_len=prefix_len)

        evict_mask = ~keep_mask
        attn_ctx = torch.ones(prefix_len, dtype=torch.long, device="cuda")
        attn_ctx[evict_mask] = 0
        attn_full = torch.cat([attn_ctx, torch.ones(cont_len, dtype=torch.long, device="cuda")])

        with torch.no_grad():
            cout = model(full_ids[:, prefix_len:], past_key_values=kv,
                         attention_mask=attn_full.unsqueeze(0), use_cache=True)
            logits = cout.logits[:, :-1, :].float()
            targets = full_ids[:, prefix_len + 1:]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            ppl = torch.exp(loss).item()

        delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
        print(f"{cfg['name']:<20s} {ppl:8.4f} {delta:+8.2f}% {info['ratio']:6.2f}x {info['n_kept']:5d}")
        results.append({
            "name": cfg["name"], "evict_pct": evict_pct,
            "ppl": ppl, "baseline_ppl": baseline_ppl,
            "delta": delta, "ratio": info["ratio"],
            "n_kept": info["n_kept"],
            "prefix_len": prefix_len,
            "obs_window_fixed": obs_window_fixed,
        })

    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # Comparison table vs old broken scorer
    OLD_RESULTS = {
        0:  {"delta": None,    "note": "no-evict baseline"},
        35: {"delta": +1.476,  "note": "old broken scorer"},
        50: {"delta": +5.398,  "note": "old broken scorer"},
        60: {"delta": +42.084, "note": "old broken scorer"},
        70: {"delta": +43.272, "note": "old broken scorer"},
        80: {"delta": +48.782, "note": "old broken scorer"},
    }
    print("\n" + "=" * 70)
    print("COMPARISON: OLD (broken obs_window=32) vs FIXED scorer")
    print("=" * 70)
    print(f"{'Evict%':<8s} {'OLD scorer Δ%':>14s} {'FIXED scorer Δ%':>16s} {'Ratio':>8s}")
    print("-" * 50)
    for r in results:
        ep = r["evict_pct"]
        old = OLD_RESULTS.get(ep, {}).get("delta")
        old_str = f"{old:+.2f}%" if old is not None else "baseline"
        print(f"{ep:<8d} {old_str:>14s} {r['delta']:>+14.2f}% {r['ratio']:>7.2f}x")

    return results, prefix_len, obs_window_fixed


# ======================================================================
# TASK B: LongBench-Style QA Eval on Mistral-7B
# ======================================================================
@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    secrets=[HF_SECRET],
    memory=32768,
)
def run_qa_eval():
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
    print("NEXUSQUANT — TASK B: LongBench-Style QA Eval on Mistral-7B")
    print("=" * 80)
    print("5 diverse QA pairs | Baseline vs 35% vs 60% eviction | MATCH/PARTIAL/WRONG")
    print()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "mistralai/Mistral-7B-v0.1"
    print(f"\nLoading {model_id}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, token=os.environ["HF_TOKEN"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        token=os.environ["HF_TOKEN"],
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s")

    n_layers   = model.config.num_hidden_layers   # 32
    n_kv_heads = model.config.num_key_value_heads # 8
    head_dim   = model.config.hidden_size // model.config.num_attention_heads  # 128
    rope_base  = getattr(model.config, 'rope_theta', 10000.0)
    print(f"Config: {n_layers}L, {n_kv_heads}KVH, d={head_dim}, rope_theta={rope_base}")

    sliding_window = 32

    # ------------------------------------------------------------------ FIXED importance scorer
    def score_importance(kv_cache, prefix_len):
        obs_window = max(32, prefix_len // 16)
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
            pool_kernel = 5
            if seq_len > pool_kernel:
                imp_1d = layer_imp.unsqueeze(0).unsqueeze(0)
                layer_imp = F.avg_pool1d(imp_1d, kernel_size=pool_kernel,
                                          padding=pool_kernel // 2, stride=1).squeeze()[:seq_len]
            all_imp += layer_imp
        return all_imp / n_layers

    # ------------------------------------------------------------------ evict + 2-bit E8 quantize
    def evict_quantize(kv_cache, keep_mask, key_bits, val_bits, prefix_len):
        H = hadamard_matrix(head_dim).cpu()
        n_kept = keep_mask.sum().item()
        total_fp16 = 0
        cctx = zstandard.ZstdCompressor(level=22)
        all_key_coords = []
        all_val_coords = []

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
        mask_bytes  = math.ceil(prefix_len / 8) * n_layers
        total = total_idx + scale_bytes + mask_bytes
        return {
            "fp16": total_fp16, "total": total,
            "ratio": total_fp16 / total if total > 0 else 0,
            "n_kept": n_kept,
        }, kv_cache

    # ------------------------------------------------------------------ build keep mask
    def build_keep_mask(prefix_len, evict_pct, importance):
        if evict_pct == 0:
            return torch.ones(prefix_len, dtype=torch.bool)
        keep_mask = torch.zeros(prefix_len, dtype=torch.bool)
        keep_mask[0] = True
        keep_mask[-sliding_window:] = True
        n_to_keep = max(int(prefix_len * (100 - evict_pct) / 100), sliding_window + 1)
        n_from_imp = n_to_keep - keep_mask.sum().item()
        if n_from_imp > 0:
            imp = importance.clone()
            imp[keep_mask] = -float('inf')
            _, top_idx = imp.topk(min(n_from_imp, (~keep_mask).sum().item()))
            keep_mask[top_idx] = True
        return keep_mask

    # ------------------------------------------------------------------ keyword scoring
    def score_answer(answer_text, expected_keywords):
        answer_lower = answer_text.lower()
        found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        n_found = len(found)
        n_total = len(expected_keywords)
        if n_found == n_total:
            return "MATCH", found
        elif n_found >= max(1, n_total // 2):
            return "PARTIAL", found
        else:
            return "WRONG", found

    # ------------------------------------------------------------------ run QA eval
    qa_results = []
    evict_configs = [
        {"name": "baseline",  "evict_pct": 0},
        {"name": "35%evict",  "evict_pct": 35},
        {"name": "60%evict",  "evict_pct": 60},
    ]

    for qa_idx, qa in enumerate(QA_PAIRS):
        print(f"\n{'='*70}")
        print(f"QA {qa_idx+1}: [{qa['type']}] {qa['id']}")
        print(f"Question: {qa['question']}")
        print(f"Expected keywords: {qa['expected_keywords']}")

        prompt_text = qa["passage"] + "\n\nQuestion: " + qa["question"] + "\nAnswer:"
        inputs_enc = tok(prompt_text, return_tensors="pt")
        prefix_ids = inputs_enc.input_ids.to("cuda")
        prefix_len = prefix_ids.shape[1]
        print(f"Prompt tokens: {prefix_len}")

        # Compute importance once (FIXED obs_window) using full prefix KV
        with torch.no_grad():
            pout_imp = model(prefix_ids, use_cache=True)
        importance = score_importance(pout_imp.past_key_values, prefix_len)

        qa_trial_results = []
        for cfg in evict_configs:
            torch.cuda.empty_cache()
            evict_pct = cfg["evict_pct"]

            # Run full prefix to get KV cache
            with torch.no_grad():
                pout = model(prefix_ids, use_cache=True)
                kv = pout.past_key_values

            if evict_pct > 0:
                keep_mask = build_keep_mask(prefix_len, evict_pct, importance)
                info, kv = evict_quantize(kv, keep_mask, key_bits=2, val_bits=2,
                                          prefix_len=prefix_len)
                evict_mask = ~keep_mask
                attn_past = torch.ones(prefix_len, dtype=torch.long, device="cuda")
                attn_past[evict_mask] = 0
                ratio = info["ratio"]
            else:
                attn_past = torch.ones(prefix_len, dtype=torch.long, device="cuda")
                ratio = 1.0

            # Manual greedy decoding — avoids model.generate() internal cache_position issues
            # with custom-modified KV caches. Feed one token at a time.
            generated_ids = []
            cur_kv = kv
            # Build initial attention mask: [1, prefix_len] (all prefix positions)
            cur_attn = attn_past.unsqueeze(0)  # [1, prefix_len]
            # Get the first new token by feeding a dummy BOS to probe next-token logits
            # Actually: the prefix KV already includes all prefix tokens, so we need
            # to feed a continuation token. Use the last logits from pout directly.
            last_logits = pout.logits[:, -1, :].float()  # [1, vocab]
            next_token = last_logits.argmax(dim=-1, keepdim=True)  # [1, 1]
            generated_ids.append(next_token.item())

            for _ in range(99):  # generate up to 99 more tokens (100 total)
                if next_token.item() == tok.eos_token_id:
                    break
                # extend attention mask by 1 for the newly generated token
                cur_attn = torch.cat([
                    cur_attn,
                    torch.ones(1, 1, dtype=torch.long, device="cuda"),
                ], dim=1)
                with torch.no_grad():
                    out = model(next_token, past_key_values=cur_kv,
                                attention_mask=cur_attn, use_cache=True)
                cur_kv = out.past_key_values
                next_token = out.logits[:, -1, :].float().argmax(dim=-1, keepdim=True)
                generated_ids.append(next_token.item())

            answer_text = tok.decode(generated_ids, skip_special_tokens=True).strip()

            score, found_kw = score_answer(answer_text, qa["expected_keywords"])
            print(f"  [{cfg['name']:10s}] {score:7s} | found={found_kw} | ratio={ratio:.1f}x")
            print(f"               Answer: {answer_text[:200]}")

            qa_trial_results.append({
                "config": cfg["name"],
                "evict_pct": evict_pct,
                "answer": answer_text,
                "score": score,
                "found_keywords": found_kw,
                "ratio": ratio,
            })

        qa_results.append({
            "id": qa["id"],
            "type": qa["type"],
            "question": qa["question"],
            "expected_keywords": qa["expected_keywords"],
            "prefix_len": prefix_len,
            "trials": qa_trial_results,
        })

    # ------------------------------------------------------------------ summary table
    print("\n" + "=" * 80)
    print("QA EVAL SUMMARY")
    print("=" * 80)
    print(f"{'QA':4s} {'Type':22s} {'baseline':>10s} {'35%evict':>10s} {'60%evict':>10s}")
    print("-" * 60)
    for qa_r in qa_results:
        scores = {t["config"]: t["score"] for t in qa_r["trials"]}
        b  = scores.get("baseline", "N/A")
        e35 = scores.get("35%evict", "N/A")
        e60 = scores.get("60%evict", "N/A")
        print(f"{qa_r['id'][:4]:4s} {qa_r['type']:22s} {b:>10s} {e35:>10s} {e60:>10s}")

    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    return qa_results


# ======================================================================
# LOCAL ENTRYPOINT
# ======================================================================
@app.local_entrypoint()
def main():
    import time

    print("\n" + "=" * 80)
    print("NEXUSQUANT: Fixed Scorer 8K Rerun + LongBench-Style QA Eval")
    print("=" * 80)

    print("\nLaunching Task A: Fixed Scorer 8K Rerun on Mistral-7B...")
    fa = run_fixed_scorer.spawn()

    print("Launching Task B: LongBench-Style QA Eval on Mistral-7B...")
    fb = run_qa_eval.spawn()

    print("\nWaiting for both tasks to complete...\n")
    results_a_raw = fa.get()
    results_b = fb.get()

    results_a, prefix_len, obs_window = results_a_raw

    # ------------------------------------------------------------------ old results for comparison
    OLD_RESULTS = {
        0:  None,
        35: +1.476,
        50: +5.398,
        60: +42.084,
        70: +43.272,
        80: +48.782,
    }

    # ------------------------------------------------------------------ print Task A summary
    print("\n" + "=" * 80)
    print(f"TASK A: FIXED SCORER RESULTS (prefix={prefix_len}, obs_window={obs_window})")
    print("=" * 80)
    print(f"{'Evict%':<8s} {'OLD scorer Δ%':>14s} {'FIXED scorer Δ%':>16s} {'Ratio':>8s} {'Kept':>7s}")
    print("-" * 58)
    for r in results_a:
        ep = r["evict_pct"]
        old = OLD_RESULTS.get(ep)
        old_str = f"{old:+.3f}%" if old is not None else "baseline"
        tag = ""
        if old is not None and r["delta"] < old:
            tag = " IMPROVED"
        elif old is not None and r["delta"] > old + 5:
            tag = " WORSE"
        print(f"{ep:<8d} {old_str:>14s} {r['delta']:>+14.3f}% {r['ratio']:>7.2f}x {r['n_kept']:>6d}{tag}")

    # ------------------------------------------------------------------ print Task B summary
    print("\n" + "=" * 80)
    print("TASK B: QA EVAL RESULTS")
    print("=" * 80)
    print(f"{'#':3s} {'Type':22s} {'Question (short)':35s} {'baseline':>9s} {'35%':>9s} {'60%':>9s}")
    print("-" * 92)
    for i, qa_r in enumerate(results_b):
        q_short = qa_r["question"][:33] + ".." if len(qa_r["question"]) > 35 else qa_r["question"]
        scores = {t["config"]: t["score"] for t in qa_r["trials"]}
        b   = scores.get("baseline", "N/A")
        e35 = scores.get("35%evict", "N/A")
        e60 = scores.get("60%evict", "N/A")
        print(f"{i+1:3d} {qa_r['type']:22s} {q_short:35s} {b:>9s} {e35:>9s} {e60:>9s}")

    # ------------------------------------------------------------------ write results file
    out_dir = os.path.join(os.path.dirname(__file__), "..", ".company", "engineering")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "final_fixes_results.md")

    with open(out_path, "w") as f:
        f.write("# Final Fixes Results — NexusQuant\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write("**Model:** mistralai/Mistral-7B-v0.1 (32L, 8KVH, d=128, rope_theta=10000)\n")
        f.write("**Pipeline:** RoPE removal → Hadamard → 2-bit E8 VQ → temporal delta → zstd-22\n\n")

        f.write("---\n\n")
        f.write("## Task A: Fixed Importance Scorer — 8K Context Rerun\n\n")
        f.write("### The Bug\n\n")
        f.write("```python\n")
        f.write("# OLD (broken):\n")
        f.write("def score_importance(kv_cache, obs_window=32):  # hardcoded!\n")
        f.write("    # At 2924 prefix → observes only last 32/2924 = 1.1% of context\n\n")
        f.write("# FIXED:\n")
        f.write("def score_importance(kv_cache, n_layers, n_kv_heads, head_dim, prefix_len):\n")
        f.write("    obs_window = max(32, prefix_len // 16)  # scales with context\n")
        f.write("    # At 2924 prefix → obs_window=182, observes 6.25% of context\n")
        f.write("```\n\n")
        f.write(f"**Context:** {prefix_len*2} tokens total → {prefix_len} prefix + {prefix_len} continuation\n")
        f.write(f"**obs_window (fixed):** {obs_window} tokens "
                f"({obs_window/prefix_len*100:.1f}% of prefix vs old 32/{prefix_len}="
                f"{32/prefix_len*100:.1f}%)\n")
        f.write(f"**Baseline PPL:** {results_a[0]['baseline_ppl']:.4f}\n\n")

        f.write("### Comparison Table\n\n")
        f.write("| Evict% | OLD scorer Δ% | FIXED scorer Δ% | Ratio | Kept | Improvement |\n")
        f.write("|--------|--------------|-----------------|-------|------|-------------|\n")
        for r in results_a:
            ep = r["evict_pct"]
            old = OLD_RESULTS.get(ep)
            old_str = f"{old:+.3f}%" if old is not None else "baseline"
            if old is not None:
                improvement = old - r["delta"]
                imp_str = f"{improvement:+.3f}pp"
                verdict = "BETTER" if improvement > 0.5 else ("SIMILAR" if abs(improvement) <= 0.5 else "WORSE")
            else:
                imp_str = "—"
                verdict = "—"
            f.write(f"| {ep}% | {old_str} | {r['delta']:+.3f}% | {r['ratio']:.2f}x "
                    f"| {r['n_kept']} | {imp_str} ({verdict}) |\n")

        f.write("\n### Key Finding\n\n")
        # Find evict configs where delta improved
        improved = [(r, OLD_RESULTS[r["evict_pct"]]) for r in results_a
                    if r["evict_pct"] > 0 and OLD_RESULTS.get(r["evict_pct"]) is not None
                    and r["delta"] < OLD_RESULTS[r["evict_pct"]]]
        if improved:
            f.write(f"- Fixed scorer improved quality on **{len(improved)}/5** eviction configs\n")
            best_imp = max(improved, key=lambda x: x[1] - x[0]["delta"])
            r_best, old_best = best_imp
            f.write(f"- Biggest improvement: **{r_best['name']}** → "
                    f"old {old_best:+.3f}% vs fixed {r_best['delta']:+.3f}% "
                    f"(+{old_best - r_best['delta']:.3f}pp better)\n")
        else:
            f.write("- Fixed scorer results recorded for all eviction levels\n")
        f.write(f"- Best fixed-scorer config: ")
        best_r = min([r for r in results_a if r["evict_pct"] > 0], key=lambda x: abs(x["delta"]))
        f.write(f"**{best_r['name']}** → Δ%={best_r['delta']:+.3f}%, ratio={best_r['ratio']:.2f}x\n\n")

        f.write("---\n\n")
        f.write("## Task B: LongBench-Style QA Evaluation\n\n")
        f.write("**Setup:** Passage as prefix, question appended, model generates answer (max_new_tokens=100, do_sample=False)\n")
        f.write("**Compression:** baseline (no compression), 35% eviction, 60% eviction\n")
        f.write("**Scoring:** MATCH = all expected keywords found | PARTIAL = ≥50% found | WRONG = <50% found\n\n")

        f.write("### Summary Table\n\n")
        f.write("| # | Type | Question | Baseline | 35% evict | 60% evict |\n")
        f.write("|---|------|----------|----------|-----------|----------|\n")
        for i, qa_r in enumerate(results_b):
            q_short = qa_r["question"][:60] + "..." if len(qa_r["question"]) > 60 else qa_r["question"]
            scores = {t["config"]: t["score"] for t in qa_r["trials"]}
            b   = scores.get("baseline", "N/A")
            e35 = scores.get("35%evict", "N/A")
            e60 = scores.get("60%evict", "N/A")
            f.write(f"| {i+1} | {qa_r['type']} | {q_short} | {b} | {e35} | {e60} |\n")

        f.write("\n### Detailed Results\n\n")
        for i, qa_r in enumerate(results_b):
            f.write(f"#### QA{i+1}: {qa_r['type']} — {qa_r['id']}\n\n")
            f.write(f"**Question:** {qa_r['question']}\n\n")
            f.write(f"**Expected keywords:** {qa_r['expected_keywords']}\n\n")
            f.write(f"**Passage tokens:** {qa_r['prefix_len']}\n\n")
            for trial in qa_r["trials"]:
                f.write(f"**{trial['config']}** (ratio={trial['ratio']:.1f}x): "
                        f"**{trial['score']}** | found={trial['found_keywords']}\n\n")
                f.write(f"> {trial['answer'][:500]}\n\n")

        f.write("---\n\n")
        f.write("## Summary\n\n")
        # Score QA degradation
        total_match_baseline = sum(1 for qa_r in results_b
            for t in qa_r["trials"] if t["config"] == "baseline" and t["score"] == "MATCH")
        total_match_35 = sum(1 for qa_r in results_b
            for t in qa_r["trials"] if t["config"] == "35%evict" and t["score"] == "MATCH")
        total_match_60 = sum(1 for qa_r in results_b
            for t in qa_r["trials"] if t["config"] == "60%evict" and t["score"] == "MATCH")
        n_qa = len(results_b)
        f.write(f"- **QA accuracy:** baseline={total_match_baseline}/{n_qa} MATCH, "
                f"35%evict={total_match_35}/{n_qa} MATCH, "
                f"60%evict={total_match_60}/{n_qa} MATCH\n")
        f.write(f"- **Fixed scorer:** obs_window={obs_window} at prefix_len={prefix_len} "
                f"(was 32, now {obs_window})\n")
        best_a = min([r for r in results_a if r["evict_pct"] > 0], key=lambda x: abs(x["delta"]))
        f.write(f"- **Best PPL config:** {best_a['name']} → Δ%={best_a['delta']:+.3f}%, "
                f"ratio={best_a['ratio']:.2f}x\n")

    print(f"\nResults written to: {out_path}")
    print("\nDone.")
