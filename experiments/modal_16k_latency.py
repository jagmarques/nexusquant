"""Combined A100 experiment: 16K context quality sweep + latency measurement on Mistral-7B.

Task A: 16K Context Quality Sweep
  - ~16K tokens (20+ diverse topics), split into 8192 prefix + 8192 continuation
  - Configs: 2b + 0/35/50/60/70/80% eviction
  - Key question: does the 60% eviction catastrophe from 3K prefix persist at 8K+?

Task B: Latency Measurement
  - Baseline vs 35% evict vs 60% evict
  - Prefill / compression / generation (100 tokens) wall-clock time
  - 3 iterations, report mean ± std

Results written to .company/engineering/16k_latency_results.md
"""
import modal
import os

app = modal.App("nexusquant-16k-latency")

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

# ======================================================================
# 16K TEXT CORPUS — 20 diverse topics, ~200 words each ≈ 16K+ tokens
# ======================================================================
TEXT_16K = (
    # 1. Physics (~250 words)
    "The Standard Model of particle physics describes three of the four fundamental forces and classifies "
    "all known elementary particles. Fermions — six quarks and six leptons — are the building blocks of "
    "matter. Gauge bosons mediate the forces between fermions. The Higgs mechanism gives mass to particles "
    "through spontaneous symmetry breaking. The strong force binds quarks into protons and neutrons via "
    "gluon exchange; quantum chromodynamics is its governing theory. The electroweak force unifies "
    "electromagnetism and the weak nuclear force, with the W and Z bosons as mediators. General relativity "
    "describes gravity as spacetime curvature caused by mass and energy; it predicts black holes, "
    "gravitational waves, and the expansion of the universe. Quantum mechanics governs the microscopic "
    "world with wave functions, operators, and the uncertainty principle. The measurement problem and "
    "the interpretation of quantum mechanics remain deeply contested among physicists. Dark matter and "
    "dark energy together constitute roughly 95 percent of the universe's energy content but have not "
    "been directly detected. The Large Hadron Collider confirmed the Higgs boson in 2012, completing "
    "the Standard Model, yet many open questions remain: the hierarchy problem, matter-antimatter "
    "asymmetry, the nature of dark matter, and the unification of gravity with quantum field theory. "
    "Supersymmetry and string theory are candidate extensions of the Standard Model, but neither has "
    "found experimental confirmation. Hawking radiation predicts that black holes slowly evaporate "
    "through quantum effects near the event horizon, linking gravity, thermodynamics, and quantum "
    "mechanics. The arrow of time, entropy, and the second law of thermodynamics pose deep puzzles "
    "about the initial conditions of the universe and the direction of physical processes. "
    # 2. History (~250 words)
    "The Industrial Revolution transformed human society from agrarian economies to machine-based "
    "manufacturing beginning in Britain in the late eighteenth century. James Watt's improved steam "
    "engine became the universal power source; railways linked cities and created national markets. "
    "The factory system concentrated labor and created new urban working classes. Child labor and "
    "dangerous working conditions sparked reform movements and the first labor laws. The Second "
    "Industrial Revolution in the late nineteenth century brought electricity, steel, chemicals, and "
    "mass production. The Roman Empire unified the Mediterranean world under a single legal and "
    "administrative system for five centuries. Roman law, Latin language, engineering, and "
    "infrastructure shaped European civilization profoundly. The fall of the Western Empire in 476 AD "
    "initiated the medieval period of fragmented power and the rise of the Catholic Church. The "
    "Renaissance, beginning in fourteenth-century Italy, revived classical learning, developed "
    "perspective in painting, and produced figures like Leonardo, Michelangelo, and Galileo. The "
    "printing press, invented by Gutenberg around 1440, democratized knowledge and accelerated the "
    "Reformation. The French Revolution of 1789 overthrew the monarchy, proclaimed the rights of man, "
    "and launched an era of nationalism and revolutionary politics. Napoleon spread the Napoleonic "
    "Code across Europe, reshaping legal systems. The World Wars of the twentieth century caused "
    "unprecedented destruction and reshaped the global order, ending European colonial empires and "
    "initiating the Cold War between the United States and the Soviet Union. Decolonization between "
    "1945 and 1975 created dozens of new nations in Africa, Asia, and the Caribbean with complex "
    "legacies of colonialism that persist to the present. "
    # 3. Biology (~250 words)
    "Evolution by natural selection, proposed by Darwin and Wallace, is the unifying theory of biology. "
    "Heritable variation plus differential reproduction causes populations to adapt to their environments "
    "over generations. The modern evolutionary synthesis combines Darwinian selection with Mendelian "
    "genetics and population genetics. DNA carries genetic information in a double-helical structure "
    "discovered by Watson and Crick in 1953, using X-ray data from Rosalind Franklin. The genetic code "
    "maps three-nucleotide codons to the twenty standard amino acids and is nearly universal across life. "
    "The human genome contains about three billion base pairs encoding roughly twenty thousand protein-coding "
    "genes. CRISPR-Cas9 allows precise genome editing with revolutionary implications for medicine and "
    "agriculture. Cell biology distinguishes prokaryotes — bacteria and archaea — from eukaryotes, which "
    "have membrane-bound organelles including nuclei, mitochondria, and chloroplasts. Mitochondria generate "
    "ATP through oxidative phosphorylation; they carry their own circular DNA, evidence of an ancient "
    "endosymbiotic origin. The immune system provides innate and adaptive defenses against pathogens. "
    "T lymphocytes mediate cellular immunity; B lymphocytes produce antibodies. Vaccines train adaptive "
    "immunity without causing disease. Ecosystems are communities of organisms interacting with each other "
    "and their abiotic environment. Energy flows from producers through herbivores and carnivores to "
    "decomposers; nutrients cycle continuously. Biodiversity is threatened by habitat destruction, climate "
    "change, overexploitation, invasive species, and pollution, driving what many scientists call the sixth "
    "mass extinction. Epigenetics studies heritable changes in gene expression that do not involve changes "
    "to the DNA sequence, mediated by methylation and histone modification. "
    # 4. Computer Science (~250 words)
    "The transformer architecture, introduced in Attention Is All You Need in 2017, uses self-attention "
    "to process sequences in parallel and has become the foundation for most modern large language models. "
    "Self-attention computes pairwise interactions between all tokens in a sequence, enabling the model "
    "to capture long-range dependencies. The key-value cache stores intermediate computations during "
    "autoregressive generation, growing linearly with context length and becoming a memory bottleneck "
    "at long contexts. Quantization reduces model precision from 32-bit or 16-bit floating point to "
    "lower bit-widths, trading accuracy for memory and speed. Lattice vector quantization, such as the "
    "E8 lattice, provides theoretically optimal quantization for high-dimensional vectors by packing "
    "sphere of influence maximally efficiently. Eviction policies decide which KV cache tokens to drop "
    "when memory is constrained. Attention-based importance scoring selects the most important tokens "
    "by measuring how much attention the recent context attends to each historical token. RoPE "
    "rotary position embeddings encode position information directly into key and query vectors via "
    "rotation matrices; removing and re-applying RoPE before and after quantization aligns vectors "
    "in a position-free space for better compression. Operating systems manage hardware resources "
    "through process scheduling, virtual memory, file systems, and device drivers. The P versus NP "
    "problem asks whether every problem whose solution can be verified in polynomial time can also "
    "be solved in polynomial time, and is one of the most important open problems in mathematics. "
    "Distributed systems coordinate multiple computers to appear as a single coherent system, "
    "facing challenges of consensus, fault tolerance, and consistency. "
    # 5. Mathematics (~250 words)
    "Pure mathematics discovers patterns and structures through rigorous proof. Number theory studies "
    "integers and primes; the Riemann hypothesis about the zeros of the zeta function is the most "
    "famous unsolved problem, with profound implications for the distribution of prime numbers. "
    "Abstract algebra studies groups, rings, fields, and modules. Group theory underlies the "
    "symmetries of physical laws; Galois theory uses groups to determine which polynomial equations "
    "are solvable by radicals, resolving questions that had puzzled mathematicians for three centuries. "
    "Topology studies properties preserved under continuous deformations. The Poincare conjecture, "
    "proved by Perelman in 2003, characterizes the three-sphere among compact three-manifolds. "
    "Differential geometry describes curved spaces using calculus; it is the mathematical language "
    "of general relativity. Algebraic geometry studies zero sets of polynomial equations and connects "
    "deep arithmetic questions to geometric ones — the Weil conjectures, proved by Deligne, exemplify "
    "this connection. Category theory provides a unifying language for mathematics, abstracting "
    "structural relationships between different fields. Probability theory gives a rigorous foundation "
    "for reasoning under uncertainty; Bayesian inference updates beliefs in light of evidence. The "
    "central limit theorem explains the ubiquity of the normal distribution as the sum of many "
    "independent random variables. Functional analysis studies infinite-dimensional vector spaces "
    "and operators between them, providing the mathematical framework for quantum mechanics. "
    "Combinatorics counts structures, while graph theory models networks; together they underpin "
    "computer science, operations research, and statistical physics. "
    # 6. Philosophy (~250 words)
    "Philosophy examines fundamental questions about reality, knowledge, morality, language, and mind. "
    "Metaphysics asks what exists and what is its nature. Ontology catalogues the categories of being: "
    "substances, properties, events, numbers, and abstract objects. The mind-body problem asks how "
    "physical brain processes give rise to subjective conscious experience. The hard problem of "
    "consciousness — why there is something it is like to see red — resists reduction to functional "
    "or physical descriptions. Dualism posits distinct mental and physical substances; physicalism "
    "holds that everything is physical; functionalism defines mental states by their causal roles. "
    "Epistemology asks what knowledge is and how it is possible. Descartes doubted everything he "
    "could doubt and reached the cogito — I think therefore I am — as a bedrock certainty. Hume "
    "argued that causal necessity cannot be observed but is a habit of mind. Kant synthesized "
    "rationalism and empiricism: the mind imposes structure on experience through pure intuitions "
    "of space and time and the categories of the understanding. Ethics asks how we should act. "
    "Consequentialism judges acts by their outcomes; utilitarianism maximizes total well-being. "
    "Deontology holds that some acts are intrinsically right or wrong regardless of consequences. "
    "Virtue ethics focuses on the character of the agent rather than rules or outcomes. Political "
    "philosophy asks what makes political authority legitimate. Social contract theories from Hobbes, "
    "Locke, and Rousseau ground authority in rational consent. Rawls argued that just principles "
    "are those rational agents would choose behind a veil of ignorance about their own position. "
    # 7. Economics (~250 words)
    "Economics studies how individuals, firms, and governments allocate scarce resources. "
    "Microeconomics models how prices coordinate supply and demand in markets. Consumer choice theory "
    "maximizes utility subject to budget constraints; revealed preference reconstructs preferences "
    "from observed behavior. Production theory models cost minimization and profit maximization by "
    "firms operating in competitive, oligopolistic, or monopolistic markets. Game theory studies "
    "strategic interaction: the Nash equilibrium is a profile of strategies from which no player "
    "has unilateral incentive to deviate. Mechanism design asks how to construct rules that induce "
    "desired outcomes from self-interested agents, with applications in auctions, matching, and "
    "regulation. Macroeconomics examines aggregate variables: output, employment, inflation, and "
    "growth. Keynesian theory holds that aggregate demand drives output in the short run and "
    "justifies fiscal stimulus in recessions. Monetarism emphasizes money supply in determining "
    "nominal output and inflation. Real business cycle theory attributes fluctuations to technology "
    "shocks rather than demand disturbances. Growth theory studies why incomes differ across "
    "countries and grow over time; endogenous growth models show how innovation and human capital "
    "accumulation drive long-run prosperity. International trade theory shows that comparative "
    "advantage — not absolute advantage — determines trade patterns and mutual gains from exchange. "
    "Behavioral economics incorporates psychological evidence that humans systematically deviate "
    "from the rational-agent model through heuristics and biases, loss aversion, and hyperbolic "
    "discounting. Financial economics studies asset pricing, risk, and the role of intermediaries "
    "in allocating capital across the economy. "
    # 8. Chemistry (~250 words)
    "Chemistry is the science of matter at the atomic and molecular level. The periodic table organizes "
    "elements by atomic number and reveals periodic trends: atomic radius, electronegativity, ionization "
    "energy, and electron affinity. Chemical bonding arises from the electrostatic interaction between "
    "electrons and nuclei. Covalent bonds share electron pairs; ionic bonds transfer electrons, creating "
    "oppositely charged ions; metallic bonds delocalize electrons over a lattice. Molecular orbital "
    "theory gives a quantum-mechanical description of bonding, predicting bond orders, magnetic properties, "
    "and spectra. Thermodynamics determines reaction spontaneity through free energy: the Gibbs energy "
    "combines enthalpy and entropy changes. Le Chatelier's principle predicts how equilibria shift in "
    "response to perturbations. Reaction kinetics depends on activation energy barriers; transition "
    "state theory and Arrhenius equation connect temperature to rate constants. Catalysts lower "
    "activation energies without being consumed. Organic chemistry studies the enormous diversity of "
    "carbon compounds: functional groups including alcohols, aldehydes, ketones, carboxylic acids, "
    "amines, and esters define reactivity. Stereochemistry describes the spatial arrangement of atoms; "
    "chirality is crucial in pharmaceuticals since enantiomers can have dramatically different "
    "biological activities. Biochemistry bridges chemistry and biology: enzymes catalyze metabolic "
    "reactions with exquisite specificity; ATP is the universal energy currency; nucleic acids encode "
    "and express genetic information. Analytical chemistry develops methods for determining composition "
    "and structure, including NMR spectroscopy, mass spectrometry, X-ray crystallography, and "
    "chromatography. "
    # 9. Geology (~250 words)
    "Geology studies the solid Earth, its composition, structure, and the processes that shape it over "
    "time. The Earth consists of an inner solid iron-nickel core, an outer liquid core, a semi-solid "
    "mantle, and a thin rigid crust. Plate tectonics, established in the mid-twentieth century, explains "
    "continental drift, seafloor spreading, mountain building, earthquakes, and volcanism as consequences "
    "of lithospheric plates moving over the asthenosphere driven by mantle convection. Convergent plate "
    "boundaries create subduction zones with deep trenches and volcanic arcs; divergent boundaries "
    "create mid-ocean ridges where new ocean floor forms; transform boundaries produce strike-slip faults. "
    "The San Andreas Fault marks the transform boundary between the Pacific and North American plates. "
    "The rock cycle describes how rocks transition between igneous, sedimentary, and metamorphic forms "
    "driven by heat, pressure, weathering, and erosion. Radiometric dating measures the decay of "
    "radioactive isotopes to determine the ages of rocks; uranium-lead dating places the age of Earth "
    "at 4.54 billion years. The stratigraphic column records the history of life through fossils, "
    "revealing five major mass extinctions. The Cretaceous-Paleogene extinction 66 million years ago, "
    "caused by a bolide impact and volcanism, eliminated the non-avian dinosaurs and three-quarters "
    "of species. Ice cores from Greenland and Antarctica preserve climate records extending hundreds "
    "of thousands of years, showing cycles of glaciation linked to Milankovitch cycles in Earth's "
    "orbital parameters. "
    # 10. Psychology (~250 words)
    "Psychology is the scientific study of behavior and mental processes. Sensation and perception "
    "convert physical stimuli into neural signals and organize them into coherent experiences. "
    "Psychophysics measures the relationship between stimulus intensity and perceived magnitude. "
    "Attention selects among competing stimuli; inattentional blindness shows how unattended stimuli "
    "can go completely unnoticed. Memory encodes information in sensory, short-term, and long-term "
    "stores. Working memory, with a capacity of roughly four items, maintains information for active "
    "processing. Long-term memory distinguishes declarative memory — episodic and semantic — from "
    "procedural memory. The hippocampus is critical for forming new declarative memories; its "
    "bilateral removal produced the famous patient H.M., who could no longer form new long-term "
    "memories. Cognitive psychology studies how mental representations guide behavior. The "
    "information-processing approach models the mind as a computational system. Cognitive biases — "
    "confirmation bias, availability heuristic, anchoring — lead to systematic errors in judgment. "
    "Social psychology examines how others influence our thoughts, feelings, and behavior. Asch's "
    "conformity experiments, Milgram's obedience studies, and Zimbardo's prison simulation revealed "
    "disturbing situational influences on behavior. Developmental psychology tracks change across "
    "the lifespan. Piaget's stages of cognitive development — sensorimotor, preoperational, concrete "
    "operational, formal operational — describe how children's thinking becomes more abstract. "
    "Attachment theory, developed by Bowlby, shows that early bonds with caregivers shape later "
    "relationships and emotional regulation. Clinical psychology diagnoses and treats mental "
    "disorders using evidence-based interventions including cognitive-behavioral therapy and "
    "pharmacotherapy. "
    # 11. Music (~250 words)
    "Music is the art of organizing sound and silence in time to create structures with emotional, "
    "aesthetic, or cultural meaning. Acoustic physics describes sound as longitudinal pressure waves; "
    "pitch corresponds to frequency, loudness to amplitude, and timbre to the spectral envelope "
    "of overtones. The harmonic series — the sequence of integer multiples of a fundamental frequency "
    "— underlies the intervals and consonances recognized across cultures. Western tonal music "
    "organizes pitches into scales. The major scale evokes stability and brightness; the natural "
    "minor scale evokes darker moods. Harmony arises from chords — simultaneous pitches — and "
    "their progression. Functional harmony assigns chords roles of tonic, subdominant, and dominant, "
    "creating tension and resolution. The dominant seventh chord creates strong pull toward the tonic, "
    "and this motion is the engine of classical tonality. Counterpoint is the art of combining "
    "independent melodic lines according to rules; Bach's fugues are its supreme achievement. "
    "Rhythm organizes music in time: meter groups beats into regular patterns; syncopation displaces "
    "accents, creating propulsive energy. Twentieth-century Western art music challenged tonal "
    "conventions: Schoenberg's twelve-tone serialism organized all chromatic pitches into rows; "
    "Cage explored indeterminacy and silence. Jazz emerged from African American musical traditions "
    "and developed improvisation, syncopation, swing, and harmonic sophistication. Rock, pop, "
    "hip-hop, and electronic music are the dominant popular forms of the late twentieth and "
    "early twenty-first centuries, shaped by recording technology and global distribution. "
    "Ethnomusicology studies music across cultures, revealing universal features alongside "
    "enormous diversity. "
    # 12. Art and Art History (~250 words)
    "Visual art encompasses painting, sculpture, drawing, printmaking, photography, film, and "
    "digital media. Prehistoric cave paintings at Lascaux and Altamira, dating to over 17,000 years "
    "ago, testify to the antiquity of human image-making. Ancient Egyptian art followed rigid "
    "conventions of scale and perspective tied to hierarchical social order. Greek sculptors "
    "developed idealized naturalism, culminating in the classical period of the fifth century BC. "
    "Roman art adopted Greek models and applied them to portraiture, narrative relief, and "
    "monumental architecture. Medieval art subordinated naturalism to spiritual symbolism; "
    "Gothic cathedrals integrated architecture, sculpture, stained glass, and manuscript "
    "illumination into a unified devotional program. The Italian Renaissance rediscovered linear "
    "perspective, anatomy, and classical proportion. Leonardo, Raphael, and Michelangelo pushed "
    "the High Renaissance ideal to its peak. Mannerism, the Baroque, and Rococo followed in "
    "succession, each reacting against or extending what came before. Romanticism in the early "
    "nineteenth century emphasized emotion, nature, and the sublime. Impressionism broke with "
    "academic convention, capturing transient effects of light in broken brushwork. Post-impressionism, "
    "Cubism, Expressionism, Surrealism, and Abstract Expressionism chart the radical experiments "
    "of modernism. Pop art engaged with mass culture and consumer imagery. Conceptual art shifted "
    "emphasis from the object to the idea. Contemporary art resists definition by medium or style, "
    "engaging with globalization, identity, politics, and technology. "
    # 13. Medicine (~250 words)
    "Medicine encompasses the science and practice of diagnosing, treating, and preventing disease. "
    "The germ theory, established by Pasteur and Koch in the nineteenth century, showed that specific "
    "microorganisms cause specific infectious diseases. Antiseptic and aseptic techniques by Semmelweis "
    "and Lister dramatically reduced surgical mortality. Vaccination, pioneered by Jenner, harnesses "
    "the immune system to prevent disease; the eradication of smallpox is the greatest achievement "
    "of public health. Antibiotics transformed bacterial infections from leading killers to treatable "
    "conditions, but antibiotic resistance now threatens to reverse these gains. Molecular medicine "
    "uses understanding of genes and proteins to diagnose and treat disease at the molecular level. "
    "The Human Genome Project, completed in 2003, produced a reference sequence for the entire human "
    "genome, accelerating the identification of disease genes and drug targets. Monoclonal antibodies "
    "engineered to recognize specific targets are now important drugs for cancer, autoimmune diseases, "
    "and other conditions. CRISPR gene therapy offers the prospect of correcting genetic defects "
    "directly. Medical imaging — X-ray, CT, MRI, PET, ultrasound — allows non-invasive visualization "
    "of anatomy and function. Evidence-based medicine synthesizes clinical trial data through "
    "systematic reviews and meta-analyses to guide treatment. Epidemiology studies disease at the "
    "population level, identifying risk factors and evaluating interventions. The COVID-19 pandemic "
    "demonstrated both the power of rapid mRNA vaccine development and the challenges of global "
    "health coordination. "
    # 14. Engineering (~250 words)
    "Engineering applies scientific and mathematical principles to design and build systems that "
    "solve human problems. Civil engineering designs infrastructure — roads, bridges, dams, buildings, "
    "water supply, and sanitation systems. The Romans' mastery of the arch and concrete enabled "
    "aqueducts, the Pantheon, and roads that lasted millennia. Modern structural engineering uses "
    "steel, reinforced concrete, and computational analysis to build skyscrapers and long-span bridges. "
    "Mechanical engineering designs machines, engines, and thermal systems. The steam engine and "
    "internal combustion engine transformed transportation and power generation. Thermodynamic "
    "efficiency limits govern all heat engines through the Carnot cycle. Electrical engineering "
    "develops systems that generate, transmit, and use electrical energy, as well as electronics, "
    "telecommunications, and signal processing. Maxwell's equations govern electromagnetism; "
    "semiconductor physics underlies transistors and integrated circuits. Moore's law observed "
    "that transistor density doubles approximately every two years, driving exponential growth "
    "in computing power for decades. Chemical engineering designs processes for transforming raw "
    "materials into useful products through reaction, separation, and heat transfer operations. "
    "Aerospace engineering develops aircraft and spacecraft; thermodynamics, fluid mechanics, "
    "and control theory are its foundations. Biomedical engineering applies engineering methods "
    "to medicine: prosthetics, medical devices, tissue engineering, and diagnostic equipment. "
    "Systems engineering manages the complexity of large sociotechnical systems. Software "
    "engineering develops methods for creating reliable, maintainable, and scalable software. "
    # 15. Law (~250 words)
    "Law is the system of rules and institutions through which societies organize collective life "
    "and resolve disputes. Common law systems, used in England and its former colonies, develop "
    "through judicial precedent — the doctrine of stare decisis. Civil law systems, derived from "
    "Roman law and used across continental Europe and much of the world, rely primarily on "
    "codified statutes. Constitutional law establishes the basic structure of government and "
    "fundamental rights. Judicial review — the power of courts to invalidate legislation as "
    "unconstitutional — is a distinctive feature of the American system, established in Marbury "
    "v. Madison. Criminal law defines offenses against the state and prescribes punishments; "
    "it requires proof beyond reasonable doubt. Civil law resolves disputes between private "
    "parties through remedies like damages and injunctions, requiring only a preponderance of "
    "evidence. Contract law enforces agreements by holding parties to their promises. Tort law "
    "provides remedies for civil wrongs, including negligence and intentional harms. Property "
    "law governs rights over physical and intellectual objects. International law regulates "
    "relations between states through treaties, customary international law, and institutions "
    "like the United Nations and the International Court of Justice. Human rights law enshrines "
    "fundamental protections — life, liberty, freedom from torture, fair trial — as universal "
    "norms. Administrative law governs the exercise of governmental power by regulatory agencies. "
    "Legal theory asks what law is and what makes it legitimate, with positivism and natural law "
    "theory as the main competing accounts. "
    # 16. Agriculture (~250 words)
    "Agriculture is the practice of cultivating plants and raising animals for food, fiber, and "
    "other products. The Neolithic agricultural revolution, which began independently in several "
    "regions about 10,000 years ago, transformed human societies from mobile foragers to settled "
    "communities. The domestication of wheat, rice, maize, and other staple crops, and of cattle, "
    "pigs, sheep, and horses, fundamentally changed human ecology. Plant breeding over millennia "
    "selected for yield, disease resistance, and palatability; the Green Revolution of the 1960s "
    "introduced high-yield dwarf wheat and rice varieties that dramatically increased food "
    "production and averted famines. Synthetic nitrogen fertilizers, derived from the Haber-Bosch "
    "process, doubled the nitrogen available to crops and are estimated to feed half the current "
    "world population. Pesticides control insects, diseases, and weeds but can harm non-target "
    "organisms and accumulate in food chains. Irrigation supplies water to arid regions but can "
    "cause salinization and groundwater depletion. Modern intensive agriculture achieves high "
    "yields but at significant environmental cost: soil erosion, nutrient runoff, greenhouse "
    "gas emissions, and loss of biodiversity. Agroecology seeks to apply ecological principles "
    "to farming systems, maintaining productivity while reducing environmental impact. Genetic "
    "engineering of crops allows introduction of specific traits — herbicide tolerance, insect "
    "resistance, nutritional enhancement — more precisely than traditional breeding. Food security "
    "remains a critical global challenge: despite sufficient total production, hundreds of millions "
    "suffer from hunger due to poverty, distribution failures, and conflict. "
    # 17. Sports (~250 words)
    "Sport encompasses all forms of physical competition governed by rules, practiced individually "
    "or in teams. Ancient Greek athletics included the Olympic Games, first recorded in 776 BC, "
    "featuring foot races, wrestling, discus, javelin, and the pentathlon. The modern Olympics, "
    "revived by Pierre de Coubertin in 1896, grew to encompass hundreds of events across summer "
    "and winter editions. Sport involves complex biomechanics: sprint performance depends on "
    "stride length, stride frequency, and force application; swimming efficiency requires minimal "
    "drag and maximal propulsive force. Exercise physiology studies how the body responds to "
    "physical training. Aerobic capacity, measured as maximal oxygen uptake, limits endurance "
    "performance; anaerobic capacity determines explosive power output. Resistance training "
    "increases muscle mass and strength through progressive overload, inducing myofibrillar "
    "hypertrophy. Sports psychology addresses the mental aspects of performance: concentration, "
    "confidence, arousal regulation, and the psychology of teams. Team sports require coordination, "
    "communication, and role differentiation; effective teams develop shared mental models and "
    "collective efficacy. Technology transforms sport: equipment advances in cycling, swimming, "
    "and athletics have pushed performance records. Video analysis and wearable sensors enable "
    "detailed biomechanical assessment. Nutrition science guides athlete fueling strategies: "
    "carbohydrate loading before endurance events, protein timing for muscle adaptation, and "
    "hydration for thermoregulation. Anti-doping programs police the use of performance-enhancing "
    "drugs; the WADA code provides the international framework. Sport connects to culture, "
    "identity, commerce, and geopolitics in complex ways. "
    # 18. Literature (~250 words)
    "Literature encompasses written works valued for their aesthetic and intellectual qualities. "
    "The oral epic tradition — Homer's Iliad and Odyssey — preceded writing and shaped Greek "
    "identity and values. Classical Greek drama, divided into tragedy and comedy, explored fate, "
    "hubris, civic duty, and the human condition. Roman literature adapted Greek models; Virgil's "
    "Aeneid legitimized Roman imperial identity by connecting Rome to the fall of Troy. Medieval "
    "literature includes courtly romance, the Divine Comedy of Dante, and the Canterbury Tales "
    "of Chaucer. The novel, emerging in Europe in the seventeenth and eighteenth centuries, "
    "became the dominant literary form of modernity. Cervantes' Don Quixote is often called the "
    "first modern novel. The realist novel of the nineteenth century — Balzac, Flaubert, Tolstoy, "
    "Dostoevsky, Dickens, George Eliot — depicted society and individual psychology with "
    "unprecedented detail. Modernism in literature, contemporaneous with modernism in art, "
    "abandoned linear narrative and omniscient narration in favor of stream of consciousness, "
    "fragmentation, and unreliable narrators. Joyce's Ulysses, Woolf's Mrs Dalloway, and "
    "Proust's In Search of Lost Time are canonical modernist works. Postmodern literature plays "
    "with genre conventions, self-reference, and the instability of meaning. World literature "
    "now includes major works from Africa, Latin America, South and East Asia, addressing "
    "colonialism, identity, and modernity from non-Western perspectives. Literary theory and "
    "criticism examine how texts produce meaning through close reading, structuralism, "
    "deconstruction, psychoanalytic, feminist, and postcolonial approaches. "
    # 19. Astronomy (~250 words)
    "Astronomy studies celestial objects and phenomena beyond Earth's atmosphere. The cosmic "
    "distance ladder uses parallax for nearby stars, Cepheid variable stars for nearby galaxies, "
    "and Type Ia supernovae as standard candles for distant galaxies to measure distances across "
    "the universe. The Big Bang model describes the origin of the universe 13.8 billion years "
    "ago as an extremely hot dense state that expanded and cooled. Cosmic microwave background "
    "radiation, the afterglow of the Big Bang, provides the most detailed map of the early "
    "universe. Inflation theory posits an exponential expansion in the first fraction of a second "
    "that explains the flatness, horizon, and monopole problems. Stars form in dense molecular "
    "clouds when gravity overcomes thermal pressure. Nuclear fusion in stellar cores converts "
    "hydrogen to helium, releasing energy. More massive stars fuse heavier elements up to iron, "
    "then explode as supernovae, seeding the interstellar medium with heavy elements. Neutron "
    "stars and black holes are the remnants of massive stellar deaths. Binary neutron star mergers "
    "produce gravitational wave signals and r-process nucleosynthesis, creating gold and other "
    "heavy elements. The Milky Way contains several hundred billion stars arranged in a "
    "spiral disk and central bulge with a supermassive black hole, Sagittarius A*, at its center. "
    "Galaxies cluster into groups, clusters, and superclusters connected by the cosmic web of "
    "filaments and voids. Dark energy, causing the universe's expansion to accelerate, represents "
    "the greatest mystery in modern cosmology. "
    # 20. Environmental Science (~250 words)
    "Environmental science integrates biology, chemistry, physics, geology, and social science "
    "to study the natural world and the impact of human activity on it. The carbon cycle transfers "
    "carbon between the atmosphere, biosphere, oceans, and lithosphere. Fossil fuel combustion "
    "has increased atmospheric CO2 from 280 ppm in 1750 to over 420 ppm today, enhancing the "
    "greenhouse effect and warming the climate. The greenhouse effect arises because CO2, methane, "
    "and water vapor absorb outgoing infrared radiation and re-emit it in all directions, including "
    "back toward Earth's surface. Climate models project continued warming, sea level rise, "
    "intensified extreme weather, altered precipitation patterns, and ecosystem disruption under "
    "continued emissions. The Paris Agreement aims to limit warming to 1.5-2 degrees Celsius "
    "above pre-industrial levels through nationally determined contributions to emissions reduction. "
    "Deforestation destroys carbon sinks and reduces biodiversity; tropical forests store enormous "
    "quantities of carbon and host the majority of terrestrial species. Ocean acidification, caused "
    "by absorption of CO2, threatens marine calcifiers including corals and shellfish. The nitrogen "
    "cycle has been substantially altered by synthetic fertilizers, causing eutrophication of "
    "freshwater and coastal ecosystems. Biodiversity loss undermines ecosystem services including "
    "pollination, pest control, water purification, and soil formation. Renewable energy from "
    "solar, wind, and hydropower can replace fossil fuels; costs have fallen dramatically. "
    "Circular economy principles aim to reduce waste by designing products for reuse, repair, "
    "and recycling. "
)


# ======================================================================
# MAIN FUNCTION: Both tasks on single A100
# ======================================================================
@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    secrets=[HF_SECRET],
    memory=65536,
)
def run_16k_and_latency():
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
    print("NEXUSQUANT — A100 16K + Latency Experiment")
    print("=" * 80)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

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
    load_time = time.time() - t0
    print(f"Loaded in {load_time:.1f}s")

    n_layers   = model.config.num_hidden_layers           # 32
    n_kv_heads = model.config.num_key_value_heads         # 8
    head_dim   = model.config.hidden_size // model.config.num_attention_heads  # 128
    rope_base  = getattr(model.config, 'rope_theta', 10000.0)
    print(f"Config: {n_layers}L, {n_kv_heads}KVH, d={head_dim}, rope_theta={rope_base}")

    sliding_window = 32

    # ------------------------------------------------------------------ importance scorer
    def score_importance(kv_cache, prefix_len):
        obs_window = max(32, prefix_len // 16)
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
            causal  = (all_pos <= obs_pos)
            scores  = scores.masked_fill(~causal.unsqueeze(0), float('-inf'))
            attn    = F.softmax(scores, dim=-1, dtype=torch.float32)
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
                    rotated   = kept_data @ H.T
                    amax      = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                    sc        = amax / (levels / 2)
                    normalized = rotated / sc
                    groups    = normalized.reshape(-1, 8)
                    lp        = E8Lattice.nearest_point(groups).clamp(-levels / 2, levels / 2)
                    coords    = lp.reshape(-1, head_dim)
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

    # ======================================================================
    # TOKENIZE
    # ======================================================================
    print("\nTokenizing 16K text (targeting 16384 tokens)...")
    inputs = tok(TEXT_16K, return_tensors="pt", max_length=16384, truncation=True)
    full_ids = inputs.input_ids.to("cuda")
    n_tok = full_ids.shape[1]

    target_prefix = 8192
    if n_tok < 6000:
        print(f"WARNING: Only {n_tok} tokens — need at least 6K. Using full text as-is.")
        prefix_len = n_tok // 2
    elif n_tok >= target_prefix * 2:
        prefix_len = target_prefix
    else:
        prefix_len = n_tok // 2

    cont_len = n_tok - prefix_len
    print(f"Text: total={n_tok} tokens, prefix={prefix_len}, continuation={cont_len}")
    print(f"GPU free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1e9:.1f} GB")

    # ======================================================================
    # TASK A: 16K QUALITY SWEEP
    # ======================================================================
    print("\n" + "=" * 80)
    print("TASK A: 16K Context Quality Sweep")
    print("=" * 80)

    # Baseline PPL
    with torch.no_grad():
        pout = model(full_ids[:, :prefix_len], use_cache=True)
        cout = model(full_ids[:, prefix_len:], past_key_values=pout.past_key_values, use_cache=True)
        logits  = cout.logits[:, :-1, :].float()
        targets = full_ids[:, prefix_len + 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        baseline_ppl = torch.exp(loss).item()
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    # Compute importance once
    with torch.no_grad():
        pout = model(full_ids[:, :prefix_len], use_cache=True)
    importance = score_importance(pout.past_key_values, prefix_len)

    # Quality sweep
    quality_configs = [
        {"name": "2b no-evict", "evict_pct": 0},
        {"name": "2b+35%evict", "evict_pct": 35},
        {"name": "2b+50%evict", "evict_pct": 50},
        {"name": "2b+60%evict", "evict_pct": 60},
        {"name": "2b+70%evict", "evict_pct": 70},
        {"name": "2b+80%evict", "evict_pct": 80},
    ]

    print(f"\n{'Config':<20s} {'PPL':>8s} {'Delta%':>9s} {'Ratio':>7s} {'Kept':>6s}")
    print("-" * 56)
    quality_results = []

    for cfg in quality_configs:
        torch.cuda.empty_cache()
        evict_pct = cfg["evict_pct"]

        with torch.no_grad():
            pout = model(full_ids[:, :prefix_len], use_cache=True)
            kv   = pout.past_key_values

        keep_mask = build_keep_mask(prefix_len, evict_pct, importance)
        info, kv  = evict_quantize(kv, keep_mask, key_bits=2, val_bits=2, prefix_len=prefix_len)

        evict_mask = ~keep_mask
        attn_ctx   = torch.ones(prefix_len, dtype=torch.long, device="cuda")
        attn_ctx[evict_mask] = 0
        attn_full  = torch.cat([attn_ctx, torch.ones(cont_len, dtype=torch.long, device="cuda")])

        with torch.no_grad():
            cout = model(full_ids[:, prefix_len:], past_key_values=kv,
                         attention_mask=attn_full.unsqueeze(0), use_cache=True)
            logits  = cout.logits[:, :-1, :].float()
            targets = full_ids[:, prefix_len + 1:]
            loss    = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            ppl     = torch.exp(loss).item()

        delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
        print(f"{cfg['name']:<20s} {ppl:8.4f} {delta:+8.2f}% {info['ratio']:6.2f}x {info['n_kept']:5d}")
        quality_results.append({
            "name": cfg["name"],
            "evict_pct": evict_pct,
            "ppl": ppl,
            "baseline_ppl": baseline_ppl,
            "delta": delta,
            "ratio": info["ratio"],
            "n_kept": info["n_kept"],
        })

    print(f"\nPeak GPU memory after Task A: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # ======================================================================
    # TASK B: LATENCY MEASUREMENT
    # ======================================================================
    print("\n" + "=" * 80)
    print("TASK B: Latency Measurement (3 iterations each, wall-clock)")
    print("=" * 80)
    print(f"Prefix: {prefix_len} tokens | Generation: 100 tokens")

    GEN_TOKENS = 100
    N_ITERS    = 3

    def timed_sync():
        torch.cuda.synchronize()
        return time.time()

    latency_configs = [
        {"name": "Baseline",   "evict_pct": 0,  "compress": False},
        {"name": "35% evict",  "evict_pct": 35, "compress": True},
        {"name": "60% evict",  "evict_pct": 60, "compress": True},
    ]

    latency_results = []

    for cfg in latency_configs:
        evict_pct   = cfg["evict_pct"]
        do_compress = cfg["compress"]
        name        = cfg["name"]
        print(f"\n--- {name} ---")

        iter_prefill  = []
        iter_compress = []
        iter_generate = []

        for it in range(N_ITERS):
            torch.cuda.empty_cache()

            # --- PREFILL ---
            t_prefill_start = timed_sync()
            with torch.no_grad():
                pout = model(full_ids[:, :prefix_len], use_cache=True)
                kv   = pout.past_key_values
            t_prefill_end = timed_sync()
            prefill_ms = (t_prefill_end - t_prefill_start) * 1000

            # --- COMPRESSION (eviction + quantization) ---
            if do_compress:
                keep_mask = build_keep_mask(prefix_len, evict_pct, importance)
                t_compress_start = timed_sync()
                info, kv = evict_quantize(kv, keep_mask, key_bits=2, val_bits=2,
                                          prefix_len=prefix_len)
                t_compress_end = timed_sync()
                compress_ms = (t_compress_end - t_compress_start) * 1000

                evict_mask = ~keep_mask
                attn_ctx   = torch.ones(prefix_len, dtype=torch.long, device="cuda")
                attn_ctx[evict_mask] = 0
                # attn mask for generation: will use past_key_values, so we only need
                # the attention mask for the prefix portion (n_kept tokens active)
                past_attn = attn_ctx.unsqueeze(0)  # [1, prefix_len]
            else:
                compress_ms = 0.0
                past_attn   = torch.ones(1, prefix_len, dtype=torch.long, device="cuda")

            # --- GENERATION (100 tokens) ---
            gen_input_ids = full_ids[:, prefix_len:prefix_len + 1]  # seed first continuation token
            current_kv    = kv
            current_attn  = past_attn

            t_gen_start = timed_sync()
            with torch.no_grad():
                for _ in range(GEN_TOKENS):
                    step_out   = model(gen_input_ids, past_key_values=current_kv,
                                       attention_mask=torch.cat([current_attn,
                                           torch.ones(1, 1, dtype=torch.long, device="cuda")], dim=1),
                                       use_cache=True)
                    next_token = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    current_kv = step_out.past_key_values
                    current_attn = torch.cat([current_attn,
                                              torch.ones(1, 1, dtype=torch.long, device="cuda")], dim=1)
                    gen_input_ids = next_token
            t_gen_end = timed_sync()
            generate_ms = (t_gen_end - t_gen_start) * 1000

            iter_prefill.append(prefill_ms)
            iter_compress.append(compress_ms)
            iter_generate.append(generate_ms)
            print(f"  iter {it+1}: prefill={prefill_ms:.1f}ms  compress={compress_ms:.1f}ms  "
                  f"generate={generate_ms:.1f}ms  total={prefill_ms+compress_ms+generate_ms:.1f}ms")

        mean_prefill  = float(np.mean(iter_prefill))
        mean_compress = float(np.mean(iter_compress))
        mean_generate = float(np.mean(iter_generate))
        mean_total    = mean_prefill + mean_compress + mean_generate

        latency_results.append({
            "name":       name,
            "evict_pct":  evict_pct,
            "prefill_ms":  mean_prefill,
            "compress_ms": mean_compress,
            "generate_ms": mean_generate,
            "total_ms":    mean_total,
        })

    # Compute speedups relative to baseline
    baseline_total = latency_results[0]["total_ms"]
    for r in latency_results:
        r["speedup"] = baseline_total / r["total_ms"]

    # ======================================================================
    # PRINT SUMMARY
    # ======================================================================
    print("\n" + "=" * 80)
    print("=== 16K CONTEXT ===")
    print("=" * 80)
    print(f"Model: Mistral-7B-v0.1 | GPU: A100 | prefix={prefix_len} | total_tokens={n_tok}")
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"\n{'Evict%':<8s} {'PPL Δ%':>9s} {'Ratio':>7s} {'Kept':>6s}")
    print("-" * 36)
    for r in quality_results:
        ep_str = f"{r['evict_pct']}%"
        print(f"{ep_str:<8s} {r['delta']:>+8.2f}% {r['ratio']:6.2f}x {r['n_kept']:5d}")

    print("\n" + "=" * 80)
    print("=== LATENCY (ms) ===")
    print("=" * 80)
    print(f"{'Config':<14s} {'Prefill':>10s} {'Compress':>10s} {'Generate':>10s} "
          f"{'Total':>10s} {'Speedup':>8s}")
    print("-" * 68)
    for r in latency_results:
        print(f"{r['name']:<14s} {r['prefill_ms']:>9.1f}ms {r['compress_ms']:>9.1f}ms "
              f"{r['generate_ms']:>9.1f}ms {r['total_ms']:>9.1f}ms {r['speedup']:>7.2f}x")

    return quality_results, latency_results, n_tok, prefix_len, baseline_ppl


# ======================================================================
# LOCAL ENTRYPOINT
# ======================================================================
@app.local_entrypoint()
def main():
    import time

    print("\n" + "=" * 80)
    print("NEXUSQUANT: A100 16K Context + Latency")
    print("=" * 80)
    print("Launching on A100 GPU...")

    result = run_16k_and_latency.remote()
    quality_results, latency_results, n_tok, prefix_len, baseline_ppl = result

    # ------------------------------------------------------------------
    # Compute speedups for the write step too
    baseline_total = latency_results[0]["total_ms"]
    for r in latency_results:
        r["speedup"] = baseline_total / r["total_ms"]

    # ------------------------------------------------------------------
    # Print final summary
    print("\n" + "=" * 80)
    print("=== 16K CONTEXT ===")
    print(f"Model: Mistral-7B-v0.1 | GPU: A100 | prefix={prefix_len} | total_tokens={n_tok}")
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"\n{'Evict%':<8s} {'PPL Δ%':>9s} {'Ratio':>7s} {'Kept':>6s}")
    print("-" * 36)
    for r in quality_results:
        ep_str = f"{r['evict_pct']}%"
        print(f"{ep_str:<8s} {r['delta']:>+8.2f}% {r['ratio']:6.2f}x {r['n_kept']:5d}")

    print("\n=== LATENCY (ms) ===")
    print(f"{'Config':<14s} {'Prefill':>10s} {'Compress':>10s} {'Generate':>10s} "
          f"{'Total':>10s} {'Speedup':>8s}")
    print("-" * 68)
    for r in latency_results:
        print(f"{r['name']:<14s} {r['prefill_ms']:>9.1f}ms {r['compress_ms']:>9.1f}ms "
              f"{r['generate_ms']:>9.1f}ms {r['total_ms']:>9.1f}ms {r['speedup']:>7.2f}x")

    # ------------------------------------------------------------------
    # Write results file
    out_dir  = os.path.join(os.path.dirname(__file__), "..", ".company", "engineering")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "16k_latency_results.md")

    with open(out_path, "w") as f:
        f.write("# A100 16K Context + Latency — NexusQuant\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write("**GPU:** A100 (40GB)\n")
        f.write("**Model:** mistralai/Mistral-7B-v0.1 (32L, 8 KV heads, d=128, rope_theta=10000)\n")
        f.write("**Pipeline:** RoPE removal → Hadamard → 2-bit E8 VQ → temporal delta → zstd-22\n")
        f.write("**Eviction scorer:** per-layer key-key attention, obs_window=max(32, prefix//16)\n\n")
        f.write(f"**Total tokens:** {n_tok} | **Prefix:** {prefix_len} | "
                f"**Continuation:** {n_tok - prefix_len}\n\n")

        # --- Task A ---
        f.write("---\n\n")
        f.write("## Task A: 16K Context Quality Sweep\n\n")
        f.write(f"**Baseline PPL:** {baseline_ppl:.4f}\n\n")
        f.write("| Config | PPL | PPL Δ% | Ratio | Kept tokens |\n")
        f.write("|--------|-----|--------|-------|-------------|\n")
        for r in quality_results:
            f.write(f"| {r['name']} | {r['ppl']:.4f} | {r['delta']:+.3f}% | "
                    f"{r['ratio']:.2f}x | {r['n_kept']} |\n")

        # Key finding
        f.write("\n**Key findings:**\n")
        sixty = next((r for r in quality_results if r['evict_pct'] == 60), None)
        eighty = next((r for r in quality_results if r['evict_pct'] == 80), None)
        thirty5 = next((r for r in quality_results if r['evict_pct'] == 35), None)
        if sixty:
            verdict = "PERSISTS" if sixty['delta'] > 10 else "DOES NOT PERSIST"
            f.write(f"- 60% eviction catastrophe from 3K prefix: **{verdict}** "
                    f"(Δ%={sixty['delta']:+.2f}% at {prefix_len}-tok prefix)\n")
        if thirty5:
            f.write(f"- 35% eviction: Δ%={thirty5['delta']:+.2f}%, ratio={thirty5['ratio']:.2f}x\n")
        if eighty:
            f.write(f"- 80% eviction: Δ%={eighty['delta']:+.2f}%, ratio={eighty['ratio']:.2f}x\n")

        # Compare to prior reference
        f.write("\n**Prior reference (3K prefix, from playbook):**\n")
        f.write("| Evict% | 1664-tok Δ% | 2924-tok Δ% | 8K-tok Δ% (this run) |\n")
        f.write("|--------|-------------|-------------|----------------------|\n")
        prior_1664 = {35: +0.14, 60: +0.82, 80: +2.13}
        prior_2924 = {60: "catastrophic (+42%)"}
        for r in quality_results:
            if r['evict_pct'] == 0:
                continue
            ep = r['evict_pct']
            p1 = f"{prior_1664[ep]:+.2f}%" if ep in prior_1664 else "N/A"
            p2 = prior_2924.get(ep, "N/A")
            f.write(f"| {ep}% | {p1} | {p2} | {r['delta']:+.3f}% |\n")

        # --- Task B ---
        f.write("\n---\n\n")
        f.write("## Task B: Latency Measurement\n\n")
        f.write(f"**Prefix:** {prefix_len} tokens | **Generation:** 100 tokens | "
                f"**Iterations:** 3 (mean reported)\n\n")
        f.write("| Config | Prefill (ms) | Compress (ms) | Generate (ms) | Total (ms) | Speedup |\n")
        f.write("|--------|-------------|---------------|---------------|------------|--------|\n")
        for r in latency_results:
            f.write(f"| {r['name']} | {r['prefill_ms']:.1f} | {r['compress_ms']:.1f} | "
                    f"{r['generate_ms']:.1f} | {r['total_ms']:.1f} | {r['speedup']:.2f}x |\n")

        f.write("\n**Latency findings:**\n")
        for r in latency_results[1:]:
            f.write(f"- {r['name']}: {r['speedup']:.2f}x total speedup vs baseline "
                    f"({r['compress_ms']:.0f}ms compression overhead, "
                    f"{r['generate_ms']:.0f}ms generation)\n")

    print(f"\nResults written to: {out_path}")
    print("Done.")
