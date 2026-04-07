"""A100 experiment: 3 quality improvement approaches on Mistral-7B.

Baseline (key-key scorer, uniform eviction, zero+mask) vs:
  1. Real attention weights (eager mode hooks, softmax attn importance)
  2. Weighted scorer (key-key with layer weight 1x→2x, later layers count more)
  3. Physical truncation (key-key scorer, physically remove evicted KV tokens)

Eviction rates tested: 35% / 60% / 80%
Text: same 20-topic multi-paragraph corpus as modal_16k_latency.py
Prefix: ~3544 tokens (short-prefix regime matching the baseline numbers)

Results written to .company/engineering/quality_improve_results.md
"""
import modal
import os

app = modal.App("nexusquant-quality-improve")

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
# 20-TOPIC TEXT CORPUS (same as modal_16k_latency.py)
# ======================================================================
TEXT_16K = (
    # 1. Physics
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
    # 2. History
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
    # 3. Biology
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
    # 4. Computer Science
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
    # 5. Mathematics
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
    # 6. Philosophy
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
    # 7. Economics
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
    # 8. Chemistry
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
    # 9. Geology
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
    # 10. Psychology
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
    # 11. Music
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
    # 12. Art and Art History
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
    # 13. Medicine
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
    # 14. Engineering
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
    # 15. Law
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
    # 16. Agriculture
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
    # 17. Sports
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
    # 18. Literature
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
    # 19. Astronomy
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
    # 20. Environmental Science
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
# MAIN FUNCTION: all 4 approaches on single A100
# ======================================================================
@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
    secrets=[HF_SECRET],
    memory=65536,
)
def run_quality_improve():
    import sys, os
    sys.path.insert(0, "/root")
    # Reduce fragmentation — helps with the eager attention (seq, seq) allocations
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import math
    import numpy as np
    import torch
    import torch.nn.functional as F
    import zstandard

    from nexusquant.core.e8_lattice import E8Lattice
    from nexusquant.core.hadamard import hadamard_matrix
    from nexusquant.core.rope_utils import inverse_rope, forward_rope

    print("=" * 80)
    print("NEXUSQUANT — Quality Improvement Experiment (A100)")
    print("=" * 80)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ------------------------------------------------------------------
    # Config
    EVICT_RATES = [35, 60, 80]
    SLIDING_WINDOW = 32
    TARGET_PREFIX = 3544   # match baseline regime

    # ------------------------------------------------------------------
    # KV cache accessors
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

    # ------------------------------------------------------------------
    # Model loader (sdpa/default — flash-attn not in image)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os

    model_id = "mistralai/Mistral-7B-v0.1"

    def load_model(attn_impl="sdpa"):
        print(f"\nLoading {model_id} (attn_implementation={attn_impl})...")
        tok_ = AutoTokenizer.from_pretrained(model_id, token=os.environ["HF_TOKEN"])
        kwargs = dict(
            torch_dtype=torch.float16,
            device_map="auto",
            token=os.environ["HF_TOKEN"],
        )
        if attn_impl != "sdpa":
            kwargs["attn_implementation"] = attn_impl
        m_ = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        m_.eval()
        return tok_, m_

    # Load ONE model in eager mode — used for all approaches.
    # Key-key scorer and weighted scorer work on any KV cache (eager or sdpa).
    # Loading only one model avoids OOM on 40GB A100.
    tok, model_eager = load_model("eager")

    n_layers   = model_eager.config.num_hidden_layers           # 32
    n_kv_heads = model_eager.config.num_key_value_heads         # 8
    head_dim   = model_eager.config.hidden_size // model_eager.config.num_attention_heads  # 128
    rope_base  = getattr(model_eager.config, 'rope_theta', 10000.0)
    print(f"Config: {n_layers}L, {n_kv_heads}KVH, d={head_dim}, rope_theta={rope_base}")

    # ------------------------------------------------------------------
    # Tokenize — target ~3544 token prefix
    print(f"\nTokenizing (target prefix={TARGET_PREFIX})...")
    inputs   = tok(TEXT_16K, return_tensors="pt", max_length=16384, truncation=True)
    full_ids = inputs.input_ids.to("cuda")
    n_tok    = full_ids.shape[1]
    prefix_len = min(TARGET_PREFIX, n_tok // 2)
    cont_len   = n_tok - prefix_len
    print(f"Total tokens: {n_tok}, prefix: {prefix_len}, continuation: {cont_len}")

    # ------------------------------------------------------------------
    # Baseline PPL (no compression)
    print("\nComputing baseline PPL...")
    with torch.no_grad():
        pout    = model_eager(full_ids[:, :prefix_len], use_cache=True)
        cout    = model_eager(full_ids[:, prefix_len:],
                              past_key_values=pout.past_key_values, use_cache=True)
        logits  = cout.logits[:, :-1, :].float()
        targets = full_ids[:, prefix_len + 1:]
        loss    = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        baseline_ppl = torch.exp(loss).item()
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    # ================================================================
    # SCORER 1: key-key attention (baseline scorer)
    # ================================================================
    def score_importance_keykee(kv_cache):
        """Per-layer key-key attention, uniform layer weight."""
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
            causal  = (all_pos <= obs_pos)
            scores  = scores.masked_fill(~causal.unsqueeze(0), float('-inf'))
            attn    = F.softmax(scores, dim=-1, dtype=torch.float32)
            layer_imp = attn.sum(dim=1).mean(dim=0)
            pool_kernel = 5
            if seq_len > pool_kernel:
                imp_1d = layer_imp.unsqueeze(0).unsqueeze(0)
                layer_imp = F.avg_pool1d(
                    imp_1d, kernel_size=pool_kernel,
                    padding=pool_kernel // 2, stride=1,
                ).squeeze()[:seq_len]
            all_imp += layer_imp
        return all_imp / n_layers

    # ================================================================
    # SCORER 2: layer-weighted key-key (later layers count more)
    # ================================================================
    def score_importance_weighted(kv_cache):
        """Key-key scorer with linear layer weight: layer 0 → 1.0x, layer 31 → 2.0x."""
        obs_window = max(32, prefix_len // 16)
        k0, _ = get_kv(kv_cache, 0)
        seq_len = k0.shape[2]
        w = min(obs_window, seq_len)
        all_imp   = torch.zeros(seq_len, device='cpu')
        total_wt  = 0.0
        for l in range(n_layers):
            layer_weight = 1.0 + float(l) / (n_layers - 1)  # 1.0 at l=0, 2.0 at l=31
            kl, _ = get_kv(kv_cache, l)
            k = kl[0].float().cpu()
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
                layer_imp = F.avg_pool1d(
                    imp_1d, kernel_size=pool_kernel,
                    padding=pool_kernel // 2, stride=1,
                ).squeeze()[:seq_len]
            all_imp  += layer_imp * layer_weight
            total_wt += layer_weight
        return all_imp / total_wt

    # ================================================================
    # EVICT + QUANTIZE: zero-and-mask style (baseline / weighted / eager)
    # ================================================================
    def evict_quantize_mask(kv_cache, keep_mask):
        """Quantize kept tokens, zero evicted, return compression info."""
        H = hadamard_matrix(head_dim).cpu()
        n_kept     = keep_mask.sum().item()
        total_fp16 = 0
        all_coords = []
        cctx       = zstandard.ZstdCompressor(level=22)
        key_bits   = 2
        val_bits   = 2

        for l in range(n_layers):
            kl, vl = get_kv(kv_cache, l)
            k = kl.float().cpu()
            v = vl.float().cpu()
            total_fp16 += k.numel() * 2 + v.numel() * 2

            for is_key, tensor, bits in [
                (True,  k, key_bits),
                (False, v, val_bits),
            ]:
                levels = 2 ** bits
                t = tensor[0].clone()
                for h in range(n_kv_heads):
                    if is_key:
                        t_head = inverse_rope(t[h:h+1], base=rope_base)[0]
                    else:
                        t_head = t[h]
                    kept_data  = t_head[keep_mask]
                    rotated    = kept_data @ H.T
                    amax       = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                    sc         = amax / (levels / 2)
                    normalized = rotated / sc
                    groups     = normalized.reshape(-1, 8)
                    lp         = E8Lattice.nearest_point(groups).clamp(-levels / 2, levels / 2)
                    coords     = lp.reshape(-1, head_dim)
                    quantized  = (coords * sc) @ H
                    ic         = coords.detach().numpy()
                    has_half   = np.any(np.abs(ic.flatten() - np.round(ic.flatten())) > 0.25)
                    all_coords.append(
                        np.round(ic.flatten() * 2).astype(np.int8) if has_half
                        else np.round(ic.flatten()).astype(np.int8)
                    )
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

        # compute compressed size
        total_idx = 0
        for coords_arr in all_coords:
            arr       = coords_arr.ravel()
            n_per_tok = len(arr) // n_kept if n_kept > 0 else 0
            if n_per_tok > 0 and len(arr) % n_kept == 0:
                reshaped = arr.reshape(n_kept, n_per_tok)
                delta    = np.zeros_like(reshaped)
                delta[0] = reshaped[0]
                delta[1:] = reshaped[1:] - reshaped[:-1]
                total_idx += len(cctx.compress(delta.astype(np.int8).tobytes()))
            else:
                total_idx += len(cctx.compress(arr.tobytes()))

        scale_bytes = n_kept * n_layers * 2 * n_kv_heads * 2
        mask_bytes  = math.ceil(prefix_len / 8) * n_layers
        total       = total_idx + scale_bytes + mask_bytes
        return {
            "fp16": total_fp16, "total": total,
            "ratio": total_fp16 / total if total > 0 else 0,
            "n_kept": n_kept,
        }

    # ================================================================
    # EVICT + QUANTIZE: physical truncation (Approach 3)
    # ================================================================
    def evict_quantize_physical(kv_cache, keep_mask):
        """Physically remove evicted tokens from KV tensors, then quantize."""
        H = hadamard_matrix(head_dim).cpu()
        n_kept     = keep_mask.sum().item()
        total_fp16 = 0
        all_coords = []
        cctx       = zstandard.ZstdCompressor(level=22)
        key_bits   = 2
        val_bits   = 2

        for l in range(n_layers):
            kl, vl = get_kv(kv_cache, l)
            # [1, n_kv_heads, seq_len, head_dim] → physically keep only mask
            k = kl[:, :, keep_mask, :].float().cpu()   # [1, H, n_kept, d]
            v = vl[:, :, keep_mask, :].float().cpu()
            total_fp16 += kl.numel() * 2 + vl.numel() * 2  # original size

            for is_key, tensor, bits in [
                (True,  k, key_bits),
                (False, v, val_bits),
            ]:
                levels = 2 ** bits
                t = tensor[0].clone()  # [n_kv_heads, n_kept, head_dim]
                for h in range(n_kv_heads):
                    if is_key:
                        t_head = inverse_rope(t[h:h+1], base=rope_base)[0]
                    else:
                        t_head = t[h]
                    # all tokens are kept — quantize everything
                    rotated    = t_head @ H.T
                    amax       = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
                    sc         = amax / (levels / 2)
                    normalized = rotated / sc
                    groups     = normalized.reshape(-1, 8)
                    lp         = E8Lattice.nearest_point(groups).clamp(-levels / 2, levels / 2)
                    coords     = lp.reshape(-1, head_dim)
                    quantized  = (coords * sc) @ H
                    ic         = coords.detach().numpy()
                    has_half   = np.any(np.abs(ic.flatten() - np.round(ic.flatten())) > 0.25)
                    all_coords.append(
                        np.round(ic.flatten() * 2).astype(np.int8) if has_half
                        else np.round(ic.flatten()).astype(np.int8)
                    )
                    if is_key:
                        t[h] = forward_rope(quantized.unsqueeze(0), base=rope_base)[0]
                    else:
                        t[h] = quantized
                # write truncated, quantized tensor back
                if is_key:
                    set_kv(kv_cache, l, t.unsqueeze(0).half().to("cuda"), vl)
                else:
                    kl_now, _ = get_kv(kv_cache, l)
                    set_kv(kv_cache, l, kl_now, t.unsqueeze(0).half().to("cuda"))

        # compression size (same bookkeeping as mask variant)
        total_idx = 0
        for coords_arr in all_coords:
            arr       = coords_arr.ravel()
            n_per_tok = len(arr) // n_kept if n_kept > 0 else 0
            if n_per_tok > 0 and len(arr) % n_kept == 0:
                reshaped  = arr.reshape(n_kept, n_per_tok)
                delta     = np.zeros_like(reshaped)
                delta[0]  = reshaped[0]
                delta[1:] = reshaped[1:] - reshaped[:-1]
                total_idx += len(cctx.compress(delta.astype(np.int8).tobytes()))
            else:
                total_idx += len(cctx.compress(arr.tobytes()))

        scale_bytes = n_kept * n_layers * 2 * n_kv_heads * 2
        mask_bytes  = math.ceil(prefix_len / 8) * n_layers
        total       = total_idx + scale_bytes + mask_bytes
        return {
            "fp16": total_fp16, "total": total,
            "ratio": total_fp16 / total if total > 0 else 0,
            "n_kept": n_kept,
        }

    # ================================================================
    # BUILD KEEP MASK
    # ================================================================
    def build_keep_mask(evict_pct, importance):
        if evict_pct == 0:
            return torch.ones(prefix_len, dtype=torch.bool)
        keep_mask = torch.zeros(prefix_len, dtype=torch.bool)
        keep_mask[0] = True
        keep_mask[-SLIDING_WINDOW:] = True
        n_to_keep   = max(int(prefix_len * (100 - evict_pct) / 100), SLIDING_WINDOW + 1)
        n_from_imp  = n_to_keep - keep_mask.sum().item()
        if n_from_imp > 0:
            imp = importance.clone()
            imp[keep_mask] = -float('inf')
            _, top_idx = imp.topk(min(n_from_imp, (~keep_mask).sum().item()))
            keep_mask[top_idx] = True
        return keep_mask

    # ================================================================
    # EVAL: run continuation and return PPL + delta
    # Pass the model explicitly to avoid closure issues after reload.
    # ================================================================
    def eval_ppl_mask(m, kv_cache, keep_mask):
        """Evaluate PPL using zero-and-mask approach."""
        evict_mask = ~keep_mask
        attn_ctx   = torch.ones(prefix_len, dtype=torch.long, device="cuda")
        attn_ctx[evict_mask] = 0
        attn_full  = torch.cat([attn_ctx,
                                 torch.ones(cont_len, dtype=torch.long, device="cuda")])
        with torch.no_grad():
            cout    = m(full_ids[:, prefix_len:],
                        past_key_values=kv_cache,
                        attention_mask=attn_full.unsqueeze(0),
                        use_cache=True)
            logits  = cout.logits[:, :-1, :].float()
            targets = full_ids[:, prefix_len + 1:]
            loss    = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return torch.exp(loss).item()

    def eval_ppl_physical(m, kv_cache):
        """Evaluate PPL after physical truncation — no attention mask needed."""
        with torch.no_grad():
            cout    = m(full_ids[:, prefix_len:],
                        past_key_values=kv_cache,
                        use_cache=True)
            logits  = cout.logits[:, :-1, :].float()
            targets = full_ids[:, prefix_len + 1:]
            loss    = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return torch.exp(loss).item()

    def delta_pct(ppl):
        return (ppl - baseline_ppl) / baseline_ppl * 100.0

    # ================================================================
    # Pre-compute all importance scores using one prefill pass each.
    # Only one model (eager) is loaded — no OOM risk.
    # ================================================================

    # --- key-key and weighted importances (from KV cache keys) ---
    print("\n--- Pre-computing key-key and weighted importance ---")
    with torch.no_grad():
        pout_base = model_eager(full_ids[:, :prefix_len], use_cache=True)
    imp_keykee   = score_importance_keykee(pout_base.past_key_values)
    imp_weighted = score_importance_weighted(pout_base.past_key_values)
    print("  key-key and weighted importance done")

    # --- real attention weights via forward hooks ---
    # Problem: eager_attention_forward materialises (1, 32, seq, seq) fp32 per
    # layer. At seq=3544 that is ~1.6 GB per layer. With model weights (~14 GB)
    # and activations already live, a 40 GB A100 OOMs.
    #
    # Solution: feed only the last ATTN_WIN tokens as new tokens, with the rest
    # of the prefix provided as past_key_values (so KV computation is free).
    # The attention matrix is then (1, 32, ATTN_WIN, prefix_len) fp32,
    # which is tiny. The importance scores reflect which prefix tokens are
    # attended to by the observation window — exactly what we want.
    #
    # We first do a no-attention-output prefill to fill the KV cache for the
    # first (prefix_len - ATTN_WIN) tokens, then do a short eager pass with
    # output_attentions=True for the last ATTN_WIN tokens.

    ATTN_WIN = max(32, prefix_len // 16)   # matches key-key obs_window
    split    = prefix_len - ATTN_WIN       # tokens fed to fill KV cache

    print(f"\n--- Collecting real attention weights (eager, obs_window={ATTN_WIN}) ---")

    # Step 1: fill KV cache for first `split` tokens (no attn output needed)
    del pout_base
    torch.cuda.empty_cache()
    with torch.no_grad():
        past_fill = model_eager(
            full_ids[:, :split],
            use_cache=True,
            output_attentions=False,
        ).past_key_values

    torch.cuda.empty_cache()

    # Step 2: run the last ATTN_WIN tokens with output_attentions=True
    # Attention matrix is (1, 32, ATTN_WIN, split+ATTN_WIN) — manageable.
    attention_accum = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                attn_weights = output[1]  # (1, n_heads, ATTN_WIN, full_seq)
                # sum over query dim (ATTN_WIN) then heads → (full_seq,)
                importance = attn_weights.float().sum(dim=2).sum(dim=1).squeeze(0).cpu()
                del attn_weights
                if layer_idx not in attention_accum:
                    attention_accum[layer_idx] = importance
                else:
                    attention_accum[layer_idx] += importance
        return hook_fn

    hooks = []
    for i, layer in enumerate(model_eager.model.layers):
        hooks.append(layer.self_attn.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        model_eager(
            full_ids[:, split:prefix_len],
            past_key_values=past_fill,
            use_cache=False,
            output_attentions=True,
        )

    for h in hooks:
        h.remove()
    del past_fill
    torch.cuda.empty_cache()

    if attention_accum:
        full_seq  = prefix_len          # importance vector length = full prefix
        imp_eager = torch.zeros(full_seq, device='cpu')
        for l in range(n_layers):
            if l in attention_accum:
                layer_imp = attention_accum[l]
                # layer_imp length may be split+ATTN_WIN = prefix_len (full KV seen)
                # pad/trim to prefix_len just in case
                if layer_imp.shape[0] < full_seq:
                    pad       = torch.zeros(full_seq - layer_imp.shape[0])
                    layer_imp = torch.cat([pad, layer_imp])
                else:
                    layer_imp = layer_imp[:full_seq]
                pool_kernel = 5
                if full_seq > pool_kernel:
                    imp_1d    = layer_imp.unsqueeze(0).unsqueeze(0)
                    layer_imp = F.avg_pool1d(
                        imp_1d, kernel_size=pool_kernel,
                        padding=pool_kernel // 2, stride=1,
                    ).squeeze()[:full_seq]
                imp_eager += layer_imp
        imp_eager = imp_eager / n_layers
        print(f"  eager importance computed from {len(attention_accum)} layers")
    else:
        print("  WARNING: no attention weights captured — falling back to key-key scorer")
        imp_eager = imp_keykee

    # ================================================================
    # RUN ALL APPROACHES ACROSS EVICTION RATES
    # ================================================================
    results = {
        "Baseline (key-key)":    {},
        "Real attention (eager)": {},
        "Weighted scorer":        {},
        "Physical truncation":    {},
    }

    print("\n" + "=" * 80)
    print("Running eviction sweep...")
    print("=" * 80)
    print(f"{'Approach':<28s} {'Evict%':>7s} {'PPL':>9s} {'Delta%':>9s} {'Ratio':>7s}")
    print("-" * 65)

    for evict_pct in EVICT_RATES:
        # --- Baseline ---
        torch.cuda.empty_cache()
        keep_mask = build_keep_mask(evict_pct, imp_keykee)
        with torch.no_grad():
            pout = model_eager(full_ids[:, :prefix_len], use_cache=True)
        info = evict_quantize_mask(pout.past_key_values, keep_mask)
        ppl  = eval_ppl_mask(model_eager, pout.past_key_values, keep_mask)
        results["Baseline (key-key)"][evict_pct] = {"ppl": ppl, "delta": delta_pct(ppl), "ratio": info["ratio"]}
        print(f"{'Baseline (key-key)':<28s} {evict_pct:>6d}% {ppl:>9.4f} {delta_pct(ppl):>+8.2f}% {info['ratio']:>6.2f}x")

        # --- Real attention (eager) ---
        torch.cuda.empty_cache()
        keep_mask_eager = build_keep_mask(evict_pct, imp_eager)
        with torch.no_grad():
            pout = model_eager(full_ids[:, :prefix_len], use_cache=True)
        info = evict_quantize_mask(pout.past_key_values, keep_mask_eager)
        ppl  = eval_ppl_mask(model_eager, pout.past_key_values, keep_mask_eager)
        results["Real attention (eager)"][evict_pct] = {"ppl": ppl, "delta": delta_pct(ppl), "ratio": info["ratio"]}
        print(f"{'Real attention (eager)':<28s} {evict_pct:>6d}% {ppl:>9.4f} {delta_pct(ppl):>+8.2f}% {info['ratio']:>6.2f}x")

        # --- Weighted scorer ---
        torch.cuda.empty_cache()
        keep_mask_wt = build_keep_mask(evict_pct, imp_weighted)
        with torch.no_grad():
            pout = model_eager(full_ids[:, :prefix_len], use_cache=True)
        info = evict_quantize_mask(pout.past_key_values, keep_mask_wt)
        ppl  = eval_ppl_mask(model_eager, pout.past_key_values, keep_mask_wt)
        results["Weighted scorer"][evict_pct] = {"ppl": ppl, "delta": delta_pct(ppl), "ratio": info["ratio"]}
        print(f"{'Weighted scorer':<28s} {evict_pct:>6d}% {ppl:>9.4f} {delta_pct(ppl):>+8.2f}% {info['ratio']:>6.2f}x")

        # --- Physical truncation ---
        torch.cuda.empty_cache()
        keep_mask_phys = build_keep_mask(evict_pct, imp_keykee)
        with torch.no_grad():
            pout = model_eager(full_ids[:, :prefix_len], use_cache=True)
        info = evict_quantize_physical(pout.past_key_values, keep_mask_phys)
        ppl  = eval_ppl_physical(model_eager, pout.past_key_values)
        results["Physical truncation"][evict_pct] = {"ppl": ppl, "delta": delta_pct(ppl), "ratio": info["ratio"]}
        print(f"{'Physical truncation':<28s} {evict_pct:>6d}% {ppl:>9.4f} {delta_pct(ppl):>+8.2f}% {info['ratio']:>6.2f}x")

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    approaches = [
        "Baseline (key-key)",
        "Real attention (eager)",
        "Weighted scorer",
        "Physical truncation",
    ]

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Model: Mistral-7B-v0.1 | GPU: A100 | prefix={prefix_len} | baseline_PPL={baseline_ppl:.4f}")
    print()
    hdr = f"{'Approach':<28s}"
    for e in EVICT_RATES:
        hdr += f"  {e}%evict Δ%"
    print(hdr)
    print("-" * (28 + 14 * len(EVICT_RATES)))
    for ap in approaches:
        row = f"{ap:<28s}"
        for e in EVICT_RATES:
            d = results[ap][e]["delta"]
            row += f"  {d:>+9.2f}%"
        print(row)

    return results, baseline_ppl, prefix_len, n_tok


# ======================================================================
# LOCAL ENTRYPOINT
# ======================================================================
@app.local_entrypoint()
def main():
    import os
    import time

    print("\n" + "=" * 80)
    print("NEXUSQUANT: Quality Improvement — launching on A100")
    print("=" * 80)

    results, baseline_ppl, prefix_len, n_tok = run_quality_improve.remote()

    EVICT_RATES = [35, 60, 80]
    approaches  = [
        "Baseline (key-key)",
        "Real attention (eager)",
        "Weighted scorer",
        "Physical truncation",
    ]

    # ------------------------------------------------------------------
    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Model: Mistral-7B-v0.1 | GPU: A100 | prefix={prefix_len} | baseline_PPL={baseline_ppl:.4f}")
    print(f"Prior baseline (3544-tok): 35%=+0.43% / 60%=+1.34% / 80%=+2.64%")
    print()
    hdr = f"{'Approach':<28s}"
    for e in EVICT_RATES:
        hdr += f"  {e}%evict Δ%"
    print(hdr)
    print("-" * (28 + 14 * len(EVICT_RATES)))
    for ap in approaches:
        row = f"{ap:<28s}"
        for e in EVICT_RATES:
            d = results[ap][e]["delta"]
            row += f"  {d:>+9.2f}%"
        print(row)

    # ------------------------------------------------------------------
    # Write results file
    out_dir  = os.path.join(os.path.dirname(__file__), "..", ".company", "engineering")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "quality_improve_results.md")

    with open(out_path, "w") as f:
        f.write("# Quality Improvement Experiment — NexusQuant\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write("**GPU:** A100 (40GB)\n")
        f.write("**Model:** mistralai/Mistral-7B-v0.1 (32L, 8 KV heads, d=128, rope_theta=10000)\n")
        f.write(f"**Prefix:** {prefix_len} tokens | **Total tokens:** {n_tok}\n")
        f.write(f"**Baseline PPL:** {baseline_ppl:.4f}\n\n")
        f.write("**Prior reference (3544-tok prefix baseline):**\n")
        f.write("- 35% evict → +0.43% / 10.4x\n")
        f.write("- 60% evict → +1.34% / 16.8x\n")
        f.write("- 80% evict → +2.64% / 33.3x\n\n")
        f.write("---\n\n")
        f.write("## Results\n\n")

        # table header
        f.write("| Approach | 35% evict Δ% | 35% ratio | 60% evict Δ% | 60% ratio | 80% evict Δ% | 80% ratio |\n")
        f.write("|----------|-------------|-----------|-------------|-----------|-------------|----------|\n")
        for ap in approaches:
            row = f"| {ap} |"
            for e in EVICT_RATES:
                d = results[ap][e]["delta"]
                r = results[ap][e]["ratio"]
                row += f" {d:+.3f}% | {r:.2f}x |"
            f.write(row + "\n")

        f.write("\n---\n\n")
        f.write("## Analysis\n\n")

        # find best approach per eviction rate
        for e in EVICT_RATES:
            best_ap   = min(approaches, key=lambda a: results[a][e]["delta"])
            best_delta = results[best_ap][e]["delta"]
            baseline_d = results["Baseline (key-key)"][e]["delta"]
            improvement = baseline_d - best_delta
            f.write(f"**{e}% eviction:** best={best_ap} ({best_delta:+.3f}%), "
                    f"baseline={baseline_d:+.3f}%, improvement={improvement:+.3f}%\n")

        f.write("\n### Approach notes\n")
        f.write("- **Real attention (eager):** Uses actual softmax attention weights from forward pass "
                "to score token importance. Theoretically more accurate than key-key proxy.\n")
        f.write("- **Weighted scorer:** Same key-key scorer but later layers contribute 2x more "
                "(linear ramp 1.0→2.0). Tests whether final-layer importance is more predictive.\n")
        f.write("- **Physical truncation:** Physically removes evicted tokens from KV tensors "
                "(no zero-and-mask). Model sees shorter but complete KV — no attention mask needed. "
                "Tests whether the attention mask zeroing introduces quality loss vs true removal.\n")

    print(f"\nResults written to: {out_path}")
