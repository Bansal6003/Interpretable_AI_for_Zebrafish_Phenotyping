**Human-Inspired AI for Zebrafish Phenotype & Behavior Analysis**

**Problem**

Manual zebrafish phenotyping is slow, subjective, and difficult to scale. In an NIH R24-funded effort to model human disease via CRISPR-Cas9 knockouts, we needed automated, quantitative, and biologically meaningful phenotypic analysis.

**Solution**

I developed an end-to-end AI system for:

•	Automated morphological classification

•	Fine-grained anatomical segmentation and morphometry

•	Behavioral time-series analysis under electromechanical stimuli

**Data**

•	10,000+ larval images and behavioral sequences

•	~400 CRISPR mutant lines

•	Multiple anatomical regions: whole larva, eye, head, yolk sac, pericardium, etc.

**Models**

•	Classification: RegNetY-16 (transfer learning)

•	Segmentation: DeepLabV3+ with RegNetY-16 encoder

•	Behavior: Siamese LSTM (time-series similarity learning)

**Results**

•	98.5% classification accuracy

•	High IoU and Dice scores across anatomical regions

•	Automated morphometry validated by zebrafish biologists

•	Significant reduction in manual FIJI-based analysis time

•	Experiment tracked and parameters logged using MLflow


**Impact**

•	Enabled high-throughput, reproducible phenotyping

•	Accelerated genotype-phenotype discovery

•	Provided interpretable, quantitative outputs trusted by domain experts

•	Readily available phenotypic and behvaioral information on 400+ different mutants

Implementation was accelerated using AI-assisted coding tools; all model design, experimentation, debugging, and validation were performed manually.

