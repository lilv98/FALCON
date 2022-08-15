# FALCON: Sound and Complete Neural Semantic Entailment over ALC Ontologies

## Requirements

- Python packages
    * python == 3.8.5
    * torch == 1.8.1
    * numpy == 1.19.2
    * pandas == 1.0.1
    * tqdm == 4.61.0

- Others
    * groovy == 4.0.0
    * JVM == 1.8.0_333
    * Protégé (https://protege.stanford.edu/)


## Run
- Family Ontology
    > `cd ./code/model`

    > `python family.py`
- Pizza Ontology
    > `cd ./code/model`
    
    > `python pizza.py`
- Human Phenotype Ontology
    > `cd ./data/HPO/ && unzip BIOGRID-ALL-4.4.211.tab.zip && cd ../../code/model`

    > `sh run_hpo.sh`

## Data Preparation
We elaborate the steps of data preparation to foster further research. This section is unnecessary for running the experiments. 
- Human Phenotype Ontology
    - Download datasets
        > `cd ./data/HPO/`

        > `wget http://purl.obolibrary.org/obo/hp.owl`

        > `wget https://downloads.thebiogrid.org/File/BioGRID/Release-Archive/BIOGRID-4.4.211/BIOGRID-ALL-4.4.211.tab.zip`

        > `unzip ./data/HPO/BIOGRID-ALL-4.4.211.tab.zip`
        
        > `wget http://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt`
    - Semantic Entailment (Generating the True Testing Axioms)
        > Open `hp.owl` with the graphical interface of `Protégé` 

        > Select `ELK` as the logical reasoner

        > Save the inferred ontology as `hpInferred.owl`
    - OWL to Axioms
        > `groovy ./code/ppc/GetTBox.groovy ./data/HPO/hp.owl > ./data/HPO/TBox.txt`

        > `groovy ./code/ppc/GetTBox.groovy ./data/HPO/hpInferred.owl > ./data/HPO/TBoxInferred.txt`

- Pizza Ontology
    - Download datasets
        > `cd ./data/Pizza/`

        > `wget https://protege.stanford.edu/ontologies/pizza/pizza.owl`

    - Semantic Entailment (Generating the True Testing Axioms)
        > Open `pizza.owl` with the graphical interface of `Protégé` 

        > Select `HermiT` as the logical reasoner

        > Save the inferred ontology as `pizzaInferred.owl`

    - OWL to Axioms
        > `groovy ./code/ppc/GetTBox.groovy ./data/Pizza/pizza.owl > ./data/Pizza/pizzaTBox.txt`

        > `groovy ./code/ppc/GetTBox.groovy ./data/Pizza/pizzaInferred.owl > ./data/Pizza/pizzaTBoxInferred.txt`
