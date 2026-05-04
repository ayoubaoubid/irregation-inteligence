# Rapport de Projet MLOps

## Titre
Plateforme d'irrigation intelligente avec pipeline MLOps automatise

## Resume
Ce projet presente la conception et la realisation d'une plateforme d'irrigation intelligente appuyee par des techniques de Machine Learning et des pratiques MLOps. L'objectif principal est d'aider a la decision d'irrigation a partir de variables agronomiques et climatiques tout en assurant la tracabilite des donnees, l'automatisation du pipeline et la reproductibilite des resultats. La solution developpee combine une application Django, un pipeline DVC pour le versionnement des donnees et des artefacts, MLflow pour le suivi des experiences, ainsi qu'une integration continue basee sur GitHub Actions et DagsHub. Les tests locaux ont valide le flux automatique d'ajout de donnees, de mise a jour du suivi DVC et de reexecution du pipeline. La publication distante des artefacts est deleguee a la CI afin de rendre l'architecture plus robuste.

## Abstract
This project presents the design and implementation of an intelligent irrigation platform supported by Machine Learning techniques and MLOps practices. The main objective is to assist irrigation decisions from agronomic and climate variables while ensuring data traceability, pipeline automation, and experiment reproducibility. The implemented solution combines a Django application, a DVC pipeline for data and artifact versioning, MLflow for experiment tracking, and a continuous integration workflow based on GitHub Actions and DagsHub. Local tests validated the automated flow for data ingestion, DVC update, and pipeline reexecution. Remote artifact publication is delegated to CI in order to improve reliability.

## Introduction generale
L'agriculture moderne fait face a plusieurs defis lies a l'optimisation des ressources hydriques, a la variabilite climatique et a la recherche de meilleurs rendements. Dans ce contexte, l'irrigation intelligente represente une approche pertinente pour appuyer la prise de decision en s'appuyant sur les donnees. Le present projet s'inscrit dans cette dynamique en proposant une plateforme capable de centraliser les donnees, de lancer un pipeline de traitement et d'entrainement, puis de fournir une base technique exploitable pour la prediction du besoin d'irrigation.

Au-dela de la simple construction d'un modele de classification, le projet adopte une logique MLOps. Cette orientation vise a assurer un cycle de vie plus fiable du modele, allant de la preparation des donnees a l'automatisation du reentrainement, en passant par le versionnement des artefacts et l'integration continue. Le travail realise ne se limite donc pas a la performance predictive, mais integre egalement des considerations de reproductibilite, de maintenance et de collaboration.

Le rapport presente successivement le contexte du projet, les donnees utilisees, la chaine de preparation et de modelisation, la mise en oeuvre du pipeline MLOps, ainsi que les tests et les limites identifiees.

## Chapitre 1 - Analyse du besoin et etude prealable
### 1.1 Contexte
La decision d'irrigation depend d'un ensemble de facteurs tels que l'humidite, la temperature, la pluviometrie, le type d'irrigation, la phase de croissance de la culture et les caracteristiques du sol. Une decision inadaptee peut entrainer soit une surconsommation d'eau, soit un stress hydrique pour la culture. L'automatisation de cette decision a l'aide de modeles de Machine Learning constitue donc un axe de travail pertinent.

### 1.2 Problematique
La problematique traitee dans ce projet est la suivante : comment concevoir une plateforme capable de collecter de nouvelles observations agricoles, de maintenir un pipeline de donnees reproductible et de reentrainer automatiquement un modele de prediction du besoin d'irrigation ?

### 1.3 Objectifs
- Mettre en place une application web pour consulter, analyser et enrichir le jeu de donnees.
- Construire un pipeline de preparation et d'entrainement reproductible.
- Versionner les donnees, les artefacts de modele et les etats du pipeline.
- Automatiser le flux apres l'ajout de nouvelles donnees.
- Inte grer une chaine CI pour la validation et la publication des artefacts.

### 1.4 Contraintes
Le projet doit fonctionner dans un environnement d'apprentissage avec des moyens limites, ce qui impose des choix simples mais robustes. Les contraintes principales concernent la disponibilite des dependances, la gestion des acces reseau aux services distants, ainsi que la coordination entre Git, DVC, Django et GitHub Actions.

## Chapitre 2 - Conception et architecture de la solution
### 2.1 Vue d'ensemble
La solution repose sur plusieurs composants interconnectes. L'application Django constitue la couche de presentation et permet l'ajout de nouvelles donnees, l'analyse visuelle et l'acces aux fonctionnalites de prediction. Les scripts Python assurent les traitements de preparation, d'entrainement et d'evaluation. DVC orchestre les differentes etapes du pipeline et suit les jeux de donnees et les artefacts. MLflow joue le role de suivi d'experimentation. Enfin, GitHub Actions et DagsHub assurent la validation continue et la synchronisation distante.

### 2.2 Architecture logicielle
Le depot contient plusieurs repertoires representatifs de cette architecture. Le dossier `Plateform_IA/` heberge l'application Django. Le dossier `scripts/` contient les scripts metier du pipeline. Le dossier `DataOps/Statics/` centralise les donnees sources et transformees. Le dossier `models/` regroupe les artefacts produits par l'entrainement. Les fichiers `dvc.yaml` et `dvc.lock` decrivent et figent l'etat du pipeline.

### 2.3 Pipeline DVC
Le pipeline comprend quatre etapes principales.
- `preprocess` : preparation du dataset final a partir du jeu de donnees suivi.
- `train` : entrainement des modeles et generation des artefacts.
- `test` : execution des tests automatiques.
- `evaluate` : calcul et journalisation des metriques finales.

Cette structuration permet de relancer uniquement les etapes devenues invalides apres une modification des dependances.

## Chapitre 3 - Donnees et preparation
### 3.1 Jeu de donnees
Le fichier `DataOps/Statics/irrigation_prediction_Variables_Important.csv` constitue le jeu de donnees central suivi par DVC. Il contient 10002 lignes et 15 colonnes. Les variables disponibles sont : `Soil_pH`, `Soil_Moisture`, `Organic_Carbon`, `Electrical_Conductivity`, `Temperature_C`, `Humidity`, `Rainfall_mm`, `Sunlight_Hours`, `Wind_Speed_kmh`, `Crop_Growth_Stage`, `Irrigation_Type`, `Field_Area_hectare`, `Mulching_Used`, `Previous_Irrigation_mm` et `Irrigation_Need`.

### 3.2 Variable cible
La variable cible du projet est `Irrigation_Need`. Elle represente le besoin d'irrigation a predire et sert de base a la tache de classification supervisee.

### 3.3 Preparation des donnees
Le script `scripts/preprocess.py` prepare un jeu de donnees transforme appele `irrigation_prediction_processed.csv`. A l'etat actuel du projet, ce fichier contient 17592 lignes et 22 colonnes, ce qui montre que certaines transformations et expansions de variables ont ete appliquees, notamment l'encodage des categories.

Les principales etapes de preparation sont :
- lecture du dataset source ;
- nettoyage et transformation des colonnes ;
- encodage des variables categorielles ;
- production d'un dataset exploitable pour l'entrainement.

## Chapitre 4 - Modelisation et evaluation
### 4.1 Demarche de modelisation
Le projet adopte une approche comparative entre plusieurs algorithmes de classification. Le script `scripts/train.py` charge le dataset pretraite, separe les donnees en ensembles d'entrainement et de test avec un ratio de 80 pour cent et 20 pour cent, puis applique une normalisation par `StandardScaler`.

### 4.2 Modeles testes
Les modeles actuellement integres sont :
- `LogisticRegression`
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `SVC`
- `KNeighborsClassifier`

Les hyperparametres principaux sont centralises dans `params.yml`, ce qui facilite leur versionnement et leur reproduction.

### 4.3 Selection du meilleur modele
Le critere principal de selection est le score F1 pondere. Lors d'une execution observee dans les logs du pipeline, le meilleur modele obtenu est `RandomForest` avec un score F1 de `0.9994316563`. Les scores constates pour les autres modeles montrent egalement un comportement satisfaisant, avec des performances particulierement elevees pour `GradientBoosting` et bonnes pour `SVM`.

### 4.4 Artefacts produits
L'etape d'entrainement genere plusieurs artefacts essentiels :
- `models/best_model.pkl`
- `models/scaler.pkl`
- `models/features.pkl`
- `models/classes.pkl`

Ces fichiers sont suivis par DVC et reutilises par les etapes suivantes.

### 4.5 Evaluation
Le script `scripts/evaluate.py` charge le modele retenu, aligne les variables d'entree et calcule plusieurs metriques. Une execution observee dans les logs a montre une `Accuracy` de `0.9998863119` et un `F1-score` de `0.9998863119` sur le dataset evalue, avec une matrice de confusion tres proche d'une classification parfaite.

## Chapitre 5 - Mise en oeuvre MLOps
### 5.1 Versionnement des donnees et artefacts
Le projet s'appuie sur DVC pour suivre les jeux de donnees et les artefacts produits par le pipeline. Le fichier `dvc.yaml` decrit les etapes, leurs dependances et leurs sorties. Le fichier `dvc.lock` enregistre l'etat exact des dependances et des sorties apres execution. Ce mecanisme permet de figer les hashes des fichiers et de verifier si un stage doit etre rejoue.

### 5.2 Suivi des experiences
MLflow est utilise pour journaliser les parametres, les metriques et les modeles. Cette integration apporte une trace des essais realises et facilite la comparaison entre approches de modelisation.

### 5.3 Automatisation apres ajout de donnees
La vue `add_data` de l'application Django permet d'ajouter une nouvelle observation au fichier CSV principal. Une fois la ligne enregistree, un worker asynchrone declenche automatiquement :
- `dvc add` sur le dataset suivi ;
- `dvc repro` pour rejouer le pipeline ;
- `git add`, `git commit` et `git push` pour versionner les metadonnees mises a jour.

Cette logique rend l'application capable de prendre en compte de nouvelles donnees sans intervention manuelle sur les principales etapes locales.

### 5.4 Integration continue
Le workflow `mlops-pipeline.yml` configure une chaine CI sur GitHub Actions. Celle-ci installe les dependances, configure l'acces au remote DVC, tente un `dvc pull`, execute `dvc repro` lorsque les artefacts distants sont disponibles, puis lance des controles Django et des tests conditionnels. Une etape `dvc push` est egalement prevue lors des evenements de type `push`, ce qui deplace la publication vers DagsHub hors du serveur applicatif.

### 5.5 Interet de l'architecture retenue
La separation entre le traitement local dans Django et la publication distante dans GitHub Actions renforce la robustesse du systeme. Elle evite qu'un probleme reseau ou d'authentification sur le serveur applicatif bloque l'ensemble du flux metier.

## Chapitre 6 - Realisation de la plateforme
### 6.1 Fonctionnalites principales
La plateforme propose plusieurs fonctionnalites :
- consultation des donnees ;
- analyse statistique et visuelle ;
- ajout de nouvelles observations ;
- execution automatique du pipeline associe ;
- exposition des resultats de prediction.

### 6.2 Interface d'analyse
La vue `analysis` calcule plusieurs elements de synthese a partir du dataset : distributions par niveau de besoin, type d'irrigation, utilisation du paillage, stade de croissance, graphiques radar, nuages de points et bulles. Cette partie transforme les donnees en indicateurs interpretabes par l'utilisateur.

### 6.3 Interface d'ajout de donnees
La vue `add_data` collecte les valeurs d'un formulaire, les stocke dans le fichier CSV et met a jour un statut de pipeline accessible a l'utilisateur. Cette fonctionnalite joue un role central dans la boucle de mise a jour du modele.

## Chapitre 7 - Tests, validation et discussion
### 7.1 Validation locale
Des tests ont ete menes pour verifier le flux `add_data -> dvc add -> dvc repro`. La soumission via la vue Django a bien ajoute une observation au dataset. Le pipeline local a ensuite relance les etapes de preparation, d'entrainement et de test. Les journaux consultes montrent que `pytest` a execute 7 tests avec succes.

### 7.2 Resultats de validation
La logique locale de mise a jour du pipeline peut donc etre consideree comme validee. La structure du projet permet d'illustrer un cas concret de reentrainement automatise a partir d'une interface applicative.

### 7.3 Limites observees
Une limite importante a ete observee lors des tests de publication distante directe : le `dvc push` effectue depuis l'environnement local peut echouer en raison d'un acces reseau ou d'une authentification insuffisante vers DagsHub. Cette contrainte ne remet pas en cause la logique applicative, mais a motive le transfert de cette responsabilite vers GitHub Actions.

### 7.4 Discussion
Le projet atteint un bon niveau de coherence pedagogique pour un cas d'usage MLOps. Il combine une application web, un pipeline de donnees, une orchestration par DVC, des artefacts de modele versionnes, un suivi d'experience et une integration continue. Les limitations restantes sont surtout liees a l'environnement d'execution et non au principe technique retenu.

## Conclusion generale
Ce projet a permis de mettre en place une chaine MLOps appliquee a un cas d'irrigation intelligente. La solution obtenue ne se limite pas a un modele de classification, mais integre egalement le versionnement des donnees, la reproductibilite des traitements, la gestion des artefacts et l'automatisation de plusieurs etapes critiques.

Les resultats montrent que la plateforme est capable d'integrer de nouvelles observations, de relancer localement le pipeline et de conserver une trace exploitable des differents etats du systeme. L'integration de GitHub Actions et DagsHub constitue une extension naturelle pour industrialiser la publication distante.

Parmi les perspectives d'amelioration, il est possible d'envisager un monitoring plus pousse, une meilleure gestion des secrets et credentials, une base de suivi MLflow plus robuste que le stockage fichier, ainsi qu'une strategie de deploiement plus complete. Dans l'ensemble, le projet constitue une base solide pour illustrer les principes du MLOps dans un contexte applicatif concret.

## Bibliographie indicative
- Documentation officielle de Django
- Documentation officielle de DVC
- Documentation officielle de MLflow
- Documentation officielle de GitHub Actions
- Documentation officielle de DagsHub
- Documentation officielle de scikit-learn
