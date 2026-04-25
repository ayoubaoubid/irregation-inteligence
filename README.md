# irregation-inteligence

Pipeline automatique minimale pour ce projet MLOps:

1. validation des fichiers de donnees dans `DataOps/Statics`
2. preprocessing des datasets a partir du CSV brut
3. entrainement + evaluation du modele via DVC
4. verification Django avec `manage.py check`
5. execution des tests Django

Lancer en local:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/run_pipeline.py
dvc repro
```

Executer seulement les validations sans tests:

```powershell
python scripts/run_pipeline.py --skip-tests
```

Pipeline CI:

- le workflow GitHub Actions est dans `.github/workflows/mlops-pipeline.yml`
- il se lance automatiquement sur chaque `push` et `pull_request`
- les datasets de `DataOps/Statics` sont suivis par DVC, pas directement par Git
- le remote DVC par defaut pointe vers DagsHub
- la CI GitHub n'execute pas `dvc pull` ni `dvc repro`
- la CI garde des verifications compatibles cloud, comme `manage.py check`
- les tests qui dependent des artefacts de modele locaux sont ignores si `models/` n'est pas disponible

Pipeline data/model local recommande:

```powershell
dvc pull
dvc repro
python scripts/evaluate.py
python scripts/run_pipeline.py --skip-tests
```

Etapes DVC:

- `preprocess`: construit `irrigation_prediction_Variables_Important.csv` et `irrigation_prediction_processed.csv` depuis `irrigation_prediction.csv`
- `train`: entraine les modeles et sauvegarde les artefacts dans `models/`
- `test`: execute `pytest tests/`
- `evaluate`: calcule les metriques sur le dataset preprocess
