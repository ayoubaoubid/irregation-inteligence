# irregation-inteligence

Pipeline automatique minimale pour ce projet MLOps:

1. validation des fichiers de donnees dans `DataOps/Statics`
2. verification Django avec `manage.py check`
3. execution des tests Django

Lancer en local:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/run_pipeline.py
```

Executer seulement les validations sans tests:

```powershell
python scripts/run_pipeline.py --skip-tests
```

Pipeline CI:

- le workflow GitHub Actions est dans `.github/workflows/mlops-pipeline.yml`
- il se lance automatiquement sur chaque `push` et `pull_request`
- avant `dvc repro`, la CI execute `dvc pull`
- la CI ne peut pas utiliser un remote local comme `../dvc-storage`; il faut un remote DVC partage accessible depuis GitHub Actions, puis executer `dvc push` depuis votre machine
