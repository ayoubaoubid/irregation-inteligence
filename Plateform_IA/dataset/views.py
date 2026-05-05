import csv
import json
import os
import subprocess
import sys
import threading
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

import pandas as pd
from sklearn.preprocessing import StandardScaler
from django.conf import settings
from django.shortcuts import redirect, render

from .forms import AddDataForm


_DVC_PIPELINE_LOCK = threading.Lock()
_DVC_PIPELINE_RUNNING = False

IMPORTANT_COLUMNS = [
    'Soil_pH',
    'Soil_Moisture',
    'Organic_Carbon',
    'Electrical_Conductivity',
    'Temperature_C',
    'Humidity',
    'Rainfall_mm',
    'Sunlight_Hours',
    'Wind_Speed_kmh',
    'Crop_Growth_Stage',
    'Irrigation_Type',
    'Field_Area_hectare',
    'Mulching_Used',
    'Previous_Irrigation_mm',
    'Irrigation_Need',
]

NUMERIC_COLUMNS = [
    'Soil_pH',
    'Soil_Moisture',
    'Organic_Carbon',
    'Electrical_Conductivity',
    'Temperature_C',
    'Humidity',
    'Rainfall_mm',
    'Sunlight_Hours',
    'Wind_Speed_kmh',
    'Field_Area_hectare',
    'Previous_Irrigation_mm',
]

CATEGORICAL_COLUMNS = [
    'Crop_Growth_Stage',
    'Irrigation_Type',
    'Mulching_Used',
]


def get_dataset_csv_path():
    return (
        settings.BASE_DIR.parent
        / 'DataOps'
        / 'Statics'
        / 'irrigation_prediction_Variables_Important.csv'
    )


def get_project_root():
    configured_root = os.getenv('PROJECT_ROOT')
    if configured_root:
        return Path(configured_root)

    base_dir = settings.BASE_DIR.resolve()
    candidate_roots = [base_dir, *base_dir.parents]
    for candidate in candidate_roots:
        if (candidate / 'dvc.yaml').exists():
            return candidate

    return settings.BASE_DIR.parent


def get_dvc_log_path():
    return get_project_root() / 'logs' / 'dvc_pipeline.log'


def get_dvc_status_path():
    return get_project_root() / 'logs' / 'dvc_pipeline_status.json'


def get_new_data_path():
    return get_project_root() / 'logs' / 'new_data.csv'


def get_pipeline_unavailable_reason():
    project_root = get_project_root()
    required_paths = [
        project_root / 'dvc.yaml',
        project_root / 'scripts',
        project_root / 'DataOps' / 'Statics',
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        return (
            'Le pipeline DVC est indisponible dans cet environnement. '
            'Chemins manquants: ' + ', '.join(missing_paths)
        )

    git_dir = project_root / '.git'
    if not git_dir.exists():
        return (
            "Le pipeline DVC est indisponible dans ce conteneur Docker car le depot Git "
            "complet n'est pas monte. Monte le projet racine avec son dossier .git pour "
            'autoriser les etapes git/dvc automatiques.'
        )

    return None


def ensure_git_identity(project_root, log_file):
    git_email = os.getenv('PIPELINE_GIT_EMAIL', 'pipeline@local.dev')
    git_name = os.getenv('PIPELINE_GIT_NAME', 'DVC Pipeline')

    for key, value in (
        ('user.email', git_email),
        ('user.name', git_name),
    ):
        subprocess.run(
            ['git', 'config', '--local', key, value],
            cwd=project_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def write_dvc_status(state, message, started_at=None, finished_at=None):
    status_path = get_dvc_status_path()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'state': state,
        'message': message,
        'started_at': started_at,
        'finished_at': finished_at,
    }
    temp_path = status_path.with_suffix('.tmp')
    with open(temp_path, 'w', encoding='utf-8') as status_file:
        json.dump(payload, status_file)
        status_file.flush()
        os.fsync(status_file.fileno())
    os.replace(temp_path, status_path)


def read_dvc_status():
    status_path = get_dvc_status_path()
    if not status_path.exists():
        return None

    try:
        with open(status_path, 'r', encoding='utf-8') as status_file:
            content = status_file.read().strip()
            if not content:
                return None
            return json.loads(content)
    except (json.JSONDecodeError, OSError):
        return None


def get_visible_pipeline_status(request):
    pipeline_status = read_dvc_status()
    if not pipeline_status:
        return None

    if pipeline_status.get('state') == 'running':
        return pipeline_status

    if request.GET.get('success') == '1':
        return pipeline_status

    return None


def append_new_data_for_drift(raw_row):
    dataset_path = get_dataset_csv_path()
    new_data_path = get_new_data_path()

    if not dataset_path.exists():
        return

    historical_df = pd.read_csv(dataset_path, encoding='utf-8-sig')
    if historical_df.empty:
        return

    important_df = historical_df.loc[:, IMPORTANT_COLUMNS].copy().dropna().reset_index(drop=True)
    numeric_part = important_df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors='coerce')
    categorical_part = important_df[CATEGORICAL_COLUMNS].fillna('')

    scaler = StandardScaler()
    scaled_numeric = pd.DataFrame(
        scaler.fit_transform(numeric_part),
        columns=NUMERIC_COLUMNS,
        index=important_df.index,
    )
    encoded_categorical = pd.get_dummies(categorical_part)
    processed_features = pd.concat([scaled_numeric, encoded_categorical], axis=1)

    latest_raw = important_df.iloc[[-1]].copy()
    latest_numeric = latest_raw[NUMERIC_COLUMNS].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    latest_scaled = pd.DataFrame(
        scaler.transform(latest_numeric),
        columns=NUMERIC_COLUMNS,
        index=latest_raw.index,
    )
    latest_encoded = pd.get_dummies(latest_raw[CATEGORICAL_COLUMNS].fillna(''))
    latest_encoded = latest_encoded.reindex(
        columns=encoded_categorical.columns,
        fill_value=0.0,
    )

    latest_processed = pd.concat([latest_scaled, latest_encoded], axis=1)
    latest_processed = latest_processed.reindex(
        columns=processed_features.columns,
        fill_value=0.0,
    )

    new_data_path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = (not new_data_path.exists()) or new_data_path.stat().st_size == 0
    latest_processed.to_csv(
        new_data_path,
        mode='a',
        header=should_write_header,
        index=False,
    )


def trigger_dvc_pipeline_async():
    global _DVC_PIPELINE_RUNNING

    with _DVC_PIPELINE_LOCK:
        if _DVC_PIPELINE_RUNNING:
            return False
        _DVC_PIPELINE_RUNNING = True

    def worker():
        global _DVC_PIPELINE_RUNNING
        log_path = get_dvc_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started_at = datetime.utcnow().isoformat(timespec='seconds')
        project_root = get_project_root()
        last_command = None

        try:
            unavailable_reason = get_pipeline_unavailable_reason()
            if unavailable_reason:
                write_dvc_status(
                    state='skipped',
                    message=unavailable_reason,
                    started_at=started_at,
                    finished_at=started_at,
                )
                with open(log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(
                        f"\n[{started_at}] DVC pipeline skipped: {unavailable_reason}\n"
                    )
                return

            write_dvc_status(
                state='running',
                message='Le pipeline DVC local et la preparation du commit Git sont en cours d execution.',
                started_at=started_at,
                finished_at=None,
            )
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"\n[{started_at}] Starting DVC pipeline after add-data submission\n")
                log_file.flush()

                for command in (
                    [
                        sys.executable,
                        '-m',
                        'dvc',
                        'add',
                        'DataOps/Statics/irrigation_prediction_Variables_Important.csv',
                    ],
                    [sys.executable, '-m', 'dvc', 'repro'],
                ):
                    last_command = command
                    log_file.write(f"[{started_at}] Running: {' '.join(command)}\n")
                    log_file.flush()
                    subprocess.run(
                        command,
                        cwd=project_root,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )

                git_status_cmd = ['git', 'status', '--short']
                last_command = git_status_cmd
                log_file.write(f"[{started_at}] Running: {' '.join(git_status_cmd)}\n")
                log_file.flush()
                git_status = subprocess.run(
                    git_status_cmd,
                    cwd=project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=True,
                )
                log_file.write(git_status.stdout)

                ensure_git_identity(project_root, log_file)

                tracked_paths = [
                    'DataOps/Statics/irrigation_prediction_Variables_Important.csv.dvc',
                    'dvc.lock',
                ]
                # DVC stage outputs stay out of Git; only metadata files should be staged here.
                git_add_cmd = ['git', 'add', *tracked_paths]
                last_command = git_add_cmd
                log_file.write(f"[{started_at}] Running: {' '.join(git_add_cmd)}\n")
                log_file.flush()
                subprocess.run(
                    git_add_cmd,
                    cwd=project_root,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                )

                git_diff_cmd = ['git', 'diff', '--cached', '--quiet']
                last_command = git_diff_cmd
                log_file.write(f"[{started_at}] Running: {' '.join(git_diff_cmd)}\n")
                log_file.flush()
                git_diff = subprocess.run(
                    git_diff_cmd,
                    cwd=project_root,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

                if git_diff.returncode == 1:
                    commit_message = (
                        f"Update irrigation dataset and model artifacts {started_at}"
                    )
                    git_commit_cmd = ['git', 'commit', '-m', commit_message]
                    last_command = git_commit_cmd
                    log_file.write(f"[{started_at}] Running: {' '.join(git_commit_cmd)}\n")
                    log_file.flush()
                    subprocess.run(
                        git_commit_cmd,
                        cwd=project_root,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                elif git_diff.returncode != 0:
                    raise subprocess.CalledProcessError(git_diff.returncode, git_diff_cmd)

                finished_at = datetime.utcnow().isoformat(timespec='seconds')
                log_file.write(f"[{finished_at}] DVC pipeline completed successfully\n")
                write_dvc_status(
                    state='success',
                    message='Le pipeline DVC local s est termine avec succes. Les changements sont prets en local pour une publication Git manuelle ou via GitHub Actions.',
                    started_at=started_at,
                    finished_at=finished_at,
                )
        except subprocess.CalledProcessError as exc:
            failed_at = datetime.utcnow().isoformat(timespec='seconds')
            failed_command = ' '.join(last_command or exc.cmd or [])
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(
                    f"[{failed_at}] DVC pipeline failed with exit code {exc.returncode}\n"
                )
                if failed_command:
                    log_file.write(f"[{failed_at}] Failed command: {failed_command}\n")
            write_dvc_status(
                state='failed',
                message=(
                    f'Le pipeline DVC a echoue avec le code {exc.returncode}. '
                    f'Commande: {failed_command or "inconnue"}. '
                    'Consulte logs/dvc_pipeline.log.'
                ),
                started_at=started_at,
                finished_at=failed_at,
            )
        except Exception as exc:
            failed_at = datetime.utcnow().isoformat(timespec='seconds')
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"[{failed_at}] Unexpected pipeline error: {exc}\n")
            write_dvc_status(
                state='failed',
                message=f'Le pipeline DVC a echoue: {exc}. Consulte logs/dvc_pipeline.log.',
                started_at=started_at,
                finished_at=failed_at,
            )
        finally:
            with _DVC_PIPELINE_LOCK:
                _DVC_PIPELINE_RUNNING = False

    threading.Thread(target=worker, daemon=True).start()
    return True


def dataset(request):
    return render(request, 'dataset/dataset.html')


def analysis(request):
    csv_path = get_dataset_csv_path()
    if csv_path.exists():
        with open(csv_path, newline='', encoding='utf-8-sig') as file:
            rows = list(csv.DictReader(file))
    else:
        rows = []

    need_order = ['Low', 'Medium', 'High']
    irrigation_order = ['Drip', 'Sprinkler', 'Canal', 'Rainfed']
    stage_order = ['Sowing', 'Vegetative', 'Flowering', 'Harvest']
    mulching_order = ['Yes', 'No']

    def to_float(row, key, default=0.0):
        value = row.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def normalize_group_means(grouped_values):
        all_values = [value for values in grouped_values.values() for value in values]
        if not all_values:
            return {key: 0 for key in grouped_values}

        min_value = min(all_values)
        max_value = max(all_values)
        if max_value == min_value:
            return {key: 0 for key in grouped_values}

        normalized = {}
        for key, values in grouped_values.items():
            avg = mean(values) if values else min_value
            normalized[key] = round((avg - min_value) / (max_value - min_value), 3)
        return normalized

    irrigation_need_counts = Counter(row['Irrigation_Need'] for row in rows)
    irrigation_type_counts = Counter(row['Irrigation_Type'] for row in rows)
    mulching_counts = Counter(row['Mulching_Used'] for row in rows)
    growth_stage_counts = Counter(row['Crop_Growth_Stage'] for row in rows)

    scatter_points = defaultdict(list)
    line_metrics = {
        'Rainfall_mm': defaultdict(list),
        'Humidity': defaultdict(list),
        'Sunlight_Hours': defaultdict(list),
    }
    radar_metrics = {
        'Soil_Moisture': defaultdict(list),
        'Soil_pH': defaultdict(list),
        'Organic_Carbon': defaultdict(list),
        'Electrical_Conductivity': defaultdict(list),
        'Temperature_C': defaultdict(list),
        'Humidity': defaultdict(list),
    }
    stacked_counts = {
        stage: Counter() for stage in stage_order
    }
    bubble_points = defaultdict(list)

    for row in rows:
        need = row['Irrigation_Need']
        irrigation_type = row['Irrigation_Type']
        stage = row['Crop_Growth_Stage']

        soil_moisture = to_float(row, 'Soil_Moisture')
        temperature = to_float(row, 'Temperature_C')
        rainfall = to_float(row, 'Rainfall_mm')
        humidity = to_float(row, 'Humidity')
        sunlight = to_float(row, 'Sunlight_Hours')
        field_area = to_float(row, 'Field_Area_hectare')

        scatter_points[need].append({
            'x': round(soil_moisture, 2),
            'y': round(temperature, 2),
        })

        line_metrics['Rainfall_mm'][need].append(rainfall)
        line_metrics['Humidity'][need].append(humidity)
        line_metrics['Sunlight_Hours'][need].append(sunlight)

        radar_metrics['Soil_Moisture'][irrigation_type].append(soil_moisture)
        radar_metrics['Soil_pH'][irrigation_type].append(to_float(row, 'Soil_pH'))
        radar_metrics['Organic_Carbon'][irrigation_type].append(to_float(row, 'Organic_Carbon'))
        radar_metrics['Electrical_Conductivity'][irrigation_type].append(to_float(row, 'Electrical_Conductivity'))
        radar_metrics['Temperature_C'][irrigation_type].append(temperature)
        radar_metrics['Humidity'][irrigation_type].append(humidity)

        if stage in stacked_counts:
            stacked_counts[stage][need] += 1

        bubble_points[need].append({
            'x': round(rainfall, 2),
            'y': round(soil_moisture, 2),
            'r': max(4, min(round(field_area * 1.2, 1), 18)),
        })

    line_chart_data = {
        'labels': need_order,
        'datasets': [
            {
                'label': 'Rainfall (Norm 0-1)',
                'data': [normalize_group_means(line_metrics['Rainfall_mm']).get(label, 0) for label in need_order],
                'borderColor': '#1f77b4',
            },
            {
                'label': 'Humidity (Norm 0-1)',
                'data': [normalize_group_means(line_metrics['Humidity']).get(label, 0) for label in need_order],
                'borderColor': '#2ca02c',
            },
            {
                'label': 'Sunlight Hours (Norm 0-1)',
                'data': [normalize_group_means(line_metrics['Sunlight_Hours']).get(label, 0) for label in need_order],
                'borderColor': '#ff7f0e',
            },
        ],
    }

    radar_labels = [
        'Soil Moisture',
        'Soil pH (Norm)',
        'Organic Carbon (Norm)',
        'Conductivity (Norm)',
        'Temp (Norm)',
        'Humidity',
    ]
    radar_metric_order = [
        'Soil_Moisture',
        'Soil_pH',
        'Organic_Carbon',
        'Electrical_Conductivity',
        'Temperature_C',
        'Humidity',
    ]
    radar_colors = {
        'Drip': ('#4e79a7', 'rgba(78, 121, 167, 0.2)'),
        'Sprinkler': ('#f28e2b', 'rgba(242, 142, 43, 0.2)'),
        'Canal': ('#e15759', 'rgba(225, 87, 89, 0.2)'),
        'Rainfed': ('#76b7b2', 'rgba(118, 183, 178, 0.2)'),
    }

    normalized_radar_metrics = {
        metric: normalize_group_means(grouped)
        for metric, grouped in radar_metrics.items()
    }

    radar_chart_data = {
        'labels': radar_labels,
        'datasets': [],
    }
    for irrigation_type in irrigation_order:
        border_color, background_color = radar_colors[irrigation_type]
        radar_chart_data['datasets'].append({
            'label': irrigation_type,
            'data': [
                round(normalized_radar_metrics[metric].get(irrigation_type, 0) * 100, 2)
                for metric in radar_metric_order
            ],
            'borderColor': border_color,
            'backgroundColor': background_color,
            'borderWidth': 2,
        })

    stacked_chart_data = {
        'labels': stage_order,
        'datasets': [],
    }
    total_rows = len(rows)
    stacked_colors = {
        'Low': '#87ab69',
        'Medium': '#ffc107',
        'High': '#fd7e14',
    }
    for need in need_order:
        series = []
        for stage in stage_order:
            total = sum(stacked_counts[stage].values())
            percent = (stacked_counts[stage][need] / total * 100) if total else 0
            series.append(round(percent, 2))
        stacked_chart_data['datasets'].append({
            'label': f'{need} Need',
            'data': series,
            'backgroundColor': stacked_colors[need],
        })

    context = {
        'chart1_labels': json.dumps(need_order),
        'chart1_values': json.dumps([irrigation_need_counts.get(label, 0) for label in need_order]),
        'chart2_labels': json.dumps(irrigation_order),
        'chart2_values': json.dumps([irrigation_type_counts.get(label, 0) for label in irrigation_order]),
        'chart3_labels': json.dumps([
            f'{label} ({round((mulching_counts.get(label, 0) / total_rows) * 100) if total_rows else 0}%)'
            for label in mulching_order
        ]),
        'chart3_values': json.dumps([mulching_counts.get(label, 0) for label in mulching_order]),
        'chart4_labels': json.dumps(stage_order),
        'chart4_values': json.dumps([growth_stage_counts.get(label, 0) for label in stage_order]),
        'chart5_datasets': json.dumps([
            {
                'label': f'{need} Need',
                'data': scatter_points.get(need, []),
            }
            for need in need_order
        ]),
        'chart6_data': json.dumps(line_chart_data),
        'chart7_data': json.dumps(radar_chart_data),
        'chart8_data': json.dumps(stacked_chart_data),
        'chart9_datasets': json.dumps([
            {
                'label': f'{need} Need',
                'data': bubble_points.get(need, []),
            }
            for need in need_order
        ]),
    }
    return render(request, 'dataset/analysis.html', context)


def add_data(request):
    csv_path = get_dataset_csv_path()
    fieldnames = IMPORTANT_COLUMNS

    if request.method == 'POST':
        form = AddDataForm(request.POST)
        if not form.is_valid():
            return render(
                request,
                'dataset/add_data.html',
                {
                    'success': False,
                    'pipeline_status': get_visible_pipeline_status(request),
                    'form_errors': form.errors,
                    'submitted_data': request.POST,
                },
                status=400,
            )

        data = {field: form.cleaned_data[field] for field in fieldnames}

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_path.exists()
        should_write_header = (not file_exists) or csv_path.stat().st_size == 0

        with open(csv_path, mode='a', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if should_write_header:
                writer.writeheader()
            writer.writerow(data)

        append_new_data_for_drift(data)

        pipeline_started = trigger_dvc_pipeline_async()
        if not pipeline_started:
            write_dvc_status(
                state='running',
                message='Un pipeline DVC est deja en cours. Cette nouvelle donnee sera prise en compte dans ce cycle.',
                started_at=datetime.utcnow().isoformat(timespec='seconds'),
                finished_at=None,
            )
        return redirect(f"{request.path}?success=1")

    return render(
        request,
        'dataset/add_data.html',
        {
            'success': request.GET.get('success') == '1',
            'pipeline_status': get_visible_pipeline_status(request),
            'form_errors': None,
            'submitted_data': {},
        },
    )
