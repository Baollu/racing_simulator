# Racing Simulator AI

Simulateur de course autonome basé sur l'apprentissage par imitation. Le système permet de collecter des données en conduisant manuellement, d'entraîner un réseau de neurones, puis de laisser l'IA conduire de manière autonome.

## Architecture du projet

```
racing_simulator/
├── main.py                     # Point d'entrée CLI (toutes les commandes)
├── requirements.txt            # Dépendances Python
├── install.sh                  # Script d'installation
├── config/
│   ├── agents_config.json      # Configuration des capteurs de la voiture
│   └── training_config.json    # Hyperparamètres d'entraînement
├── src/
│   ├── client.py               # Communication avec le simulateur Unity
│   ├── input_manager.py        # Gestion clavier / manette
│   ├── data_collector.py       # Enregistrement des données de conduite
│   ├── eda.py                  # Analyse exploratoire des données
│   ├── model.py                # Architecture du réseau de neurones
│   ├── train.py                # Boucle d'entraînement
│   └── drive.py                # Conduite autonome (inférence)
├── data/                       # Données CSV collectées
├── models/                     # Modèles entraînés
└── RacingSimulatorLinux/       # Binaire du simulateur Unity
```

## Pipeline complet

Le projet suit 4 étapes séquentielles :

### Étape 1 — Collecte de données

Tu conduis manuellement la voiture dans le simulateur. Chaque observation (capteurs raycast) et chaque action (direction + accélération) sont enregistrées en CSV.

```bash
python main.py collect --track circuit1
```

**Contrôles clavier :**

| Touche | Action |
|--------|--------|
| W / Flèche haut | Accélérer |
| S / Flèche bas | Freiner / Marche arrière |
| A / Flèche gauche | Tourner à gauche |
| D / Flèche droite | Tourner à droite |
| Espace | Frein d'urgence |
| Q | Quitter |

**Options :**
- `--input controller` : utiliser une manette au lieu du clavier
- `--smoothing 0.5` : lisser les entrées (0 = instantané, 1 = très lissé)
- `--time-scale 2.0` : accélérer la simulation
- `--width 1920 --height 1080` : taille de fenêtre personnalisée
- `--fullscreen` : mode plein écran

Les données sont sauvegardées dans `data/session_{track}_{date}.csv`.

### Étape 2 — Analyse des données (EDA)

Vérifie la qualité des données collectées avant d'entraîner.

```bash
python main.py eda
```

Génère dans `eda_output/` :
- **action_distributions.png** : distribution des valeurs de direction et accélération
- **observation_distributions.png** : distribution des lectures des capteurs
- **correlation_matrix.png** : corrélations entre les features
- **action_timeseries.png** : évolution des actions dans le temps
- **session_stats.png** : nombre d'échantillons par session
- **observation_heatmap.png** : heatmap des observations

### Étape 3 — Entraînement du modèle

Entraîne un réseau de neurones à imiter ta conduite.

```bash
python main.py train
```

**Architecture du réseau :**
```
Observations (raycast)  →  [64 neurones, ReLU]  →  [32 neurones, ReLU]  →  Tanh  →  [direction, accélération]
```

- ~7 000 paramètres (très léger, compatible embarqué)
- Fonction de perte : MSE (Mean Squared Error)
- Optimiseur : Adam (lr=0.001)
- Early stopping : arrête si la validation ne s'améliore plus pendant 10 epochs
- Scheduler : réduit le learning rate de 50% tous les 30 epochs

**Fichiers générés :**
- `models/best_model.pth` : meilleur modèle (plus basse loss de validation)
- `models/final_model.pth` : modèle final
- `models/model.onnx` : export ONNX pour Jetson Nano
- `models/normalization.json` : moyenne et écart-type des observations
- `models/training_curves.png` : courbes de loss

**Options :**
- `--device cuda` : forcer l'entraînement sur GPU
- `--training-config config/training_config.json` : modifier les hyperparamètres

### Étape 4 — Conduite autonome

L'IA prend le volant.

```bash
python main.py drive
```

Boucle d'inférence :
1. Récupère les observations des capteurs
2. Normalise avec la moyenne/écart-type sauvegardés
3. Passe au réseau de neurones → obtient [direction, accélération]
4. Envoie les actions au simulateur
5. Mesure la latence d'inférence

**Options :**
- `--model models/best_model.pth` : choisir un modèle
- `--onnx` : utiliser ONNX Runtime au lieu de PyTorch
- `--max-steps 5000` : limiter le nombre de pas
- `--fullscreen` : plein écran

## Configuration

### agents_config.json — Capteurs de la voiture

```json
{
    "agents": [
        {
            "fov": 180,
            "nbRay": 10
        }
    ]
}
```

| Paramètre | Description | Plage |
|-----------|-------------|-------|
| `fov` | Champ de vision en degrés | 1-180 |
| `nbRay` | Nombre de rayons (capteurs de distance) | 1-50 |

### training_config.json — Hyperparamètres

```json
{
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 100,
    "hidden_sizes": [64, 32],
    "dropout": 0.1,
    "validation_split": 0.2,
    "early_stopping_patience": 10,
    "scheduler_step_size": 30,
    "scheduler_gamma": 0.5
}
```

## Modules détaillés

### src/client.py — SimClient

Interface de communication avec le simulateur Unity via ML-Agents (gRPC).

| Méthode | Description |
|---------|-------------|
| `connect()` | Lance le simulateur et établit la connexion |
| `get_observations()` | Récupère les lectures des capteurs |
| `set_actions()` | Envoie les commandes [direction, accélération] |
| `step()` | Avance la simulation d'un pas |
| `reset()` | Réinitialise la simulation |
| `detect_screen_resolution()` | Auto-détecte la résolution de l'écran |

La résolution de la fenêtre est auto-détectée (80% de l'écran par défaut) et peut être personnalisée avec `--width`, `--height`, `--fullscreen`.

### src/input_manager.py — Entrées utilisateur

Deux backends :
- **KeyboardInputManager** : capture globale via `pynput` (pas besoin de focus)
- **ControllerInputManager** : manette via `pygame` (sticks analogiques + gâchettes)

Le lissage (smoothing) interpole entre la valeur actuelle et la cible pour des mouvements fluides. Une deadzone élimine les micro-mouvements parasites.

### src/data_collector.py — Enregistrement

Enregistre les paires (observation, action) en CSV avec horodatage. Flush automatique toutes les 100 lignes. La fonction `load_dataset()` recharge toutes les sessions pour l'entraînement.

### src/model.py — DrivingModel

Réseau feedforward léger avec couches cachées configurables, dropout pour la régularisation, et sortie Tanh pour borner les actions à [-1, 1]. Export ONNX intégré pour le déploiement sur Jetson Nano.

### src/train.py — Entraînement

Pipeline complet : chargement des données → normalisation → split train/val → entraînement avec early stopping → évaluation (MSE, RMSE, MAE, R²) → sauvegarde du meilleur modèle → export ONNX.

### src/drive.py — Inférence

Deux modes d'inférence :
- **PyTorch** (`run_ai_driver`) : pour le développement et le test
- **ONNX Runtime** (`run_ai_driver_with_onnx`) : pour le déploiement embarqué

## Installation

```bash
bash install.sh
```

Ou manuellement :

```bash
pip install -r requirements.txt
pip install mlagents-envs==1.1.0 --no-deps
```

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Simulation | Unity + ML-Agents |
| Communication | gRPC + protobuf |
| ML | PyTorch |
| Entrées | pynput (clavier), pygame (manette) |
| Données | pandas + numpy → CSV |
| Visualisation | matplotlib + seaborn |
| Déploiement | ONNX Runtime (Jetson Nano) |
| Métriques | scikit-learn |
