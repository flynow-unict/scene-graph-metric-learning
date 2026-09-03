"""Percorsi del progetto, risolti a partire dalla posizione di questo file.

Tutti gli script fanno riferimento a queste costanti invece di scrivere path
relativi, cosi' funzionano indipendentemente dalla directory da cui vengono
lanciati e restano validi se la struttura delle cartelle cambia.
"""

from pathlib import Path

# src/utils/paths.py -> src/utils -> src -> radice del progetto
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
RESULTS_DIR: Path = DOCS_DIR / "results"

EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
CONFIGS_DIR: Path = EXPERIMENTS_DIR / "configs"
LOGS_DIR: Path = EXPERIMENTS_DIR / "logs"

# Sotto-cartelle di data/
IMAGES_DIR: Path = DATA_DIR / "images"
SCENE_GRAPH_DIR: Path = DATA_DIR / "sceneGraph"
MODELS_DIR: Path = DATA_DIR / "models"


def scene_graphs(corpus: str = "fullset", variant: str = "semantic",
                 stage: str = "embedded") -> Path:
    """Cartella degli scene graph per un dato corpus.

    Args:
        corpus: ``fullset`` (corpus completo) oppure ``subset`` (corpus ridotto).
        variant: ``semantic`` (arricchito da Virtuoso) oppure ``baseline``.
        stage: ``embedded`` (con feature NOMIC) oppure ``raw``.
    """
    return SCENE_GRAPH_DIR / corpus / variant / stage


def checkpoints(corpus: str = "fullset") -> Path:
    """Cartella dei checkpoint del graph encoder per un dato corpus."""
    return MODELS_DIR / corpus / "checkpoints"


def embeddings(corpus: str = "fullset", variant: str = "semantic") -> Path:
    """Cartella degli embedding estratti dal graph encoder."""
    return MODELS_DIR / corpus / variant / "gcn"


def faiss_indices(corpus: str = "fullset") -> Path:
    """Cartella degli indici FAISS."""
    return MODELS_DIR / corpus / "faiss"
