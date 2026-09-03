"""Caricamento dei file di configurazione degli esperimenti.

Gli script mantengono la loro interfaccia ``argparse``: il file YAML si limita a
sostituire i valori di default, in modo che un argomento passato esplicitamente
da riga di comando abbia sempre la precedenza.

Ordine di priorita': riga di comando > file YAML > default dello script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Legge un file YAML di configurazione e ne restituisce il contenuto."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"file di configurazione non trovato: {config_path}")
    content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return content or {}


def parse_with_config(parser: argparse.ArgumentParser,
                      argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Aggiunge l'opzione ``--config`` al parser e ne applica il contenuto.

    Le chiavi dello YAML devono corrispondere ai nomi delle opzioni del parser
    (``batch_size`` per ``--batch-size``). Una chiave sconosciuta interrompe
    l'esecuzione con un errore esplicito, per non far passare in silenzio un
    esperimento configurato male.
    """
    parser.add_argument("--config", type=str, default=None,
                        help="file YAML in experiments/configs/ con i parametri dell'esperimento")
    known, _ = parser.parse_known_args(argv)
    if known.config:
        config = load_config(known.config)
        valid = {action.dest for action in parser._actions}
        unknown = set(config) - valid
        if unknown:
            raise SystemExit(
                f"[config] chiavi non riconosciute in {known.config}: {sorted(unknown)}"
            )
        parser.set_defaults(**config)
    return parser.parse_args(argv)
