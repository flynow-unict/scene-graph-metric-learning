"""Rimozione delle immagini senza contenuto dal disco e dal knowledge graph.

Legge la lista di id prodotta dai controlli di integrita', cancella i JPG
corrispondenti e rimuove le triple relative dall'istanza Virtuoso.

    python -m src.utils.delete_empty_images --dry-run
    python -m src.utils.delete_empty_images

Lo script cancella dati in modo definitivo: conviene lanciarlo prima con
``--dry-run``, che si limita a stampare cosa verrebbe rimosso.
"""

import argparse
import os
import subprocess

from src.utils.paths import DATA_DIR, PROJECT_ROOT

GRAPH_URI = "http://progetto-dl-sw.org/advanced"
IMAGE_URI = "http://progetto-dl-sw.org/images"
BATCH_SIZE = 100


def read_ids(empty_file):
    """Legge gli id delle immagini vuote, una per riga nella forma ``img:<id>``."""
    with open(empty_file, "r") as f:
        return [line.strip().replace("img:", "") for line in f if line.strip()]


def delete_images(ids, images_dir, dry_run=False):
    """Cancella i file JPG e restituisce quanti ne sono stati rimossi."""
    deleted = 0
    for img_id in ids:
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if os.path.exists(img_path):
            if not dry_run:
                os.remove(img_path)
            deleted += 1
    return deleted


def build_sparql(ids):
    """Costruisce il batch di DELETE per isql, a blocchi per non saturare il server."""
    # Il transaction logging rallenta molto una cancellazione di massa
    lines = ["log_enable(2,1);"]
    for i in range(0, len(ids), BATCH_SIZE):
        uris = ", ".join(f"<{IMAGE_URI}/{x}>" for x in ids[i:i + BATCH_SIZE])
        lines.append(f"""
    SPARQL
    WITH <{GRAPH_URI}>
    DELETE {{ ?s ?p ?o }}
    WHERE {{
        ?s ?p ?o .
        FILTER(?s IN ({uris}))
    }};
    """)
    lines.append("checkpoint;")  # forza la scrittura su disco dello stato del DB
    lines.append("exit;")
    return "\n".join(lines)


def run_on_virtuoso(query, container, user, password):
    """Esegue il batch di query sul container Virtuoso via isql."""
    try:
        process = subprocess.Popen(
            ["docker", "exec", "-i", container, "isql", "1111", user, password],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        stdout, stderr = process.communicate(input=query)
        if "Error" in stdout or "Error" in stderr:
            print("Avvisi o errori dal database:")
            if stderr:
                print(stderr)
        else:
            print("Nodi eliminati dal knowledge graph.")
    except Exception as e:
        print(f"Errore durante l'esecuzione del comando Docker: {e}")


def main(args):
    ids = read_ids(args.empty_file)
    print(f"Trovate {len(ids)} immagini vuote.")

    deleted = delete_images(ids, args.images_dir, args.dry_run)
    verb = "da eliminare" if args.dry_run else "eliminati"
    print(f"File JPG {verb}: {deleted}")

    if args.dry_run:
        print("Dry run: il knowledge graph non e' stato toccato.")
        return
    run_on_virtuoso(build_sparql(ids), args.container, args.user, args.password)
    print("Pulizia completata.")


def build_parser():
    p = argparse.ArgumentParser(description="Elimina le immagini vuote da disco e da Virtuoso")
    p.add_argument("--empty-file", dest="empty_file",
                   default=str(PROJECT_ROOT / "empty_images.txt"),
                   help="file con gli id delle immagini vuote, uno per riga")
    p.add_argument("--images-dir", dest="images_dir", default=str(DATA_DIR / "images"),
                   help="cartella da cui rimuovere i JPG")
    p.add_argument("--container", default="virtuoso_advanced")
    p.add_argument("--user", default="dba")
    p.add_argument("--password", default=os.environ.get("VIRTUOSO_PASSWORD", ""),
                   help="password di Virtuoso; per default si legge da VIRTUOSO_PASSWORD")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="mostra cosa verrebbe eliminato senza cancellare nulla")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
