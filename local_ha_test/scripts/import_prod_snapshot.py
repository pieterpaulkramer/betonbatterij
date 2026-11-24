"""Safely import a production Recorder DB snapshot into local_ha_test/config/.

Allows testing BetonBatterij's full 90-day LTS + 10-day raw Recorder backfill,
14-day hindcast evaluation, and MAP calibration locally on real house history
without modifying production Home Assistant or committing private data.

Stop the Docker container before running this script so SQLite WAL/SHM files
are not modified while Home Assistant holds them open.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import shutil
import sqlite3

_LOGGER = logging.getLogger(__name__)
_LOCAL_HA_DIR = pathlib.Path(__file__).resolve().parent.parent
_REPO_ROOT = _LOCAL_HA_DIR.parent


def _verify_sqlite_db(db_path: pathlib.Path) -> tuple[int, int]:
    """Validate that `db_path` is a readable Home Assistant Recorder database."""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        cursor = conn.cursor()
        tables = {
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "states" not in tables:
            raise ValueError(
                f"{db_path} does not look like a Home Assistant Recorder DB "
                "(missing 'states' table)."
            )
        state_count = int(
            cursor.execute("SELECT COUNT(*) FROM states").fetchone()[0]
        )
        stats_count = 0
        if "statistics" in tables:
            stats_count = int(
                cursor.execute("SELECT COUNT(*) FROM statistics").fetchone()[0]
            )
    return state_count, stats_count


def import_snapshot(
    source_db: pathlib.Path,
    *,
    copy_local_json: bool = True,
    fresh_backfill: bool = False,
    config_dir: pathlib.Path | None = None,
) -> None:
    """Copy a Recorder SQLite database into `local_ha_test/config/`.

    Args:
        source_db: Path to a copied `home-assistant_v2.db` file.
        copy_local_json: Also copy workspace `betonbatterij_local.json` into
            `local_ha_test/config/` if present so entity IDs match the DB.
        fresh_backfill: Delete existing `.storage/betonbatterij.*` files so
            BetonBatterij performs a full cold-start Recorder/LTS backfill on
            next startup.
        config_dir: Override target HA `/config` directory.
    """
    if not source_db.is_file():
        raise FileNotFoundError(f"Source database not found: {source_db}")

    state_count, stats_count = _verify_sqlite_db(source_db)
    _LOGGER.info(
        "Verified source DB %s (%d states, %d LTS statistics rows).",
        source_db,
        state_count,
        stats_count,
    )

    target_dir = config_dir or (_LOCAL_HA_DIR / "config")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_db = target_dir / "home-assistant_v2.db"

    # Remove stale WAL/SHM sidecar files before replacing the SQLite DB.
    for suffix in ("-wal", "-shm"):
        sidecar = target_dir / f"home-assistant_v2.db{suffix}"
        if sidecar.is_file():
            sidecar.unlink()

    shutil.copy2(source_db, target_db)
    _LOGGER.info("Copied Recorder DB to %s.", target_db)

    if copy_local_json:
        src_local = _REPO_ROOT / "betonbatterij_local.json"
        if src_local.is_file():
            dst_local = target_dir / "betonbatterij_local.json"
            shutil.copy2(src_local, dst_local)
            _LOGGER.info("Copied entity mapping to %s.", dst_local)
        else:
            _LOGGER.warning(
                "No %s found; BetonBatterij will use generic default entity IDs.",
                src_local,
            )

    if fresh_backfill:
        storage_dir = target_dir / ".storage"
        removed = 0
        if storage_dir.is_dir():
            for path in storage_dir.glob("betonbatterij.*"):
                if path.is_file():
                    path.unlink()
                    removed += 1
        _LOGGER.info(
            "Cleared %d BetonBatterij .storage files for cold-start backfill.",
            removed,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import a production Home Assistant Recorder DB snapshot into "
            "local_ha_test/config/ (stop Docker HA first)."
        )
    )
    parser.add_argument(
        "source_db",
        type=pathlib.Path,
        help="Path to your copied home-assistant_v2.db file.",
    )
    parser.add_argument(
        "--no-local-json",
        action="store_true",
        help="Do not copy workspace betonbatterij_local.json.",
    )
    parser.add_argument(
        "--fresh-backfill",
        action="store_true",
        help="Delete .storage/betonbatterij.* to force cold-start backfill.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = _parse_args()
    import_snapshot(
        args.source_db,
        copy_local_json=not args.no_local_json,
        fresh_backfill=args.fresh_backfill,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
