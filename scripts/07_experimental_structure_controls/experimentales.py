#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Por cada pestaña de un Excel:
  - toma entradas únicas de la 2ª columna
  - consulta GPCRdb para obtener estructuras del receptor
  - clasifica cada PDB como ACTIVE/INACTIVE mirando la página:
        https://gpcrdb.org/structure/<PDB>
    según si en LIGANDS pone "Agonist" o "Antagonist" (o "Inverse agonist")

Salida:
  - 2 hojas por pestaña: <tab>_all y <tab>_with_active_and_inactive
  - 1 hoja "summary"

Requisitos:
  pip install requests pandas openpyxl
"""

import re
import time
import requests
import pandas as pd


# ----------------------------
# Configuración
# ----------------------------
INPUT_XLSX = "gpcr_distances.xlsx"
OUTPUT_XLSX = "gpcr_active_inactive_by_sheet.xlsx"

# Si quieres ser más "amable" con el servidor:
SLEEP_BETWEEN_REQUESTS_S = 0.0  # p.ej. 0.05

# Si algún entry_name de GPCRdb no sigue el patrón esperado, sobrescríbelo aquí:
CUSTOM_NAMES = {
    # "opsb": "opsb_bovin",
}

# API de GPCRdb para estructuras por proteína:
BASE_URL = "https://gpcrdb.org/services/structure/protein/{entry_name}/"

# Página HTML por estructura (aquí es donde miramos LIGANDS y detectamos Agonist/Antagonist):
STRUCTURE_PAGE_URL = "https://gpcrdb.org/structure/{pdb}"


# ----------------------------
# Helpers Excel
# ----------------------------
def sanitize_sheet_name(name: str) -> str:
    """Excel limita nombres de hoja a 31 chars y prohíbe: : \\ / ? * [ ]"""
    name = re.sub(r"[:\\/*?\[\]]", "_", str(name))
    name = name.strip() or "sheet"
    return name[:31]


def make_unique_sheet_name(base: str, used: set) -> str:
    """Evita colisiones de nombres tras truncar/sanitizar."""
    base = sanitize_sheet_name(base)
    if base not in used:
        used.add(base)
        return base
    i = 2
    while True:
        suffix = f"_{i}"
        candidate = sanitize_sheet_name(base[: (31 - len(suffix))] + suffix)
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


def unique_from_second_column(df: pd.DataFrame):
    """Devuelve valores únicos (orden preservado) de la 2ª columna."""
    if df.shape[1] < 2:
        return []
    col = df.columns[1]
    vals = df[col].dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    return list(pd.unique(vals))


# ----------------------------
# GPCRdb: entrada -> entry_name
# ----------------------------
def short_to_entry(short_name: str) -> str:
    """
    Convierte a entry_name de GPCRdb.

    Reglas:
      - Si está en CUSTOM_NAMES: usa eso
      - Si ya parece entry_name (contiene '_' o acaba en '_human'): úsalo tal cual
      - 'gp123' -> 'gpr123_human'
      - En otro caso: '<short>_human'
    """
    if short_name is None:
        return ""
    s = str(short_name).strip().lower()
    if not s:
        return ""

    if s in CUSTOM_NAMES:
        return CUSTOM_NAMES[s]

    # Si ya viene como entry_name (ej. "gpr161_human")
    if "_" in s:
        return s

    m = re.match(r"^gp(\d+)$", s)
    if m:
        return f"gpr{m.group(1)}_human"

    return f"{s}_human"


# ----------------------------
# GPCRdb API: estructuras por entry
# ----------------------------
def fetch_structures_for_entry(entry_name: str, api_cache: dict, timeout=30):
    """Llama a la API de GPCRdb y devuelve lista JSON de estructuras para un entry_name (con cache)."""
    if not entry_name:
        return None

    if entry_name in api_cache:
        return api_cache[entry_name]

    url = BASE_URL.format(entry_name=entry_name)
    try:
        resp = requests.get(url, timeout=timeout)
    except Exception as e:
        noting = f"[WARN] {entry_name}: error de conexión ({e})"
        print(noting)
        api_cache[entry_name] = None
        return None

    if resp.status_code != 200:
        print(f"[WARN] {entry_name}: error HTTP {resp.status_code}")
        api_cache[entry_name] = None
        return None

    try:
        data = resp.json()
    except Exception as e:
        print(f"[WARN] {entry_name}: no se pudo parsear JSON ({e})")
        api_cache[entry_name] = None
        return None

    if not isinstance(data, list):
        print(f"[WARN] {entry_name}: formato inesperado (no es lista)")
        api_cache[entry_name] = None
        return None

    api_cache[entry_name] = data
    if SLEEP_BETWEEN_REQUESTS_S > 0:
        time.sleep(SLEEP_BETWEEN_REQUESTS_S)
    return data


# ----------------------------
# Scraping GPCRdb /structure/<PDB> para detectar Agonist/Antagonist
# ----------------------------
# OJO: "antagonist" contiene "agonist" -> hay que comprobar antagonista antes.
ROLE_PATTERNS = [
    ("inactive", re.compile(r"\binverse\s+agonist\b", re.I)),
    ("inactive", re.compile(r"\bantagonist\b", re.I)),
    ("active",   re.compile(r"\bagonist\b", re.I)),  # incluye "partial agonist", etc.
]


def fetch_structure_page_text(pdb_code: str, page_cache: dict, timeout=30) -> str:
    """
    Descarga la página /structure/<PDB> y devuelve texto plano alrededor del bloque LIGANDS.
    Cachea por PDB para no repetir llamadas.
    """
    pdb = (pdb_code or "").strip().upper()
    if not pdb:
        return ""

    if pdb in page_cache:
        return page_cache[pdb]

    url = STRUCTURE_PAGE_URL.format(pdb=pdb)
    try:
        r = requests.get(url, timeout=timeout)
    except Exception as e:
        print(f"[WARN] {pdb}: error descargando página ({e})")
        page_cache[pdb] = ""
        return ""

    if r.status_code != 200:
        print(f"[WARN] {pdb}: HTTP {r.status_code} en página estructura")
        page_cache[pdb] = ""
        return ""

    html = r.text
    low = html.lower()

    # Intento recortar cerca del bloque "#### LIGANDS"
    idx = low.find("#### ligands")
    if idx == -1:
        # fallback más laxo
        idx = low.find(">ligands<")

    snippet = html[idx: idx + 6000] if idx != -1 else html

    # Texto plano simple (sin dependencias extra)
    text = re.sub(r"<[^>]+>", " ", snippet)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    page_cache[pdb] = text
    if SLEEP_BETWEEN_REQUESTS_S > 0:
        time.sleep(SLEEP_BETWEEN_REQUESTS_S)
    return text


def classify_pdb_by_structure_page(pdb_code: str, page_cache: dict):
    """
    Devuelve (clase, detalle):
      - clase: "active" / "inactive" / "unknown" / "ambiguous"
      - detalle: trozo de texto donde se buscó (para auditoría)
    """
    text = fetch_structure_page_text(pdb_code, page_cache)
    if not text:
        return "unknown", "no_page_or_no_text"

    # Nos centramos en la zona donde aparece "LIGANDS"
    t_low = text.lower()
    cut = t_low.find("ligands")
    zone = text[cut: cut + 1200] if cut != -1 else text[:1200]

    hits = []
    for label, pat in ROLE_PATTERNS:
        if pat.search(zone):
            hits.append(label)

    hits = sorted(set(hits))
    if len(hits) == 1:
        consider = "agonist/antagonist hit"
        return hits[0], f"{consider} | {zone}"
    if len(hits) > 1:
        return "ambiguous", zone
    return "unknown", zone


def classify_structures_by_ligands(struct_list, page_cache: dict):
    """
    Clasifica PDBs en active/inactive mirando Agonist/Antagonist en la página /structure/<PDB>.
    """
    active, inactive = [], []
    other = {}  # motivo -> lista de PDBs

    for s in struct_list:
        pdb = (s.get("pdb_code") or "").upper().strip()
        if not pdb:
            continue

        cls, _detail = classify_pdb_by_structure_page(pdb, page_cache)
        if cls == "active":
            active.append(pdb)
        elif cls == "inactive":
            inactive.append(pdb)
        else:
            other.setdefault(cls, []).append(pdb)

    other_states_str = "; ".join(
        f"{k}: {', '.join(sorted(set(v)))}" for k, v in sorted(other.items())
    )
    return active, inactive, other_states_str


# ----------------------------
# Main
# ----------------------------
def main():
    xls = pd.ExcelFile(INPUT_XLSX)
    sheets = xls.sheet_names
    print(f"[INFO] Encontradas {len(sheets)} pestañas: {sheets}")

    api_cache = {}    # entry_name -> list|None
    page_cache = {}   # PDB -> texto plano (zona LIGANDS)

    used_sheetnames = set()
    summary_rows = []

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        for tab in sheets:
            df_tab = pd.read_excel(INPUT_XLSX, sheet_name=tab)
            receptors = unique_from_second_column(df_tab)

            print(f"\n[INFO] Pestaña '{tab}': {len(receptors)} entradas únicas (2ª columna)")

            rows = []
            for short_name in receptors:
                entry_name = short_to_entry(short_name)
                if not entry_name:
                    continue

                print(f"[INFO]   - {short_name} -> {entry_name}")

                data = fetch_structures_for_entry(entry_name, api_cache=api_cache)
                if not data:
                    rows.append({
                        "receptor_input": short_name,
                        "entry_name": entry_name,
                        "n_structures": 0,
                        "n_active_by_ligand": 0,
                        "n_inactive_by_ligand": 0,
                        "active_pdbs": "",
                        "inactive_pdbs": "",
                        "other_pdbs": "",
                        "error": "sin estructuras o error en API",
                    })
                    continue

                active, inactive, other_states_str = classify_structures_by_ligands(
                    data, page_cache=page_cache
                )

                rows.append({
                    "receptor_input": short_name,
                    "entry_name": entry_name,
                    "n_structures": len(data),
                    "n_active_by_ligand": len(set(active)),
                    "n_inactive_by_ligand": len(set(inactive)),
                    "active_pdbs": ", ".join(sorted(set(active))),
                    "inactive_pdbs": ", ".join(sorted(set(inactive))),
                    "other_pdbs": other_states_str,
                    "error": "",
                })

            df_out = pd.DataFrame(rows)

            # Filtro: al menos 1 "active" y 1 "inactive" según Agonist/Antagonist
            df_pair = df_out[
                (df_out["n_active_by_ligand"] > 0) & (df_out["n_inactive_by_ligand"] > 0)
            ].copy()

            # Nombres de hojas (evitando colisiones y el límite de Excel)
            all_name = make_unique_sheet_name(f"{tab}_all", used_sheetnames)
            pair_name = make_unique_sheet_name(f"{tab}_with_active_and_inactive", used_sheetnames)

            df_out.to_excel(writer, sheet_name=all_name, index=False)
            df_pair.to_excel(writer, sheet_name=pair_name, index=False)

            summary_rows.append({
                "tab": tab,
                "unique_inputs": len(receptors),
                "rows_written": len(df_out),
                "with_active_and_inactive": len(df_pair),
                "sheet_all": all_name,
                "sheet_with_active_and_inactive": pair_name,
            })

        # Hoja resumen
        summary_df = pd.DataFrame(summary_rows)
        summary_name = make_unique_sheet_name("summary", used_sheetnames)
        summary_df.to_excel(writer, sheet_name=summary_name, index=False)

    print(f"\n[OK] Guardado Excel en: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()

