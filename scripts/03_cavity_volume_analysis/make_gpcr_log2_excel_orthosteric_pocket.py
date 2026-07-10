#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#python3 make_gpcr_log2_excel.py   --active volumenes_active.txt   --inactive volumenes_inactive.txt   --familias familias.txt   --out gpcr_log2_active_inactive.xlsx


import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

def parse_vol_file(path: Path) -> pd.DataFrame:
    """
    Lee un archivo tipo 'volumenes_*.txt'.
    Formato esperado: dos columnas (id, valor). Soporta tabs o espacios.
    Ignora líneas vacías o con valores no numéricos.
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            key = parts[0]
            val_str = parts[-1]
            try:
                val = float(val_str)
            except Exception:
                continue
            rows.append((key, val))
    if not rows:
        return pd.DataFrame(columns=["key", "value"])
    return pd.DataFrame(rows, columns=["key", "value"])

def extract_receptor(raw: str) -> str:
    """Lo que va antes del primer '_' (o, si no hay, antes del primer '.')."""
    s = str(raw)
    if "_" in s:
        s = s.split("_", 1)[0]
    else:
        s = s.split(".", 1)[0]
    return s.strip().lower()

def build_table(active_path: Path, inactive_path: Path, familias_path: Path) -> pd.DataFrame:
    # Activo / Inactivo
    df_act  = parse_vol_file(active_path)
    df_inac = parse_vol_file(inactive_path)

    df_act["receptor"]  = df_act["key"].map(extract_receptor)
    df_inac["receptor"] = df_inac["key"].map(extract_receptor)

    # Si hubiera varias filas por receptor, promediamos
    act  = df_act.groupby("receptor", as_index=False)["value"].mean().rename(columns={"value":"Active"})
    inac = df_inac.groupby("receptor", as_index=False)["value"].mean().rename(columns={"value":"Inactive"})

    merged = inac.merge(act, on="receptor", how="outer")

    # log2(Active/Inactive) seguro
    def safe_log2(a, i):
        if a is None or i is None:
            return np.nan
        if pd.isna(a) or pd.isna(i) or a <= 0 or i <= 0:
            return np.nan
        return float(np.log2(a / i))

    merged["log2(active/inactive)"] = [
        safe_log2(a, i) for a, i in zip(merged["Active"], merged["Inactive"])
    ]

    # Familias: 1ª col = receptor en MAYÚSCULAS, última col = Class
    fam = pd.read_csv(familias_path, sep=r"\t+", engine="python", header=0)
    fam = fam.iloc[:, [0, -1]].copy()
    fam.columns = ["GPCR_CODE", "Class"]
    fam["GPCR_CODE"] = fam["GPCR_CODE"].astype(str).str.strip().str.upper()

    merged["GPCR"] = merged["receptor"].str.upper()
    final_df = merged.merge(fam, left_on="GPCR", right_on="GPCR_CODE", how="left") \
                     .drop(columns=["GPCR","GPCR_CODE"]) \
                     .rename(columns={"receptor":"GPCR_name"}) \
                     [["GPCR_name","Inactive","Active","log2(active/inactive)","Class"]]

    return final_df.sort_values("GPCR_name").reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser(description="Crea un Excel con Inactive, Active, log2(active/inactive) y Class por GPCR.")
    ap.add_argument("--active",   required=True, help="Ruta a volumenes_active.txt")
    ap.add_argument("--inactive", required=True, help="Ruta a volumenes_inactive.txt")
    ap.add_argument("--familias", required=True, help="Ruta a familias.txt (Class en la última columna)")
    ap.add_argument("--out", default="gpcr_log2_active_inactive.xlsx", help="Nombre del Excel de salida")
    args = ap.parse_args()

    active_path   = Path(args.active)
    inactive_path = Path(args.inactive)
    familias_path = Path(args.familias)
    out_path      = Path(args.out)

    df = build_table(active_path, inactive_path, familias_path)

    # Guardar Excel (pandas usará openpyxl/xlsxwriter si están instalados)
    df.to_excel(out_path, sheet_name="log2_active_inactive", index=False)

    # Reporte por consola
    n = len(df)
    matched = df["Class"].notna().sum()
    missing = n - matched
    print(f"[OK] Escrito: {out_path}")
    print(f"  Receptores: {n} | con clase: {matched} | sin clase: {missing}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

