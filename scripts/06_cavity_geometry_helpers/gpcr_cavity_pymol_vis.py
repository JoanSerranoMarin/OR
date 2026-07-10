#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script sencillo para generar los vóxels del bolsillo ortostérico (OPV)
de OR8K3 usando la función run(...) del script gpcr_cavity_volume.py.
"""

import gpcr_cavity_volume as gcv  # asegúrate de que el archivo se llama así

if __name__ == "__main__":
    gcv.run(
        inpdb="or8k3_all_active.pdb",      # tu estructura
        outpdb=None,                       # o por ejemplo "or8k3_all_active_aligned.pdb"
        tm_include="1-7",                  # TM1–TM7
        spacing=0.3,                       # el mismo spacing que en el método
        probe=0.0,                         # mismo probe que en el método
        voxels_free=None,                  # no queremos el volumen total aquí
        voxels_bwpoly="or8k3_OPV_voxels.pdb",  # <- ESTE es el archivo que quieres
        caps_pdb=None,                     # opcionalmente puedes poner "or8k3_caps.pdb"
        spike_dirs=20,                     # 20 rayos como en tu descripción
        spike_max=10.0,                    # 10 Å
        spike_probe=0.0,                   # igual que en el método
        bw_str=(
            "3.32,3.33,3.34,3.35,3.36,"
            "5.42,5.43,5.44,5.45,5.46,"
            "6.48,6.49,6.50,6.51,6.52,"
            "7.39,7.40,7.41,7.42,7.43"
        ),
    )

