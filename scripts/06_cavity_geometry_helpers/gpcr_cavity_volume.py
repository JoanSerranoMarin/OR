from pymol import cmd

def show_gpcr_cavity(
    gpcr_pdb,
    voxel_pdb,
    prefix="OPV",
    spacing=0.3,
    cartoon_color="grey80",
    cavity_color="yellow",
    make_surface="1",
    gaussian_sigma=0.8,
    gaussian_buffer=4.0,
    sphere_scale=None,
    show_spheres="1"
):
    """
    Visualiza en PyMOL una cavidad de voxels calculada con tu script de volúmenes.

    Parámetros
    ----------
    gpcr_pdb : str
        Ruta al PDB de la proteína (o nombre de objeto ya cargado).
    voxel_pdb : str
        Ruta al PDB de voxels (salida de --voxels-free o --voxels-bwpoly).
    prefix : str
        Prefijo para los nombres de objetos en PyMOL (p.ej. "OPV").
        Se crearán: <prefix>_rec, <prefix>_vox, <prefix>_surf, <prefix>_map.
    spacing : float
        Espaciado del grid de voxels en Å (el que hayas usado en tu script, p.ej. 0.3).
    cartoon_color : str
        Color para el cartoon del receptor.
    cavity_color : str
        Color para la cavidad (esferas y superficie).
    make_surface : "0" o "1"
        Si es "1", genera además una superficie suave a partir de los voxels.
    gaussian_sigma : float
        Sigma del mapa gaussiano (en Å) para la superficie (pymol: map_new gaussian).
    gaussian_buffer : float
        Distancia extra alrededor de los voxels para el mapa gaussiano.
    sphere_scale : float o None
        Radio de las esferas de los voxels. Si None, se usa 0.5 * spacing.
    show_spheres : "0" o "1"
        Si es "1", deja visibles las esferas de los voxels; si "0", solo la superficie.
    """

    # ---- Conversión de parámetros tipo string de PyMOL a tipos útiles ----
    spacing = float(spacing)
    gaussian_sigma = float(gaussian_sigma)
    gaussian_buffer = float(gaussian_buffer)
    make_surface = (str(make_surface) != "0")
    show_spheres = (str(show_spheres) != "0")

    if sphere_scale is None or str(sphere_scale).strip() == "":
        sphere_scale = 0.5 * spacing
    else:
        sphere_scale = float(sphere_scale)

    # ---- Nombres de objetos internos ----
    rec_obj  = f"{prefix}_rec"
    vox_obj  = f"{prefix}_vox"
    map_name = f"{prefix}_map"
    surf_obj = f"{prefix}_surf"

    # ---- Cargar receptor ----
    # Si gpcr_pdb coincide con un objeto ya cargado, PyMOL no se queja;
    # si es una ruta a un archivo, lo carga como objeto nuevo.
    cmd.load(gpcr_pdb, rec_obj)

    # ---- Cargar voxels ----
    cmd.load(voxel_pdb, vox_obj)

    # ---- Estilo del receptor ----
    cmd.hide("everything", rec_obj)
    cmd.show("cartoon", rec_obj)
    cmd.color(cartoon_color, rec_obj)

    # ---- Esferas para los voxels ----
    cmd.show("spheres", vox_obj)
    cmd.set("sphere_scale", sphere_scale, vox_obj)
    cmd.set("sphere_quality", 1, vox_obj)
    cmd.color(cavity_color, vox_obj)
    cmd.set("sphere_transparency", 0.4, vox_obj)

    # ---- Superficie gaussiana opcional a partir de los voxels ----
    if make_surface:
        # Crear mapa de densidad a partir de la nube de voxels
        # gaussian_sigma ~ "suavizado" del contorno de la cavidad
        # gaussian_buffer ~ cuánto se extiende el mapa alrededor de los voxels
        cmd.map_new(map_name, "gaussian", gaussian_sigma, vox_obj, gaussian_buffer)

        # Crear isosuperficie (valor 1.0 suele ir bien, pero se puede ajustar)
        cmd.isosurface(surf_obj, map_name, 1.0)
        cmd.color(cavity_color, surf_obj)
        cmd.set("surface_quality", 1, surf_obj)
        cmd.set("transparency", 0.6, surf_obj)

        # Si no queremos ver la nube de puntos, la desactivamos
        if not show_spheres:
            cmd.disable(vox_obj)

    # ---- Vista general ----
    cmd.orient(rec_obj)
    cmd.zoom(rec_obj)

# Registrar la función como comando de PyMOL
cmd.extend("show_gpcr_cavity", show_gpcr_cavity)

