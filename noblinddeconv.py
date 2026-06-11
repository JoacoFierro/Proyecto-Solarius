import re
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.time import Time
from scipy.signal import fftconvolve
from skimage.registration import phase_cross_correlation
from scipy.ndimage import gaussian_filter, shift, laplace
from skimage.restoration import richardson_lucy

# ---------------------------------------------------------------------------
# Constantes de configuracion
# ---------------------------------------------------------------------------

DATASET_DIR   = "dataset"                   
FILE_PREFIX   = "chromosphere_100p"         
OUTPUT_DIR    = "output"        

PERCENTILE_KEEP  = 80                       
UPSAMPLE_FACTOR  = 10                       

BLIND_DECONV_ITERS = 20
BLIND_PSF_SIZE     = 15
CONVERGENCE_TOL    = 1e-5

# ---------------------------------------------------------------------------
# Paso 1 — Descubrimiento de archivos
# ---------------------------------------------------------------------------

def discover_fits_files(dataset_dir: str, prefix: str) -> list[Path]:
    folder = Path(dataset_dir)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"La carpeta de dataset no existe o no es válida: '{folder.resolve()}'")

    glob_pattern = f"{prefix}_*.fits"
    candidates   = list(folder.glob(glob_pattern))

    if not candidates:
        raise ValueError(f"No se encontraron archivos con el patron '{glob_pattern}'.")

    id_regex = re.compile(rf"^{re.escape(prefix)}_(\d+)\.fits$", re.IGNORECASE)

    def extract_id(path: Path) -> int:
        match = id_regex.match(path.name)
        return int(match.group(1)) if match else float("inf")

    valid = sorted([p for p in candidates if extract_id(p) != float("inf")], key=extract_id)
    
    if not valid:
        raise ValueError(f"Ningun archivo paso la validacion del patron numerico.")

    print(f"[1/6] Dataset descubierto: {len(valid)} archivos validos en '{folder.resolve()}'")
    return valid

def load_single_cube(filepath: Path) -> np.ndarray:
    with fits.open(filepath, memmap=False) as hdul:
        data = next((hdu.data for hdu in hdul if hdu.data is not None), None)
        if data is None:
            raise ValueError(f"El archivo no contiene datos: {filepath.name}")

    data = data.astype(np.float32)
    if data.ndim == 2:
        return data[np.newaxis, :, :]            
    if data.ndim == 3:
        return data                              
    raise ValueError(f"Dimensiones inesperadas en {filepath.name}: ndim={data.ndim}")

# ---------------------------------------------------------------------------
# Paso 2 y 3 — Lucky Imaging & Alineación FFT
# ---------------------------------------------------------------------------

def measure_sharpness(frame: np.ndarray) -> float:
    return float(np.var(laplace(frame)))

def lucky_imaging_and_alignment(cube: np.ndarray, percentile: float) -> tuple[np.ndarray, np.ndarray]:
    n_frames = cube.shape[0]
    
    # 2. Selección adaptativa de frames
    sharpness_scores = np.array([measure_sharpness(f) for f in cube])
    threshold = np.percentile(sharpness_scores, percentile)
    
    keep_indices = np.where(sharpness_scores >= threshold)[0]
    best_idx_overall = np.argmax(sharpness_scores)
    reference_frame = cube[best_idx_overall]
    
    print(f"      Umbral ({percentile} pct): {threshold:.2f} | Conservando {len(keep_indices)} de {n_frames} frames.")

    aligned_frames = []
    weights = []
    
    for i, idx in enumerate(keep_indices):
        frame = cube[idx]
        weights.append(sharpness_scores[idx])
        
        if idx == best_idx_overall:
            aligned_frames.append(frame)
            continue
            
        shift_vec, _, _ = phase_cross_correlation(
            reference_frame, frame, 
            upsample_factor=UPSAMPLE_FACTOR, normalization="phase"
        )
        
        aligned_frame = shift(frame, shift=shift_vec, mode='reflect')
        aligned_frames.append(aligned_frame)

    return np.stack(aligned_frames, axis=0), np.array(weights)

# ---------------------------------------------------------------------------
# Paso 4 — Apilado por Promedio Ponderado
# ---------------------------------------------------------------------------

def weighted_average_stack(cube: np.ndarray, weights: np.ndarray) -> np.ndarray:
    print(f"[4/6] Apilando por promedio ponderado según nitidez...")
    
    # 4. Promedio ponderado para preservar información fina
    w_norm = weights / np.sum(weights)
    w_norm = w_norm[:, np.newaxis, np.newaxis]
    
    weighted_stack = np.sum(cube * w_norm, axis=0)
    return weighted_stack.astype(np.float32)

# ---------------------------------------------------------------------------
# Paso 5 — Deconvolución Ciega (Blind Richardson-Lucy vía FFT)
# ---------------------------------------------------------------------------
def standard_deconvolution(
    image: np.ndarray, 
    psf_size: int = 15, 
    sigma: float = 2.0,
    iterations: int = 20
) -> np.ndarray:
    print(f"\n[5/6] Ejecutando Deconvolución Richardson-Lucy (No Ciega)...")

    pad_width = psf_size * 2
    image_padded = np.pad(image, pad_width=pad_width, mode='reflect')
    
    img_min = image_padded.min()
    img_max = image_padded.max()
    Y = (image_padded - img_min) / (img_max - img_min)
    Y = np.clip(Y, 1e-6, 1.0) 
    
    P = np.zeros((psf_size, psf_size), dtype=np.float32)
    P[psf_size//2, psf_size//2] = 1.0
    P = gaussian_filter(P, sigma=sigma)
    P /= P.sum()

    deconv_padded = richardson_lucy(Y, P, num_iter=iterations, clip=False)

    I_cropped = deconv_padded[pad_width:-pad_width, pad_width:-pad_width]
    
    # Desnormalizamos, pero permitimos que el rango fluctúe de manera natural
    # según la energía recuperada. NO usamos np.clip aquí.
    I_restored = (I_cropped * (img_max - img_min)) + img_min
    
    return I_restored.astype(np.float32)

# ---------------------------------------------------------------------------
# Paso 6 — Exportación
# ---------------------------------------------------------------------------

def save_fits_16bit(image: np.ndarray, filepath: Path, n_frames: int, iters_run: int) -> None:
    img_min, img_max = image.min(), image.max()
    
    # Escalar dinámicamente al rango completo de 16-bits
    scaled = (image - img_min) / (img_max - img_min) * 65535.0
    img_uint16 = scaled.astype(np.uint16)

    header = fits.Header()
    header["SIMPLE"]   = True
    header["NAXIS"]    = 2
    header["NAXIS1"]   = image.shape[1]
    header["NAXIS2"]   = image.shape[0]
    header["ORIGIN"]   = "fits_noise_reduction.py"
    header["N_FRAMES"] = n_frames
    header["BD_ITERS"] = iters_run
    header["DATE"]     = Time.now().isot

    # Al pasar img_uint16, Astropy aplica la magia de FITS automáticamente
    primary_hdu = fits.PrimaryHDU(data=img_uint16, header=header)
    fits.HDUList([primary_hdu]).writeto(filepath, overwrite=True)

# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def process_file(file_path: Path, output_dir: Path) -> None:
    print(f"\n--- Procesando: {file_path.name} ---")
    
    # Extraer identificador para el archivo de salida
    file_id = file_path.stem.split('_')[-1]
    output_path = output_dir / f"resultado_{file_id}.fits"

    cube = load_single_cube(file_path)
    
    filtered_cube, weights = lucky_imaging_and_alignment(cube, PERCENTILE_KEEP)
    stacked_image = weighted_average_stack(filtered_cube, weights)
    
    deconv_image = standard_deconvolution(
        image=stacked_image, 
        psf_size=BLIND_PSF_SIZE,
        iterations=BLIND_DECONV_ITERS,
    )

    save_fits_16bit(deconv_image, output_path, n_frames=filtered_cube.shape[0], iters_run=BLIND_DECONV_ITERS)
    print(f"[6/6] Guardado exitosamente: {output_path.name}")

def run_pipeline() -> None:
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    file_paths = discover_fits_files(DATASET_DIR, FILE_PREFIX)
    

    for path in file_paths:
        process_file(path, out_dir)

if __name__ == "__main__":
    run_pipeline()