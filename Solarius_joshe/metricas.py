import os
import re
import glob
import numpy as np
import pandas as pd
from astropy.io import fits
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.feature import match_template

def procesar_y_normalizar(data):
    """
    Limpia valores NaN/Inf y normaliza la imagen al rango [0, 1].
    """
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    min_val = data.min()
    max_val = data.max()
    
    if max_val == min_val:
        return np.zeros_like(data)
    
    return (data - min_val) / (max_val - min_val)

def alinear_y_recortar(img1, img2):
    """
    Encuentra la imagen más pequeña dentro de la más grande usando Correlación Cruzada.
    Retorna ambas imágenes recortadas para que abarquen exactamente la misma zona.
    """
    # 1. Determinar cuál actúa como 'molde' (template) y cuál como 'lienzo' (search space)
    if img1.size < img2.size:
        template, search_image = img1, img2
        es_img1_template = True
    else:
        template, search_image = img2, img1
        es_img1_template = False

    # Si ya son del mismo tamaño, no hay que alinear
    if template.shape == search_image.shape:
        return img1, img2

    # 2. Ejecutar la Correlación Cruzada Normalizada en 2D
    # Esto desliza el 'template' sobre el 'lienzo' y genera un mapa de calor de coincidencias
    mapa_correlacion = match_template(search_image, template)
    
    # 3. Encontrar las coordenadas (Y, X) del pico máximo de similitud
    y_start, x_start = np.unravel_index(np.argmax(mapa_correlacion), mapa_correlacion.shape)
    
    h, w = template.shape
    
    # 4. Recortar el lienzo para extraer exactamente la zona de coincidencia
    matched_crop = search_image[y_start : y_start + h, x_start : x_start + w]
    
    print(f"   * Coincidencia hallada: offset(Y={y_start}, X={x_start}). Alineación completada.")

    # 5. Devolver las matrices en su orden original para no alterar la referencia y el resultado
    if es_img1_template:
        return template, matched_crop
    else:
        return matched_crop, template

def comparar_directorios(dir_referencia, dir_resultados, csv_salida="resultados_metricas_alineados.csv"):
    resultados = []
    
    patron_resultados = os.path.join(dir_resultados, "resultado_*.fits")
    archivos_resultado = glob.glob(patron_resultados)
    
    if not archivos_resultado:
        print(f"No se encontraron archivos en {dir_resultados}")
        return

    print(f"Se encontraron {len(archivos_resultado)} archivos. Iniciando alineación y comparación...\n")

    for path_res in sorted(archivos_resultado):
        nombre_res = os.path.basename(path_res)
        match = re.search(r"resultado_(\d+)\.fits", nombre_res)
        if not match:
            continue
            
        id_archivo = match.group(1)
        #nombre_ref = f"chromosphere_100p_{id_archivo}_MFBD.fits"
        nombre_ref = f"chromosphere_200p_{id_archivo}_MFBD.fits"
        #nombre_ref = f"continuum_200p_{id_archivo}_MFBD.fits"
        path_ref = os.path.join(dir_referencia, nombre_ref)
        
        if not os.path.exists(path_ref):
            print(f"⚠️ Sin referencia para ID {id_archivo}. Saltando...")
            continue
            
        try:
            # Cargar matrices en punto flotante
            data_ref = fits.getdata(path_ref).astype(np.float32)
            data_res = fits.getdata(path_res).astype(np.float32)
            
            # --- EL CORAZÓN DE LA SOLUCIÓN: ALINEACIÓN POR CORRELACIÓN CRUZADA ---
            data_ref_alineada, data_res_alineada = alinear_y_recortar(data_ref, data_res)
            
            # Normalizar los recortes resultantes
            img_ref_norm = procesar_y_normalizar(data_ref_alineada)
            img_res_norm = procesar_y_normalizar(data_res_alineada)
            
            # Calcular métricas
            mse_val = mean_squared_error(img_ref_norm, img_res_norm)
            psnr_val = peak_signal_noise_ratio(img_ref_norm, img_res_norm, data_range=1.0)
            
            # Ajuste de seguridad dinámico para la ventana de SSIM
            win_size = min(7, img_ref_norm.shape[0], img_ref_norm.shape[1])
            win_size = win_size if win_size % 2 != 0 else win_size - 1
            
            ssim_val = structural_similarity(img_ref_norm, img_res_norm, data_range=1.0, win_size=win_size)
            
            resultados.append({
                "ID": id_archivo,
                "MSE": mse_val,
                "PSNR_dB": psnr_val,
                "SSIM": ssim_val
            })
            print(f"✅ ID {id_archivo} procesado | Tamaño validado: {img_ref_norm.shape} | SSIM: {ssim_val:.4f} | PSNR: {psnr_val:.2f} dB")
            
        except Exception as e:
            print(f"❌ Error crítico en ID {id_archivo}: {e}")

    # Exportar resultados tabulados
    if resultados:
        df = pd.DataFrame(resultados)
        df.to_csv(csv_salida, index=False)
        print(f"\n🎉 Proceso finalizado. Resultados exportados a: {os.path.abspath(csv_salida)}")
        
        print("\n--- Nuevo Resumen Promedio de Métricas ---")
        print(df[["MSE", "PSNR_dB", "SSIM"]].mean().to_string())
    else:
        print("\nNo se completó ninguna iteración válida.")

# ==========================================
# CONFIGURACIÓN DE RUTAS
# ==========================================
if __name__ == "__main__":
    # Reemplaza estas rutas con las carpetas reales en tu sistema
    DIRECTORIO_REFERENCIA = "dataset/data_extra"  
    DIRECTORIO_RESULTADOS = "output_fits/noblindeconv_extra"
    
    comparar_directorios(DIRECTORIO_REFERENCIA, DIRECTORIO_RESULTADOS)