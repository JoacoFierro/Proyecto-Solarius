from astropy.io import fits
import matplotlib.pyplot as plt


for i in range(100):
    data = fits.getdata(f'output_fits/noblindeconv_extra/resultado_{(i):05d}.fits') #cargamos un archivo fits
    #guardamos la imagen como png
    plt.imshow(data, cmap='gray')
    plt.axis('off')
    plt.savefig(f'output_png/noblindeconv_extra/resultado_{(i):05d}.png', bbox_inches='tight', pad_inches=0)
    plt.close() 
    
