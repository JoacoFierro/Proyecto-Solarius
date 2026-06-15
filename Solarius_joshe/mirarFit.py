from astropy.io import fits
import matplotlib.pyplot as plt
data = fits.getdata('dataset/data_joshe/chromosphere_100p_00000_MFBD.fits')
#data = fits.getdata('output_fits/noblindeconv_joshe/resultado_00000.fits')
#data = fits.getdata('output/noblindeconv/resultado_00000.fits')
plt.imshow(data, cmap='gray')
plt.show()