'''
Plots the normalized absolute value of the bias of triple collocation after CDF matching in the linear Gaussian case
The error model and notation are explained in evaluateCDFBiasLinearGaussian

The bias is normalized by scaling it:
    - with respect to the standard deviation of the noise (normalization = 'noise')
    - with respect to the standard deviation of the signal = scaled anomaly (normalization = 'anomaly')
    - with respect to the standard deviation of the product (normalization = 'product')

The bias is plotted as a function of the SNR of product x and the SNR of products y and z (identical), where 
SNR_i = \frac{\alpha_i^2}{\sigma_i^2}

'''
import numpy as np
from evaluateCDFBiasLinearGaussian import evaluateCDFBiasLinearGaussian
import matplotlib.pyplot as plt
import matplotlib.cm as cm
globfigparams={'fontsize':8,'family':'serif','usetex':True,'preamble':'\usepackage{times}','column_inch':229.8775/72.27,'markersize':24,'markercolour':'#AA00AA'}
plt.rc('font',**{'size':globfigparams['fontsize'],'family':globfigparams['family']})
plt.rcParams['text.usetex']=globfigparams['usetex']
plt.rcParams['text.latex.preamble']=globfigparams['preamble']
plt.rcParams['legend.fontsize']=globfigparams['fontsize']
plt.rcParams['font.size']=globfigparams['fontsize']
width=globfigparams['column_inch']
figprops = dict(facecolor='white',figsize=(width, 0.75*width))
def plotCDFBiasLinearGaussianESNR(SNRx,SNRy,normalization='noise',fnout=None):
    # equal SNR for products y and z
    # unless fnout is None, plot is saved as a pdf with filename fnout
    assert SNRx.shape == SNRy.shape
    alphax=np.ones(SNRx.shape)
    sigma2x=1./SNRx
    D=alphax**2 + sigma2x
    M=np.zeros((SNRx.shape[0],SNRy.shape[0]))
    for jx,alphaxj in enumerate(alphax):
        for jy,SNRyj in enumerate(SNRy):
            alphayj=np.sqrt(SNRyj*D[jx]/(SNRyj+1))
            if normalization == 'anomaly':
                normf=np.sqrt(alphaxj**2)
            elif normalization == 'noise':
                normf=np.sqrt(sigma2x[jx]/alphaxj**2)
            elif normalization == 'product':
                normf=np.sqrt(D[jx])
            M[jy,jx]=100*abs(evaluateCDFBiasLinearGaussian(alphaxj, alphayj, alphayj))/(normf**2)
    fig=plt.figure(**figprops)
    ax = fig.add_axes([0.12, 0.16, 0.85, 0.8]) 
    if normalization == 'anomaly':
        cbarlabel= 'Bias relative to scaled anomaly variance [\%]'
    elif normalization == 'noise':
        cbarlabel='Bias relative to noise variance [\%]'
    elif normalization == 'product':
        cbarlabel='Bias relative to product variance $\\frac{b}{D}$ [\%]'
    cax=plt.imshow(M,cmap=cm.YlOrRd,origin='lower',vmin=0,extent=[SNRx[0],SNRx[-1],SNRy[0],SNRy[-1]])
    plt.xscale('log')
    plt.yscale('log')
    ax.set_xlabel('SNR of product $X$')
    ax.set_ylabel('SNR of products $Y$ and $Z$')
    
    cbar=fig.colorbar(cax,label=cbarlabel)
    if fnout is None:
        plt.show()
    else:
        plt.savefig(fnout,format='pdf')

def plot_all(pathout):
    # plot bias with all normalizations in pathout
    import os.path
    SNRx=np.logspace(-1,2,50)
    SNRy=SNRx
    plotCDFBiasLinearGaussianESNR(SNRx,SNRy,normalization='anomaly',fnout=os.path.join(pathout,'bias_anomaly.pdf'))
    plotCDFBiasLinearGaussianESNR(SNRx,SNRy,normalization='noise',fnout=os.path.join(pathout,'bias_noise.pdf'))
    plotCDFBiasLinearGaussianESNR(SNRx,SNRy,normalization='product',fnout=os.path.join(pathout,'bias_product.pdf'))
