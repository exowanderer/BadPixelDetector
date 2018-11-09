# darkMed = darks_flat_med_axis0 - np.min(darks_flat_med_axis0)
# darksMed_scaled = darks_flat_med_axis0 / median(darks_flat_med_axis0)# pp.scale(darks_flat_med_axis0)

diff_darks_flat_med_axis0     = np.zeros(darks_flat_med_axis0.size)
diff_darks_flat_med_axis0[1:] = diff(darks_flat_med_axis0)

rtn_features  = []

for iRTN, rtnNow in enumerate(rtn_syn):
    
    rtnNowRescaled = pp.scale((rtnNow - np.min(rtnNow))-(darks_flat_med_axis0 - np.min(darks_flat_med_axis0)))
    _, xhist = np.histogram(rtnNowRescaled, bins=20, normed=True)#, alpha=0.50)
    
    kde1 = sm.nonparametric.KDEUnivariate(rtnNowRescaled)
    kde1.fit(kernel='uni', bw=0.33*median(diff(xhist)), fft=False)
    
    leave1out = np.zeros(nFrames)
    for k in range(nFrames):
        leave1out[k] = np.std(hstack([rtnNow[:k],rtnNow[k+1:]]))
    
    kde3 = sm.nonparametric.KDEUnivariate(pp.scale(np.std(rtnNow) - leave1out))
    kde3.fit(kernel='uni', fft=False)
    
    rtnNowRescaled_KDE = kde1.density
    l1o_Rescaled_KDE   = kde3.density
    
    rtnDiffRescale    = rescale(diff(rtnNow))
    
    flagNow = rtn_loc[iRTN]
    rtn0    = rtnNow - np.min(rtnNow)
    
    rtn0_rescaled = rescale(rtn0 - diff_darks_flat_med_axis0)
    
    avgCnt   = 0
    darksAvg = np.zeros(rtnNow.size)
    if flagNow > 0:
        avgCnt   += 1
        darksAvg += (darks_norm_flat[flagNow-1] - diff_darks_flat_med_axis0) * std(darks_norm_flat[flagNow-1])
        
    if flagNow + 1 < darks_norm_flat.shape[0]:
        avgCnt   += 1
        darksAvg += (darks_norm_flat[flagNow+1] - diff_darks_flat_med_axis0) * std(darks_norm_flat[flagNow+1])
        
    if flagNow > 2048:
        avgCnt   += 1
        darksAvg += (darks_norm_flat[flagNow-2048] - diff_darks_flat_med_axis0) * std(darks_norm_flat[flagNow-2048])
        
    if flagNow + 2048 < darks_norm_flat.shape[0]:
        avgCnt   += 1
        darksAvg += (darks_norm_flat[flagNow+2048] - diff_darks_flat_med_axis0) * std(darks_norm_flat[flagNow+2048])
    
    darksAvg  = darksAvg / avgCnt
    
    darksAvg_rescaled = rescale(darksAvg)
    
    rtn_features.append(hstack([rtn0_rescaled, rtnNowRescaled_KDE, l1o_Rescaled_KDE, leave1out, rtnDiffRescale, darksAvg_rescaled]))
    