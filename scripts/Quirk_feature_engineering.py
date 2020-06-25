# darkMed = darks_flat_med_axis0 - np.min(darks_flat_med_axis0)
# darksMed_scaled = darks_flat_med_axis0 / median(darks_flat_med_axis0)# pp.scale(darks_flat_med_axis0)

diff_darks_flat_med_axis0     = np.zeros(darks_flat_med_axis0.size)
diff_darks_flat_med_axis0[1:] = diff(darks_flat_med_axis0)

quirk_features  = []

for iQuirk, quirkNow in enumerate(quirk_syn):
    
    quirkNowRescaled = pp.scale((quirkNow - np.min(quirkNow))-(darks_flat_med_axis0 - np.min(darks_flat_med_axis0)))
    _, xhist = np.histogram(quirkNowRescaled, bins=20, normed=True)#, alpha=0.50)
    
    kde1 = sm.nonparametric.KDEUnivariate(quirkNowRescaled)
    kde1.fit(kernel='uni', bw=0.33*median(diff(xhist)), fft=False)
    
    leave1out = np.zeros(nFrames)
    for k in range(nFrames):
        leave1out[k] = np.std(hstack([quirkNow[:k],quirkNow[k+1:]]))
    
    leave1out_rescaled  = rescale(leave1out)
    
    kde3 = sm.nonparametric.KDEUnivariate(pp.scale(np.std(quirkNow) - leave1out))
    kde3.fit(kernel='uni', fft=False)
    
    quirkNowRescaled_KDE = rescale(kde1.density)
    l1o_Rescaled_KDE   = rescale(kde3.density)
    
    quirkDiffRescale    = rescale(diff(quirkNow))
    
    flagNow = quirk_loc[iQuirk]
    # print(flagNow)
    quirk0    = quirkNow - np.min(quirkNow)
    
    quirk0_rescaled = rescale(quirk0 - diff_darks_flat_med_axis0)
    
    avgCnt   = 0
    darksAvg = np.zeros(quirkNow.size)
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
    
    quirk_features.append(hstack([quirk0_rescaled, quirkNowRescaled_KDE, l1o_Rescaled_KDE, leave1out_rescaled, quirkDiffRescale, darksAvg_rescaled]))