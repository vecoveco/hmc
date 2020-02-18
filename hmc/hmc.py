import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np

def mbf(x, m, a, b):
    """
    Membershipf beta function (MBF) from Dolan and Rutledge 2009
    
    m ::: Center of MBF
    a ::: Half-width of MBF
    b ::: Slope of MBF
    x ::: Variable (e.g. Zh, Zdr, Kdp or T)
    
    return ::: MBF for a Varible x
    
    """
    
    res = ((x - m) / a)**2
    mbf = 1. / (1. + res**b)
           
    return mbf  


def hmc_thompson(zh_ob, zdr_ob, kdp_ob, rho_ob, temp_ob, ml_top, ml_bot, height_ob, band='X'):
    """
    HMC after Thompson et. al 2014
    
    Input ::: 
        PolVar observation:
        zh_ob  = Refl.,
        zdr_ob = Diff. Refl.,
        kdp_ob = Spec. Diff. Phase,
        rho_ob = Cross.Corr.Coeff.,
        
        Other:
        temp_ob = Temp.
        ml_top, ml_bot = BB top,  BB bottom
        band = band of observing system (X, C or S)
        height_ob = observation height 
        
    Output ::: HMC after Thompson et. al 2014
    
    
    Note: The Values in Tab.5. in Thompson et. al 2014 are sometimes wrong
    The correction is done per eye using the MSF-Plots (Fig. 6/7/8)
    """

    # membership functions
    # --------------------

    ### ZH
    mz = np.arange(-20, 70, .5)
    mzh = ctrl.Antecedent(mz, 'refl.')
    mzh_OT = mbf(mzh.universe, 16., 17., 5.)
    mzh_WS = mbf(mzh.universe, 25., 20., 10.)
    mzh_FZ = mbf(mzh.universe, 11.,28., 15. )
    mzh_RN = mbf(mzh.universe, 19., 30., 15.)
    mzh_PL = mbf(mzh.universe, 12., 13., 10.)
    mzh_DN = mbf(mzh.universe, 17., 14., 5.)
    mzh_IC = mbf(mzh.universe, 6., 11., 5.)
    mzh_AG = mbf(mzh.universe, 28., 12., 10.)

    ### ZDR
    md = np.arange(-4, 12, .1)
    mzdr = ctrl.Antecedent(md, 'diff refl')
    mzdr_OT = mbf(mzdr.universe, 0.5, 1.5, 15.)
    mzdr_WS = mbf(mzdr.universe, 3., 5., 10.)
    mzdr_PL = mbf(mzdr.universe, 5.5, 3.7, 10)
    mzdr_DN = mbf(mzdr.universe, 2.6, 1.3, 10.)
    mzdr_IC = mbf(mzdr.universe, 0., 1., 20.)
    mzdr_AG = mbf(mzdr.universe, 0., 1., 20.)
    #mzdr_GR = fuzz.trapmf(mzdr.universe, [-0.3,0, 1, 1.3])

    ### KDP
    mk = np.arange(-1, 4, .01)
    mkdp = ctrl.Antecedent(mk, 'spec phase shift')
    ########################################################## X
    mkdp_PLx = mbf(mkdp.universe, 0.46, 0.45, 5.)
    mkdp_DNx = mbf(mkdp.universe, 1.32, 1.28, 5.)
    mkdp_ICx = mbf(mkdp.universe, 0., 0.55, 5.)
    mkdp_AGx = mbf(mkdp.universe, 0., 0.55, 5)

    ######################################################### C
    mkdp_PLc = mbf(mkdp.universe, 0.27, 0.26, 5.)
    mkdp_DNc = mbf(mkdp.universe, 1.0, 0.7, 5.)
    mkdp_ICc = mbf(mkdp.universe, 0., 0.325, 5.)
    mkdp_AGc = mbf(mkdp.universe, 0., 0.325, 5.)

    ######################################################### S
    mkdp_PLs = mbf(mkdp.universe, 0.13, 0.13, 5.)
    mkdp_DNs = mbf(mkdp.universe, 0.31, 0.3, 5.)
    mkdp_ICs = mbf(mkdp.universe, 0., 0.2, 5)
    mkdp_AGs = mbf(mkdp.universe, 0., 0.2, 5.)

    #### RHO
    mr=np.arange(0, 1.3, .01)
    mrho = ctrl.Antecedent(mr, 'corr coef')
    mrho_OT = mbf(mrho.universe, 0.96, 0.06, 10)
    mrho_WS = mbf(mrho.universe, 0.75, 0.2, 30)
    #mrho_FZ = fuzz.trapmf(mrho.universe, [.95,.97,1,1.01])
    #mrho_RN = fuzz.trapmf(mrho.universe, [.92,.95,1,1.01])
    #mrho_GR = fuzz.trapmf(mrho.universe, [.90,.97,1,1.01])

    ### T
    mt=np.arange(-40, 40, .1)
    mtemp = ctrl.Antecedent(mt, 'corr coef')
    mtemp_FZ = mbf(mtemp.universe, -4., 3., 20.)
    mtemp_RN = mbf(mtemp.universe, 25., 25., 40)
    
    
    # Fuzzy Logic
    # -------------
    
    # ZH Score
    score_zh_WS = fuzz.interp_membership(mz, mzh_WS, zh_ob)
    score_zh_OT = fuzz.interp_membership(mz, mzh_OT, zh_ob)
    score_zh_FZ = fuzz.interp_membership(mz, mzh_FZ, zh_ob)
    score_zh_RN = fuzz.interp_membership(mz, mzh_RN, zh_ob)
    score_zh_PL = fuzz.interp_membership(mz, mzh_PL, zh_ob)
    score_zh_DN = fuzz.interp_membership(mz, mzh_DN, zh_ob)
    score_zh_IC = fuzz.interp_membership(mz, mzh_IC, zh_ob)
    score_zh_AG = fuzz.interp_membership(mz, mzh_AG, zh_ob)
    
    # ZDR Score
    score_zdr_WS = fuzz.interp_membership(md, mzdr_WS, zdr_ob)
    score_zdr_OT = fuzz.interp_membership(md, mzdr_OT, zdr_ob)
    score_zdr_PL = fuzz.interp_membership(md, mzdr_PL, zdr_ob)
    score_zdr_DN = fuzz.interp_membership(md, mzdr_DN, zdr_ob)
    score_zdr_IC = fuzz.interp_membership(md, mzdr_IC, zdr_ob)
    score_zdr_AG = fuzz.interp_membership(md, mzdr_AG, zdr_ob)
    
    # KDP Score
    score_kdp_PLx = fuzz.interp_membership(mk, mkdp_PLx, kdp_ob)
    score_kdp_DNx = fuzz.interp_membership(mk, mkdp_DNx, kdp_ob)
    score_kdp_ICx = fuzz.interp_membership(mk, mkdp_ICx, kdp_ob)
    score_kdp_AGx = fuzz.interp_membership(mk, mkdp_AGx, kdp_ob)
    score_kdp_PLc = fuzz.interp_membership(mk, mkdp_PLc, kdp_ob)
    score_kdp_DNc = fuzz.interp_membership(mk, mkdp_DNc, kdp_ob)
    score_kdp_ICc = fuzz.interp_membership(mk, mkdp_ICc, kdp_ob)
    score_kdp_AGc = fuzz.interp_membership(mk, mkdp_AGc, kdp_ob)
    score_kdp_PLs = fuzz.interp_membership(mk, mkdp_PLs, kdp_ob)
    score_kdp_DNs = fuzz.interp_membership(mk, mkdp_DNs, kdp_ob)
    score_kdp_ICs = fuzz.interp_membership(mk, mkdp_ICs, kdp_ob)
    score_kdp_AGs = fuzz.interp_membership(mk, mkdp_AGs, kdp_ob)

    # RHO Score
    score_rho_WS = fuzz.interp_membership(mr, mrho_WS, rho_ob)
    score_rho_OT = fuzz.interp_membership(mr, mrho_OT, rho_ob)
    
    # Temp Score
    score_temp_FZ = fuzz.interp_membership(mt, mtemp_FZ, temp_ob)
    score_temp_RN = fuzz.interp_membership(mt, mtemp_RN, temp_ob)
    
    ### Aggregation Values
    # Aggregation value for values IN ML (ZH, ZDR, RHO for ML and OT)
    Agg_WS = np.sum([score_zh_WS*.16, score_zdr_WS*.28, score_rho_WS*.56], axis=0) /(.16 + .28 + .56)
    Agg_OT = np.sum([score_zh_OT*.16, score_zdr_OT*.28, score_rho_OT*.56], axis=0) /(.16 + .28 + .56)
        
    # Aggregation value for values BELOW ML (ZH, T for FZ and RN)   
    Agg_FZ = np.sum([score_zh_FZ*.33, score_temp_FZ*.66], axis=0) / (.33 + .66)
    Agg_RN = np.sum([score_zh_RN*.33, score_temp_RN*.66], axis=0) / (.33 + .66)
    
    # Aggregation value for values ABOve ML (ZH, ZDR, KDP for PL, DN, IC, AG)
    ### X-band
    if band=='X':
        print('HMC for X-Band!')
        Agg_PL = np.sum([score_zh_PL*.20, score_zdr_PL*.36, score_kdp_PLx*.44], axis=0) / (.20 + .36 + .44)
        Agg_DN = np.sum([score_zh_DN*.20, score_zdr_DN*.36, score_kdp_DNx*.44], axis=0) / (.20 + .36 + .44)
        Agg_IC = np.sum([score_zh_IC*.48, score_zdr_IC*.24, score_kdp_PLx*.28], axis=0) / (.48 + .24 + .28)
        Agg_AG = np.sum([score_zh_AG*.24, score_zdr_AG*.36, score_kdp_AGx*.40], axis=0) / (.24 + .36 + .40)
    ### C-band
    elif band=='C': 
        print('HMC for C-Band!')
        Agg_PL = np.sum([score_zh_PL*.20, score_zdr_PL*.36, score_kdp_PLc*.44], axis=0) / (.20 + .36 + .44)
        Agg_DN = np.sum([score_zh_DN*.20, score_zdr_DN*.36, score_kdp_DNc*.44], axis=0) / (.20 + .36 + .44)
        Agg_IC = np.sum([score_zh_IC*.48, score_zdr_IC*.24, score_kdp_PLc*.28], axis=0) / (.48 + .24 + .28)
        Agg_AG = np.sum([score_zh_AG*.24, score_zdr_AG*.36, score_kdp_AGc*.40], axis=0) / (.24 + .36 + .40)
    ### S-band
    elif band=='S':
        print('HMC for S-Band!')
        Agg_PL = np.sum([score_zh_PL*.20, score_zdr_PL*.36, score_kdp_PLs*.44], axis=0) / (.20 + .36 + .44)
        Agg_DN = np.sum([score_zh_DN*.20, score_zdr_DN*.36, score_kdp_DNs*.44], axis=0) / (.20 + .36 + .44)
        Agg_IC = np.sum([score_zh_IC*.48, score_zdr_IC*.24, score_kdp_PLs*.28], axis=0) / (.48 + .24 + .28)
        Agg_AG = np.sum([score_zh_AG*.24, score_zdr_AG*.36, score_kdp_AGs*.40], axis=0) / (.24 + .36 + .40)
    else:
        print('Determine S/C/X band!')
        
    Agg_iml = np.array([Agg_WS, Agg_OT])
    Agg_bml = np.array([Agg_FZ, Agg_RN])
    Agg_aml = np.array([Agg_PL, Agg_DN, Agg_IC, Agg_AG])
    
    idx_iml = np.argmax(Agg_iml, axis=0)
    idx_aml = np.argmax(Agg_aml, axis=0)
    idx_bml = np.argmax(Agg_bml, axis=0)
    
    # HMC labels
    labels = ['OT','WS','FZ','RN','PL','DN','IC','AG','NR']
    #Nr = No rain
    hmc_code = [101, 102, 103, 104, 105, 106, 107, 108, 109]
    
    idx_iml[idx_iml==0]=102; idx_iml[idx_iml==1]=101
    idx_bml[idx_bml==0]=103; idx_bml[idx_bml==1]=104
    idx_aml[idx_aml==0]=105; idx_aml[idx_aml==1]=106; idx_aml[idx_aml==2]=107; idx_aml[idx_aml==3]=108
    
    idx_iml[np.isnan(zh_ob)|np.isnan(zdr_ob)|np.isnan(kdp_ob)|np.isnan(rho_ob)|np.isnan(temp_ob)]=109
    idx_bml[np.isnan(zh_ob)|np.isnan(zdr_ob)|np.isnan(kdp_ob)|np.isnan(rho_ob)|np.isnan(temp_ob)]=109
    idx_aml[np.isnan(zh_ob)|np.isnan(zdr_ob)|np.isnan(kdp_ob)|np.isnan(rho_ob)|np.isnan(temp_ob)]=109
    
    idx = idx_iml.copy()
    
    print(idx.shape, height_ob.shape)
    
        
    idx[height_ob<ml_bot]=idx_bml[height_ob<ml_bot]
    idx[height_ob>ml_top]=idx_aml[height_ob>ml_top]
    
    return idx, labels, hmc_code
        
        
    
    
        
    
    
