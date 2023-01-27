import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import shutil
import flopy
import pyemu
import matplotlib.colors as colors
from pathlib import Path
import scripts.utils as utils

#mpl.use("Qt5Agg")

mf2k_exe = utils.mf2k_exe
mfnwt_exe = utils.mfnwt_exe
pestpp_exe = utils.pestpp_exe


def build_from_orig(version="ss", new_d=None, run=True, r_d='..'):
    """
    Build NWT steady state model using the orignal model as the starting point
    """
    # Locations of some things
    orig_model_dir = os.path.join("..", "prev_models", "SthDn_Trans_Scen0", "skinny_wo_glo")
    model_prefix = "SthDn_Trans_Scen0"
    nam_file = model_prefix + ".nam"
    
    # Prep new model dir
    if new_d is None:
        if version == 'ss':
            new_model_dir = os.path.join(r_d, "Dunedin_SS_base")  # new dir to play in
        elif version == 'pred':
            new_model_dir = os.path.join(r_d, "Dunedin_Pred_base")
    else:
        new_model_dir = new_d
    if os.path.exists(new_model_dir):
        shutil.rmtree(new_model_dir)
    os.mkdir(new_model_dir)
    
    # Load original
    print("Loading orginal model")
    m_org = flopy.modflow.Modflow.load(f=nam_file, model_ws=orig_model_dir, 
                                       version='mf2k',
                                       exe_name=mf2k_exe, verbose=False, 
                                       check=True)
    drn_o = m_org.drn
    wel_o = m_org.wel
    if version == "ss":
        modelname = "SouthDun_SS"
        nper = 1
        perlen = 1.
        nstp = 1
        steady = True
        drnspd = {0: drn_o.stress_period_data.data[0]}  # setup stress period data as {kper:data} dictionary
        welspd = {0: wel_o.stress_period_data.data[0]}  # setup stress period data as {kper:data} dictionary
    elif version == 'pred':
        modelname = "SouthDun_100"
        nper = m_org.dis.nper
        perlen = m_org.dis.perlen
        nstp = m_org.dis.nstp
        drnspd = drn_o.stress_period_data.data
        welspd = wel_o.stress_period_data.data
        steady = m_org.dis.steady
    else:
        raise(ValueError, "Only version 'ss' or 'pred' supported at the mo.")
        
        
    
    print(f"Building NWT {version} model in {new_model_dir}")
    # instantiate flopy object
    m = flopy.modflow.Modflow(modelname, 
                              version='mfnwt', 
                              exe_name=str(Path(new_model_dir, mfnwt_exe.name)),
                              model_ws=new_model_dir,
                              external_path='.')
    
    # Add DIS
    dis_o = m_org.dis   
   
    # Old model top
    excel_file = os.path.join("..", "data", "model_top", 
                          "model_pointvalues_watertable.xls"
                         )
    nmtp = pd.read_excel(excel_file, header=0, index_col=0).dropna(subset=['LiDAR_DEM_VD2016'])
    
    # create an array from these obs -- this would be different if multiple layers
    cur_tp = dis_o.top.array # current model top
    new_tp = cur_tp.copy()
    new_tp[tuple(nmtp[['row','cols']].T.values)] = (nmtp.LiDAR_DEM_VD2016 + 100.377)
    new_thick = new_tp - dis_o.botm.array[0]
    new_bot = dis_o.botm.array.copy()
    new_bot[0][new_thick<=10.] = new_tp[new_thick<=10.] - 10.
    assert (new_tp - new_bot[0]).min() >= 10.
    dis = flopy.modflow.ModflowDis(m,
                                   nlay=dis_o.nlay,
                                   nrow=dis_o.nrow,
                                   ncol=dis_o.ncol,
                                   nper=nper, # SS
                                   delr=dis_o.delr.array,
                                   delc=dis_o.delc.array,
                                   top=new_tp,
                                   botm=new_bot,
                                   perlen=perlen,
                                   nstp=nstp,
                                   steady=steady,  # SS
                                   itmuni=dis_o.itmuni,
                                   lenuni=dis_o.lenuni,
                                   start_datetime='01-01-2009'  # to accomodate SS period in predictive model
                                   )
    
    # setup spatial reference
    # currently using flopy 3.3.3. so this spatial referencing works:
    try:
        m.modelgrid = m.modelgrid.from_gridspec(
            os.path.join("..", "prev_models","SthDn_Trans_Scen0",
                     "skinny_wo_glo", "SthDn_Trans_mod1.spc")
        )
        m.modelgrid.set_coord_info(epsg=2193)
    except:
        m.sr = flopy.utils.reference.SpatialReference.from_gridspec(
            os.path.join("..","prev_models","SthDn_Trans_Scen0",
                     "skinny_wo_glo","SthDn_Trans_mod1.spc")
        )
        m.modelgrid.set_coord_info(xoff=m.sr.xll, yoff=m.sr.yll, epsg=2193,
                                   proj4=m.sr.proj4_str, angrot=-58.0)


    # Add BAS 
    bas_o = m_org.bas6
    bas = flopy.modflow.ModflowBas(m, stoper=2,
                                   ibound=bas_o.ibound.array,
                                   strt=dis_o.top.array,  # set to model top
                                   hnoflo=bas_o.hnoflo,
                                  )

    # Add UPW
    lpf_o = m_org.lpf
    sc_zones = os.path.join("..", "data", "k_zones", "model_pointvalues_Kstructure_2022.xlsx")
    kzone_df = pd.read_excel(sc_zones)
    kzone = np.zeros_like(lpf_o.hk.array)
    kzone[tuple(kzone_df.loc[:, ['layer', 'row', 'cols']].values.T)] = kzone_df.domain
    kzone = kzone.astype(int)
    plt.figure()
    plt.imshow(kzone[0])
    kzonemap = kzone_df.groupby('domain')[['logK1', "logK2", "logK3"]].mean()
    kzonemap = kzonemap.mask(kzonemap < -20).mean(axis=1)-kzonemap.mask(kzonemap < -20).mean(axis=1)[5]
    kzonemap = pd.concat([kzonemap,
                          lpf_o.hk.array[0][0][0] * 10 ** kzonemap], axis=1)
    kzonemap.columns = ['rellog10', 'k']
    kzonemap.loc[6, 'k'] = 3.5 #re-setting dunes because modelling sucks
    karray = lpf_o.hk.array
    for z, k in kzonemap.k.to_dict().items():
        karray[kzone==z] = k
    karray[karray > 100] = 100
    plt.figure()
    plt.imshow(karray[0])
    upw = flopy.modflow.ModflowUpw(m,
                                   ipakcb=lpf_o.ipakcb,
                                   hdry=lpf_o.hdry,
                                   iphdry=0,
                                   laytyp=lpf_o.laytyp.array,
                                   layavg=lpf_o.layavg.array,
                                   chani=lpf_o.chani.array,
                                   hani=lpf_o.hani.array,
                                   layvka=lpf_o.layvka.array,
                                   hk=karray,
                                   vka=lpf_o.vka.array,
                                   ss=lpf_o.ss.array,
                                   sy=lpf_o.sy.array,
                                  )
    
    # Add Drains 
    drn = flopy.modflow.ModflowDrn(m,
                                   ipakcb=drn_o.ipakcb,
                                   stress_period_data=drnspd,
                                  )
    # Add Wells
    wel = flopy.modflow.ModflowWel(m,
                                   ipakcb=drn_o.ipakcb,
                                   stress_period_data=welspd,
                                  )
    
    # CONVERT OLD RIV PACKAGE INTO GHB
    riv = m_org.riv
    
    # make copy of first kper from original
    rivdata = riv.stress_period_data.data.copy()
    
    # need ghb data in slightly different format (bhead instead of stage and no rbot)
    # mapping dictionary
    name_map = {'stage' : 'bhead'}
    
    # list comprehension
    rivspd = {}
    rdatalist = [name_map[n] if n in name_map.keys() else n 
                 for n in rivdata[0].dtype.names]
    
    for kper, data in rivdata.items():
        
        # Apply to  
        data.dtype.names = rdatalist
        ghbdf = pd.DataFrame.from_records(data[['k', 'i', 'j', 'bhead', 'cond']])
        #     plt.figure()
        #     plt.plot(np.sort(ghbdf.j.unique()), ls='', marker='x') # habourside and south dun coast can be differenciated by j value

        ghbdf1 = ghbdf.loc[ghbdf.j<40].copy() # harbourside
        ghbdf2 = ghbdf.loc[ghbdf.j>=40].copy() # southcoast
        # group by columns and get last row (max i)
        harghb = ghbdf1.groupby('j').max().reset_index()[['k', 'i', 'j', 'bhead', 'cond']]
        # need to update harbourside boundary condition
        # drop incorrect GHB cells at harbour boundary condition
        #harghb_drop = harghb.drop([0,1,2,3,4,5,32,33,34],axis=0,inplace=False)
        # and then add in new GHB cells
        #new_ghb = pd.DataFrame({"k":[0,0,0,0], "i":[0,1,2,19], "j":[37,36,36,5],
        #                    "bhead":[100.199997,100.199997,100.199997,100.199997],
        #                    "cond":[16000.0,16000.0,16000.0,16000.0]},
        #                   index=["0","1","2","3"])
        #updated_harghb = pd.concat([harghb, new_ghb], ignore_index=True)
        #new_ghb = pd.DataFrame({"k":[0,0,0,0,0,0,0,0,0,0,0,0,0], "i":[17,18,10,9,7,6,5,4,4,3,2,1,0], "j":[1,3,32,33,35,35,35,35,36,36,36,36,36],
        #                    "bhead":[100.199997,100.199997,100.199997,100.199997,100.199997,100.199997,100.199997,100.199997,100.199997,100.199997,100.199997,100.199997,100.199997],
        #                    "cond":[16000.0,16000.0,16000.0,16000.0,16000.0,16000.0,16000.0,16000.0,16000.0,16000.0,16000.0,16000.0,16000.0]},
        #                   index=["0","1","2","3","4","5","6","7","8","9","10","11","12"])
        #updated_harghb = pd.concat([harghb_drop, new_ghb], ignore_index=True)
        # group by rows and get first column (max j)
        scghb = ghbdf2.groupby('i').min().reset_index()[['k', 'i', 'j', 'bhead', 'cond']]
        # bring two dataframes together
        fulghb = pd.concat([harghb, scghb], ignore_index=True)
        #fulghb = pd.concat([updated_harghb, scghb], ignore_index=True)
        # get riv cells that can now be inactive -- this will retun a pandas multi index
        newinact = ghbdf.set_index(['k', 'i', 'j']).index.difference(
            fulghb.set_index(['k', 'i', 'j']).index)
        # but have updated GHB boundary condition and now have more inactive cells in addition to old river package
        #inact_df = pd.DataFrame([[0, 17, 1], [0,18,3],[0, 10, 32], [0,9,33],[0, 8, 34], 
        #                         [0,7,34], [0, 6, 34], [0, 5, 34],[0, 7, 35], [0, 6, 35],
        #                         [0, 5, 35], [0, 4, 35],[0, 3, 35], [0, 2, 35], [0, 1, 35],
        #                         [0, 0, 35], [0, 3, 36],[0, 4, 36], [0, 5, 36], [0, 6, 36],[0, 0, 36],
        #                         [0,18,4],[0,18,5]
        #                        ],columns=["k", "i", "j"])
        #new_newinact = pd.MultiIndex.from_frame(inact_df)
        #newinact = newinact.union(new_newinact)
        # convert datatypes because flopy is annoying
        fulghb.loc[:, 'cond'] = 800.  # drop to 800 m2/d as per Lee's calcs
        fulghb = fulghb.astype(
            dict([('k', int),
                 ('i', int),
                 ('j', int),
                 ("bhead", np.float32),
                 ("cond", np.float32)]))
        # convert to record array for flopy
        fulghb = fulghb.to_records(index=False)
        rivspd[kper] = fulghb

    # plotting last kper drain elevation vs model top
    plt.figure()
    df = ghbdf
    ar = np.zeros([m_org.modelgrid.nrow, m_org.modelgrid.ncol])
    ar[tuple(df[['i','j']].T.values)] = (
            df.bhead - m_org.dis.top.array[tuple(df[['i','j']].T.values)])
    plt.imshow(ar, interpolation='none', norm=colors.TwoSlopeNorm(vcenter=0),
               cmap='coolwarm')
    plt.title("rivstage - model top (m)")
    plt.colorbar()

    if version == 'ss':
        rivspd = {0: rivspd[0]}
    ghb = flopy.modflow.ModflowGhb(m,
                                   ipakcb=riv.ipakcb,
                                   stress_period_data=rivspd,
                                  )
    # Modify ibound to take into account new ghb def
    ib = m.bas6.ibound.array.copy()

    ib[tuple(newinact.to_frame().values.T)] = 0

    m.bas6.ibound = ib

    # Add RCH
    rch_o = m_org.rch
    rchshpe = rch_o.rech.array.shape
    #if os.path.exists(os.path.join("..","data","Binary Thresholding_NDVI_.tif")):
    #    imperv_fname = os.path.join("..","data","Binary Thresholding_NDVI_.tif")
    #else:
    #    imperv_fname = os.path.join(r"I:\\Groundwater","Research","NZSeaRise_SouthDunedin","v2_data")
    #try:
    #    rchmult = utils.sample_raster_at_modelgrid(
    #    imperv_fname, # if not local check gnsshared\Groundwater\Research\NZSeaRise_SouthDunedin\v2_data
    #    m,
    #    savename=os.path.join("..", "data", "recharge_imp_mults.dat")
    #)
    #except ImportError:
    rchmult = np.loadtxt(os.path.join("..", "data", "recharge_imp_mults.dat"))
    
    if version == 'ss':
        rech = 0.8 * rchmult * (0.673/365.25)  ### estimated recharge, remember rchmult is imperviousness estimate for SouthD
    elif version == 'pred':
        rech = 0.8 * rchmult * (0.673/365.25)  ### estimated recharge, remember rchmult is imperviousness estimate for SouthD

    rch = flopy.modflow.ModflowRch(m,
                                   nrchop=rch_o.nrchop,
                                   ipakcb=None,
                                   rech={kper:rech for kper in range(m.nper)},
                                   irch=rch_o.irch
                                  )
    
    # Add NWT
    pcg = m_org.pcg
    nwt = flopy.modflow.ModflowNwt(m,
                                   headtol=pcg.hclose,
                                   fluxtol=500,
                                   maxiterout=100,
                                   thickfact=1e-5,
                                   linmeth=1,
                                   options=['MODERATE'],
                                   iprnwt=1
                                  )
    
    # Add Output control
    opts = ['save head', 'save budget', 'print budget']
    ocspd = {(i, int(stp-1)): opts for i, stp in enumerate(m.dis.nstp)}
    oc = flopy.modflow.ModflowOc(m, stress_period_data=ocspd)
    print("Writing model")
    m.write_input()
    shutil.copy(mfnwt_exe, m.model_ws)
    # plt.show()
    
    if run:
        m.run_model()
    return m
    
 
def setup_pst(new_m, nreals=1000, vis_cov=False, r_d='..'):
    # SETUP PEST INTERFACE
    # define spatial ref
    # TODO these "version" definitions may need refining
    #  for alternative history and predicitve model temporal discretisation
    try:
        sr = pyemu.helpers.SpatialReference(
            delr=new_m.dis.delr.array, delc=new_m.dis.delc.array, 
            xll=new_m.modelgrid.xoffset, yll=new_m.modelgrid.yoffset
        )
        if len(new_m.dis.perlen) == 1:
            period = 'hist'
            version = 'ss'
        elif len(new_m.dis.perlen) == 101:
            period = 'proj'
            version = '101'
        modelname = new_m.name
    except AttributeError:
        if 'ss' in new_m.lower():
            period = 'hist'
            modelname = "SouthDun_SS"
            version = 'ss'
        elif 'pred' in new_m.lower():
            period = 'proj'
            modelname = "SouthDun_100"
            version = '101'
        else:
            raise ValueError("model or model string (new_m) argument not understood")
        new_m = flopy.modflow.Modflow.load(
            f=f"{modelname}.nam", 
            model_ws=os.path.join(r_d, new_m), 
            version='mfnwt',
            exe_name='mfnwt.exe', 
            verbose=True, 
            check=True
        )
        sr = pyemu.helpers.SpatialReference(
            delr=new_m.dis.delr.array, delc=new_m.dis.delc.array, 
            xll=new_m.modelgrid.xoffset, yll=new_m.modelgrid.yoffset
        )
    t_d = f"template_{period}_{version}"
    startdatetime = new_m.start_datetime  #
    dts = (pd.to_datetime(startdatetime) +
           pd.to_timedelta(np.cumsum(new_m.dis.perlen.array), unit='d'))
    
    # Instantiate PstFrom object
    pf = pyemu.utils.PstFrom(
        original_d=new_m.model_ws, 
        new_d=os.path.join(r_d, t_d), 
        spatial_reference=sr,
        start_datetime=new_m.start_datetime,
        zero_based=False, # the model is 1-based (MODFLOW start counting at 1) 
        remove_existing=True,
                    )
    
    # get active model domain (just cos it is handy to ave on a two letter variable)
    ib = new_m.bas6.ibound.array
    
    # Add some Pars
    # drn cond pars
    # define a variagram and geostruct for drain conductance covariance 
    v = pyemu.geostats.ExpVario(contribution=1.0,a=100)
    drn_gs = pyemu.geostats.GeoStruct(variograms=v, transform='log')
    drn_files = [f for f in os.listdir(pf.new_d) if "DRN" in f and "dat" in f]
    
    # grid based drain conductance
    pf.add_parameters(drn_files, 
                      par_type='grid', # cell-by-cell 
                      lower_bound=0.1, # par ranges define variance
                      upper_bound=10.,
                      index_cols=[0,1,2], # tabular list-like input for drains so need these
                      use_cols=[4], # column that relates to conductance
                      par_name_base='drncond-gr',
                      pargp='drncond-gr',
                      geostruct=drn_gs
                     )
    
    # global constant drain conductance
    pf.add_parameters(drn_files, 
                      par_type='constant', # 1 additional global par
                      lower_bound=0.1, 
                      upper_bound=10.,
                      index_cols=[0,1,2],
                      use_cols=[4],
                      par_name_base='drncond-cn',
                      pargp='dcond-cn'
                     )
    
    # drn elev pars
    # grid based drain elevation
    # define a variagram and geostruct for drain conductance covariance 
    v = pyemu.geostats.ExpVario(contribution=1.0,a=100)
    drnelev_gs = pyemu.geostats.GeoStruct(variograms=v, transform='none')
    pf.add_parameters(drn_files, 
                  par_type='grid', # 1 additional global par
                  par_style='add',
                  transform='none',
                  lower_bound=-0.5, 
                  upper_bound=0.5,
                  index_cols=[0,1,2],
                  use_cols=[3],
                  par_name_base='drnelev-gr',
                  pargp='delev-gr',
                  geostruct=drnelev_gs
                 )
    # constant based drain elevation
    pf.add_parameters(drn_files, 
                      par_type='constant', # 1 additional global par
                      par_style='add',
                      transform='none',
                      lower_bound=-0.5, 
                      upper_bound=0.5,
                      index_cols=[0,1,2],
                      use_cols=[3],
                      par_name_base='drnelev-cn',
                      pargp='delev-cn'
                     )
    
    # K pars
    # different variogram for k
    v = pyemu.geostats.ExpVario(contribution=1.0,a=400)
    k_gs = pyemu.geostats.GeoStruct(variograms=v, transform='log')
    # single layer
    # grid based HK
    pf.add_parameters("hk_layer_1.ref", # array like file
                      par_type='grid', 
                      lower_bound=0.01, 
                      upper_bound=100.,
                      par_name_base='hk-gr',
                      pargp='hk-gr',
                      geostruct=k_gs,
                      zone_array=ib[0]  # zone controls only active cells
                     )
    # constant HK
    pf.add_parameters("hk_layer_1.ref", 
                      par_type='constant', 
                      lower_bound=0.01, 
                      upper_bound=100.,
                      par_name_base='hk-cn',
                      pargp='hk-cn',
                      zone_array=ib[0]
                     )

    # rch pars
    # get all recharge array files for model
    rch_files = [f for f in os.listdir(pf.new_d) if "rech" in f and "ref" in f]
    # grid based recharge (applied to all kper)
    pf.add_parameters(rch_files, 
                      par_type='grid', 
                      lower_bound=0.5, # narrower uncertainty for recharge
                      upper_bound=2,
                      par_name_base='sprch-gr',
                      pargp='sprch-gr',
                      geostruct=k_gs,
                      zone_array=ib[0]
                     )
    # constant recharge (applied to all kper)
    pf.add_parameters(rch_files, 
                      par_type='constant', 
                      lower_bound=0.5, 
                      upper_bound=2,
                      par_name_base='sprch-cn',
                      pargp='sprch-cn',
                      zone_array=ib[0] 
                     )

    # ghb cond pars
    # collect all ghb kper files
    ghb_files = [f for f in os.listdir(pf.new_d) if "GHB" in f and f.endswith("dat")]
    # grid based conductance across all kper
    pf.add_parameters(ghb_files,  # similar to drains (similar list-like file)
                      par_type='grid', 
                      lower_bound=0.01, 
                      upper_bound=100.,
                      index_cols=[0, 1, 2],
                      use_cols=[4],
                      par_name_base='ghbcond-gr',
                      pargp='ghbcond-gr',
                      geostruct=k_gs
                      )

    # let's seperate South Coast and harbour boundary as will apply different priors
    # two independent constants split habourside to scouthcoast
    f = ghb_files[0]
    # get row numbers in ghb that relate to harbourside and southcoast
    ghb = pd.read_csv(os.path.join(pf.new_d, f), header=None, delim_whitespace=True)
    ghb.columns = ['k', 'i', 'j', 'head', 'cond']
    # harbourside where j < 40 (zerobased)
    hbghb = (ghb.loc[ghb.j<41, ['k','i','j']].values).tolist()
    scghb = (ghb.loc[ghb.j>=41, ['k','i','j']].values).tolist()
    # harbourside ghb conductance constant
    pf.add_parameters(ghb_files,
                      use_rows=hbghb,
                      par_type='constant',
                      lower_bound=0.001,
                      upper_bound=1000.,
                      index_cols=[0, 1, 2],
                      use_cols=[4],
                      par_name_base='ghb-hb-cond',
                      pargp='ghb-hb-cond'
                      )
    # south coast ghb conducatnce constant
    pf.add_parameters(ghb_files,
                      use_rows=scghb,
                      par_type='constant', 
                      lower_bound=0.01, 
                      upper_bound=100.,
                      index_cols=[0, 1, 2],
                      use_cols=[4],
                      par_name_base='ghb-sc-cond',
                      pargp='ghb-sc-cond'
                      )
    # possible temporal sea boundary elevation uncertainty
    if period == 'proj':
        # temporal recharge pars
        for f in rch_files:
            kper = f.split('.')[0].split('_')[-1]
            # spatially constant recharge parameter for each kper
            if kper != '0': # ignore first stress period            
                pf.add_parameters(f,
                                  par_type='constant',
                                  lower_bound=0.8,
                                  upper_bound=1.25,
                                  par_name_base=f'tmprch_kper:{kper}',
                                  pargp='tmprch',
                                  zone_array=ib[0]
                                 )
        # GHB projection
        # temporal correlation on projection ghb
        tv = pyemu.geostats.ExpVario(contribution=1.0, a=730.5)
        t_gs = pyemu.geostats.GeoStruct(variograms=tv, transform='log')
        # set up pars on scenario file...
        # check first that it exists -- ok that it is just a dummy file for now
        if not os.path.exists(os.path.join(pf.new_d, "scenario.csv")):
            attach_scenario(t_d, run=False)
        scendf = pd.read_csv(os.path.join(pf.new_d, "scenario.csv"))
        pf.add_parameters("scenario.csv",
                          par_type='constant',
                          lower_bound=0.41,
                          upper_bound=2.47,
                          index_cols=['Year', 'deltadays'],
                          use_cols=['0.5'],
                          par_name_base='ghbelev-cn',
                          pargp='ghbelev-cn'
                          )
        # need to trick pyemu here into doing temporal correlation on one file
        # need multiple calls to file
        for i, s in scendf.iterrows():
            userows = s.loc[['Year', 'deltadays']].tolist()
            pf.add_parameters("scenario.csv",
                              par_type='constant',
                              use_rows=[userows],
                              lower_bound=0.925,
                              upper_bound=1.075,
                              index_cols=['Year', 'deltadays'],
                              use_cols=['0.5'],
                              par_name_base=f'ghbelev-tmp_kper:{i+1}',
                              pargp='ghbelev-tmp',
                              geostruct=t_gs,
                              datetime=s.Year
                              )
        # need to add function that will assign scenario at forward model runtime
        pf.add_py_function(
            os.path.join(r_d, 'scripts', "utils.py"),
            call_str=f"scenario2ghbdat()",
            is_pre_cmd=True # will run before model
        )
        
        # SY
        pf.add_parameters("sy_layer_1.ref",  # array like file
                          par_type='grid',
                          lower_bound=0.2,
                          upper_bound=5.,
                          par_name_base='sy-gr',
                          pargp='sy-gr',
                          geostruct=k_gs,
                          zone_array=ib[0]  # zone controls only active cells
                          )
        # constant sy
        pf.add_parameters("sy_layer_1.ref",
                          par_type='constant',
                          lower_bound=0.5,
                          upper_bound=2.,
                          par_name_base='sy-cn',
                          pargp='sy-cn',
                          zone_array=ib[0],
                          ult_ubound=0.4
                          )
        # SS
        pf.add_parameters("ss_layer_1.ref",  # array like file
                          par_type='grid',
                          lower_bound=0.01,
                          upper_bound=100.,
                          par_name_base='ss-gr',
                          pargp='ss-gr',
                          geostruct=k_gs,
                          zone_array=ib[0]  # zone controls only active cells
                          )
        # constant ss
        pf.add_parameters("ss_layer_1.ref",
                          par_type='constant',
                          lower_bound=0.01,
                          upper_bound=100.,
                          par_name_base='ss-cn',
                          pargp='ss-cn',
                          zone_array=ib[0]
                          )
    
    # ADD OBS/OUTPUTS
    # prep actualy head observation data 
    obs_locs = pd.read_csv(os.path.join("..", "obs_locs", "Cox_Operational_Targets.dat"),
                       index_col=0,delim_whitespace=True)
    # loop over dataframe rows, get ij corresponding to well location
    obs_locs[['i','j']] = obs_locs.apply(lambda x: 
                                         pd.Series(new_m.modelgrid.intersect(x.Easting, x.Northing)), 
                                         axis=1).astype(int)
    obs_locs["k"] = 0  # BIG TODO when we have multi layer!!!!!!!!!!!!!!! (will need wel depths too!)
    obs_locs["sitename"] = obs_locs.index.map(lambda x: 
                                            x.replace('(','').replace(')','').replace('_','-'))
    obs_locs.to_csv(os.path.join(pf.new_d, "Cox_Operational_Targets.dat"))
    
    # convert binary output to nice tabular list file that we can pass to pstfrom
    hdobsnme, wlobsnme = utils.hds_bin2csv(d=pf.new_d, 
                                           modelfname=f"{modelname}.nam")
    # Add this file as outputs to track
    # can be as array files if we want
    # pf.add_observations("hds_kper0_lay0.csv",  prefix="hd")
    # or whole tabulated model output
    pf.add_observations(
        hdobsnme, ## "hds.csv"
        prefix="hd", 
        index_cols=['k', 'i', 'j', 'kper'], 
        use_cols='obsval',
        ofile_sep=','
    )
    if period == 'hist':
        # add less-than inequality as obs too (only for history model)
        pf.add_observations(
            hdobsnme, ## "hds.csv"
            insfile="lthds.csv.ins",
            prefix="less_hd",
            index_cols=['k', 'i', 'j', 'kper'],
            use_cols='obsval',
            ofile_sep=',',
            obsgp="less_hd"
        )
    pf.add_observations(
        wlobsnme, ## "welobs.csv"
        prefix="wl", 
        index_cols=["sitename", 'k', 'i', 'j', 'kper'], 
        use_cols='obsval',
        ofile_sep=','
    )
    # need to add the above function to our forward run script so that 
    # it is also run at model run time
    pf.add_py_function(
        os.path.join(r_d, 'scripts', "utils.py"), 
        call_str=f"hds_bin2csv('.', modelfname='{modelname}.nam')",
        is_pre_cmd=False
    )
    
    # TODO ADD MORE OBS
    # DRAIN FLUXES
    txt = "DRAINS"
    drnfobsnme, sumfobsme = utils.cbc_bin2csv(text="DRAINS", d=pf.new_d,
                                              modelfname=f"{modelname}.nam")
    pf.add_observations(
        drnfobsnme, ## "drains.csv"
        prefix="drn", 
        index_cols=['k', 'i', 'j', 'kper'], 
        use_cols='q',
        ofile_sep=','
    )
    pf.add_observations(
        sumfobsme, ## "drains_sum.csv"
        prefix="drnsum", 
        index_cols=['kper'], 
        use_cols='sum',
        ofile_sep=','
    )
    pf.add_py_function(
        os.path.join(r_d, 'scripts', "utils.py"), 
        call_str=f"cbc_bin2csv(text='{txt}', d='.', modelfname='{modelname}.nam')",
        is_pre_cmd=False
    )
    # OFFSHORE FLUXES
    txt = "HEAD DEP BOUNDS"
    ghbfobsnme, sumfobsme = utils.cbc_bin2csv(text=txt, d=pf.new_d, 
                                              modelfname=f"{modelname}.nam")
    pf.add_observations(
        ghbfobsnme, ## "head_dep_bounds.csv"
        prefix="ghb", 
        index_cols=['k', 'i', 'j', 'kper'], 
        use_cols='q',
        ofile_sep=','
    )
    pf.add_observations(
        sumfobsme, ## "head_dep_bounds_sum.csv"
        prefix="ghbsum", 
        index_cols=['kper'], 
        use_cols='sum',
        ofile_sep=','
    )
    #add ghb as obs
    for f in ghb_files:
        kper = int(f.strip('.dat').strip("GHB_"))
        pf.add_observations(f,
                            prefix=f'ghb-hd_kper:{kper}',
                            index_cols=[0, 1, 2],
                            use_cols=[3],
                            obsgp='ghb-hd',
                            includes_header=False
                            )
    
    pf.post_py_cmds.append(f"cbc_bin2csv(text='{txt}', d='.', modelfname='{modelname}.nam')")
        
    # make sure flopy is imported at run time
    pf.extra_py_imports.append("flopy")
    # make sure model output files are removed before each run
    pf.tmp_files.extend(new_m.output_fnames)
    
    # add model run command (DEFO ESSENTIAL!)
    pf.mod_sys_cmds.append(f'mfnwt {new_m.namefile}')
    
    # build PEST control file (for the first time)
    pst = pf.build_pst()

    obs = pst.observation_data
    # set all weights initially to zero
    obs.loc[:, "weight"] = 0.

    if period == 'hist':
        # WEIGHT OBS !!!!!!!!!!!!!!!!!!!!
        # define and weight total drain flux
        dsumob = obs.obsnme.str.startswith('oname:drnsum')
        obs.loc[dsumob, "obsval"] = -2000  # Wastewater 400-500 m3d-1 + other?
        obs.loc[dsumob, "weight"] = 1/500  # 1/sigma
        
        # TODO actual level obs weights
        ##### NOT USING LONG TERM DATA FOR OBSERVATION VALUES ANYMORE - JUST FOCUSSING ON 2019-2021 OBS
        # annoying that the ids in the different dataframs are different but think we want
        # split out longer term obs from datasets 
        #ltsites = ["Bathgate", "Tonga_Park_Shallow", "Culling_Park", "Kennedy"]
        # extracting obs from timeseries
        #ltobs, _ = utils.get_lta_wl_obs(os.path.join("..", "data","water_levels"),
        #                          obs_locs=obs_locs, 
        #                          plot=False, 
        #                          ltsites = ltsites)
        # setting obs values
        #allwelobs = obs.loc[welobs]
        #wl2019obs = obs_locs.drop(ltsites)
        #wl2019sel = allwelobs.sitename.apply(lambda x: x in wl2019obs.sitename.str.lower().values)
        #wl2019obssel = allwelobs.loc[wl2019sel].index

        # set all that are in 2019-2021 data
        welobs = obs.obgnme == 'oname:wl_otype:lst_usecol:obsval'
        obs.loc[welobs , "obsval"] = obs_locs.set_index(obs_locs.sitename.str.lower()
                                                       ).loc[obs.loc[welobs].sitename, 
                                                             'obs_19-21'].values
        obs.loc[welobs, "weight"] = 1/.15  # lower weight?
        
        # fitzroy is garbage
        allwelobs = obs.loc[welobs]
        fitzroy = allwelobs[allwelobs.sitename.str.contains('fitzroy')].index
        obs.loc[fitzroy, "weight"] = 1/.5  # lower weight?
        # setting obs values for longer term data
        
        # get observation names that relate to long terms sites
        #ltsel = allwelobs.sitename.apply(lambda x: x in ltobs.sitename.str.lower().values)
        #ltobssel = allwelobs.loc[ltsel].index
        # assign vals and weights
        #obs.loc[ltobssel , "obsval"] = ltobs.set_index(ltobs.sitename.str.lower()
        #                                               ).loc[obs.loc[ltobssel].sitename, 
        #                                                     'median'].values
        #obs.loc[ltobssel, "weight"] = 1/.10 # higher weight
        
        # weight and add obs val to inequality less than obs
        lessobs = obs.loc[obs.obgnme.str.contains("less_hd")].astype({'i':int, 'j':int})
        # read in model top as these will be our observation values
        top = np.loadtxt(os.path.join(pf.new_d, "model_top.ref"))
        obs.loc[lessobs.index, 'obsval'] = top[tuple(lessobs[['i', 'j']].values.T)]
        obs.loc[lessobs.index, "weight"] = 1/2.5 
        
    if vis_cov:
        # build prior covariance matrix (from parameter bounds and assigned geostructs)
        # NOTE: This is not actually srictly necessary -- we can pf.draw() to draw realisations 
        # without building the covaraince matric first, which is quicker and 
        # lighter on memory (when the problem gets BIG!)
        cov = pf.build_prior()
         # just a trick to sort indexes so that closely located pars in space are close in the covariance matrix 
        x = cov.df().sort_index(axis=0).sort_index(axis=1).values.copy()
        x[x==0.0] = np.NaN
        fig = plt.figure(figsize=(12,12))
        im = plt.imshow(x, interpolation='none')
        plt.gca().set_facecolor('k')
    
    # Draw and save parameter ensemble
    pe = pf.draw(nreals, use_specsim=True)
    pe.to_binary(os.path.join(pf.new_d, "prior_pe.jcb"))
    
    # some copying of execuatbles so we can run
    shutil.copy(pestpp_exe, pf.new_d)

    pst.pestpp_options["overdue_giveup_fac"] = 1000
    pst.pestpp_options['save_binary'] = True
    #pst.pestpp_options['ies_autoadaloc'] = True
    # prep for an initial run (just base run not ensemble or histroy matching)
    pst.control_data.noptmax = 0
    # write the control file
    pst.write(pst.filename, version=2)
    # run with noptmax = 0
    print("Running NOPTMAX=0................")
    pyemu.os_utils.run(f"pestpp-ies {pst.filename.name}", cwd=pf.new_d)
    # make sure it ran
    res_file = os.path.join(pf.new_d, f"{pst.filename.stem}.base.rei")
    assert os.path.exists(res_file), res_file
    pst.set_res(res_file)
    print(f"PHI = {pst.phi}")
    return pst


def attach_scenario(model=None, scen="SSP5", run=True):
    """
    Attach a SearRise scenario to forward model boundary condition
    :param model:
    :return:
    """
    if model is None:
        # build predictive model for original first
        model = build_from_orig(version='pred', run=False)
    if isinstance(model, str) or isinstance(model, Path):
        mpath = os.path.join("..", model)
        mname = [p for p in Path(mpath).glob("*.nam")][0]
        mname = mname.relative_to(mpath).name
        model = flopy.modflow.Modflow.load(mname, model_ws=mpath,
                                           load_only=['dis'])
    else:
        mpath = model.model_ws
        mname = model.namefile
    fname = os.path.join("..", "SLR_data", f"Dunedin_Harbourside_{scen}.csv")
    # first try to load csv (if it exists)
    if not os.path.exists(fname):
        # if csv doesnt exist, read xlsx
        # pandas can't handle too much as ns resolution
        data = pd.read_excel(fname.replace("csv", "xlsx"),
                             header=0, skiprows=1, index_col=0)
        data.to_csv(fname)
    else:
        data = pd.read_csv(fname, index_col=0)
    data = data.loc[data.index <= 2200]
    data.index = pd.to_datetime(data.index, format='%Y')
    data = data.resample('YS').interpolate(method='linear')
    # reduce to diff from current
    data = data.sub(data.iloc[0])
    data['deltadays'] = (data.index - pd.to_datetime(model.modeltime.start_datetime)).days
    # write scenario data to model directory
    data.to_csv(os.path.join(mpath, "scenario.csv"))
    if os.path.exists(os.path.join(mpath, "org")):
        # needs to be in org directory to be picked up by pset/pyemu methods
        # at run time
        data.to_csv(os.path.join(mpath, "org", "scenario.csv"))
    with open(os.path.join(mpath, 'scenario.txt'), 'w') as fp:
        fp.write(scen)
    # build ghb based on scenario
    utils.scenario2ghbdat(ghb_d=mpath)
    if run:
        # use pyemu to run in a process (not flopys built in)
        pyemu.os_utils.run(f"{mfnwt_exe.name} {mname}", cwd=mpath)
    return model