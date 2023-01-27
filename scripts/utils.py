import pandas as pd
import os
import numpy as np
import flopy
import matplotlib.pyplot as plt
from pathlib import Path
import pyemu
import shutil
import platform

if "darwin" in platform.system().lower():
    mf2k_exe = Path("..", "bin", "mf2k")
    mfnwt_exe = Path("..", "bin", "mfnwt")
    pestpp_exe = Path("..", "bin", "pestpp-ies")
else:
    mf2k_exe = Path("..", "bin", "mf2k.exe")
    mfnwt_exe = Path("..", "bin", "mfnwt.exe")
    pestpp_exe = Path("..", "bin", "pestpp-ies.exe")


def hds_bin2csv(d='.', m=None, modelfname="SouthDun_SS.nam"):
    # need to get binary modflow output in to something more friendly
    if m is None:
        m = flopy.modflow.Modflow.load(f=modelfname, model_ws=d, load_only=['DIS', 'BAS6', 'OC'])
    ib = m.bas6.ibound.array # get active model definition 
    nper = m.dis.nper
    # read it in (thanks flopy)
    with flopy.utils.HeadFile(
                os.path.join(d, [f for f in m.output_fnames if f.endswith('hds')][0])
                ) as hds:
        hd = hds.get_alldata() # returns 4D array (nper, nlay, nrow, ncol)

    # # dump out an array rep. of output for every model kper and layer
    # for t, h in enumerate(hds.get_alldata()): # loop over stress periods
    #     for k, h0 in enumerate(h): # loop over layers
    #         plt.figure()
    #         plt.imshow(np.ma.masked_where(h0==new_m.bas6.hnoflo, h0))
    #         np.savetxt(os.path.join(d, f"hds_kper{t}_lay{k}.csv"), h0, delimiter=',')

    # or output all the data as a nice (long) dataframe
    idx = np.where(ib) # get index of only active cells (where ibound is none zero)
    df0 = pd.DataFrame(list(idx), index=['k','i','j']).T # put kij indexes on a dataframe
    obs_locs0 = pd.read_csv(os.path.join(d, "Cox_Operational_Targets.dat"),
                           index_col=0)
    dfs = []  # empty list to accumulat kper
    welobs = []
    for t in range(nper): # loop over stress periods in model (may want to carefully choose stress periods for outputs (or you might get too many to handle))
        # mask out ghb
        ghb = pd.read_csv(os.path.join(d,f"GHB_{t:04d}.dat"), header=None, delim_whitespace=True,
                          usecols=[0,1,2])-1
        # ijk zerobased!!!!
        ghb.columns = ['k','i','j']
        df = df0.loc[~df0.set_index(['k','i','j']).index.isin(ghb.set_index(['k','i','j']).index)].copy()
        obs_locs = obs_locs0.copy()
        df.loc[:, 'kper'] = t  # add a "time" axis
        obs_locs.loc[:, 'kper'] = t 
        welobs.append(obs_locs)
        dfs.append(df)
    df = pd.concat(dfs)
    welobs = pd.concat(welobs)
    df['obstyp'] = "hd"
    # access elements in 4D numpy array of hds
    df['obsval'] = hd[tuple(df[['kper','k','i','j']].T.values)]
    welobs['obsval'] = hd[tuple(welobs[['kper','k','i','j']].T.values)]
    # dump to csv
    fname = "hds.csv"
    df.to_csv(os.path.join(d, fname), index=False)
    wlfname = "welobs.csv"
    welobs.to_csv(os.path.join(d, wlfname), index=True)
    return fname, wlfname


def cbc_bin2csv(text="DRAINS", d='.', m=None, modelfname="SouthDun_SS.nam"):
    # need to get binary modflow output in to something more friendly
    if m is None:
        m = flopy.modflow.Modflow.load(f=modelfname, 
                                       model_ws=d, 
                                       load_only=['DIS', 'BAS6', 'OC'])
    # Load budget file    
    with flopy.utils.CellBudgetFile(
        os.path.join(d, modelfname.replace('nam', 'cbc'))
    ) as cbc:
        # get dataset associated with "TEXT"
        ds = cbc.get_data(text=text)
    dfs=[]
    sumq = {}

    for t, df in enumerate(ds): # loop over stress periods in model (may want to carefully choose stress periods for outputs (or you might get too many to handle))
        # convert to nice dataframe instead of rank recarray blah
        df = pd.DataFrame(df)
        sumq[t] = df.q.sum()
        # get model k i j from rank node
        df.loc[:, ['k','i','j']] = pd.DataFrame(
            m.dis.get_lrc(df.node.values.tolist()), 
            columns=['k','i','j']
            )
        # add kper number
        df.loc[:, 'kper'] = t   
        dfs.append(df)
    # concat list of dataframes
    df = pd.concat(dfs)
    sumq_df = pd.DataFrame.from_dict(sumq, orient='index', columns=['sum']).rename_axis('kper')
    # prepping filename
    txt = text.lower().replace(' ','_')
    fname = f"{txt}.csv"
    df.to_csv(os.path.join(d, fname), index=False)
    sumfnme = f"{txt}_sum.csv"
    sumq_df.to_csv(os.path.join(d, sumfnme))
    return fname, sumfnme   
    
def wl_xlsx2csv(data_d="data"):
    """
    Converting xlsx files into csv because excel sucks
    """
    d_path = os.path.join("..", data_d, "water_levels") 
    for f in [f for f in os.listdir(d_path) if "xlsx" in f and "~$" not in f]:
        df = pd.read_excel(os.path.join(d_path, f), header = None, 
                           index_col = 0, skiprows = 2)
        df.index = pd.to_datetime(df.index, format='%d-%m-%Y %H:%M')
        df.index.name = f.split("_")[0]
        df.columns = ["level"]
        df.to_csv(os.path.join(d_path, f.replace('xlsx', 'csv')), index=True)

def prep_and_run(pst, t_d="template", m_d="master",
                 noptmax=-1, nreals=1000, r_d="..",
                 prior_pe="prior_pe.jcb", post_pe=None, 
                 scen=None, nworker=15):
    import pyemu
    # PREP and RUN SWEEP
    t_d = os.path.join(r_d, t_d)
    m_d = os.path.join(r_d, m_d)
    try:
        pst.control_data.noptmax = noptmax
    except AttributeError:
        pst = pyemu.Pst(os.path.join(t_d, pst))
        pst.control_data.noptmax = noptmax
    pst.filename = Path(pst.filename)
    if scen is not None:
        scenario2ghbdat(label=scen, ghb_d=t_d)
    # define what file has the parameter ensemble
    if post_pe is not None:
        # only want a forward run sweep if running a posterior
        if noptmax != -1:
            print("resetting NOPTMAX to -1")
            pst.control_data.noptmax = -1
        # read "prior" pe file (from current model)
        prpe = pyemu.ParameterEnsemble.from_binary(pst, os.path.join(t_d, prior_pe))
        prpe = prpe.iloc[:nreals, :]
        # add the parval1 vals to ensemble:
        # prpe.add_base()
        # read "posterior" (from history matche model)
        if post_pe.split('.')[-1] == 'csv':
            pope = pd.read_csv(post_pe, low_memory=False, index_col=0)
            pope = pyemu.ParameterEnsemble(pst, pope)
            prpe.index = prpe.index.astype(int)
        else:
            pope = pyemu.ParameterEnsemble.from_binary(pst, post_pe)
        # stomp on "prior" ensemble parameter values with conditioned realisations (from posterior)
        prpe.loc[:, pope.columns] = pope.loc[prpe.index, :].values
        pefname = os.path.basename(post_pe).replace('csv', 'jcb')
        # prpe = pyemu.ParameterEnsemble(pst, prpe)
        prpe.to_binary(os.path.join(t_d, pefname))
    else:
        pefname = "prior_pe.jcb"
    pst.pestpp_options["ies_par_en"] = pefname
    pst.pestpp_options["ies_num_reals"] = nreals
    # modified pst object so need to re write the control file
    pst.write(pst.filename, version=2)

    # start a locally parrellised sweep

    pyemu.os_utils.start_workers(
        t_d,
        "pestpp-ies",
        pst.filename.name,
        num_workers=nworker,  # restrict number of parallel workers for binder demo to avoid throttling
        master_dir=m_d, 
        worker_root=r_d,
        cleanup=False
    )


def scenario2ghbdat(scen=None, ghb_d="."):
    """
    Will overwrite ghb files in model run directory
    (or "org" directory in PEST setup) with annual SLR projection data
    for requested scenario
    :param scen:
    :param ghb_d:
    :return:
    TODO: -- add something more clever to algin model kper dates
        and projection dates. Currently just assuming second kper is 2010
        and all kper are annual.
    """
    #scendict={
     #         "dp21v_rcp85": "Dunedin_Harbourside_dp21v_rcp85.xlsx",
      #        "dp21v_rcp45": "Dunedin_Harbourside_dp21v_rcp45.xlsx"
       #       "dp21v_rcp26": "Dunedin_Harbourside_dp21v_rcp26.xlsx"
    #}
    def _read_ghb(ghbfname):
        df = pd.read_csv(ghbfname, header=None, delim_whitespace=True)
        df.columns = ['k', 'i', 'j', 'bhead', 'cond']
        return df

    if scen is None:
        # scenario.csv will be parameterised and updated at run time.
        scen = pd.read_csv(os.path.join(ghb_d, "scenario.csv"), index_col=0)

    save_d = ghb_d
    sep = ' '

    try:
        ghb0 = _read_ghb(os.path.join(ghb_d, "GHB_0000.dat.bkup"))
    except FileNotFoundError:
        # read in model input data from first stressperiod
        ghb0 = _read_ghb(os.path.join(ghb_d, "GHB_0000.dat"))
        ghb0.to_csv(os.path.join(ghb_d, "GHB_0000.dat.bkup"),
                    header=False, index=False, sep=sep)
    ref = ghb0.bhead
    for i, y in enumerate(scen['0.5']):
        # read in corresponding ghb (this may have had conductances altered
        #  according to realisation)
        ghbname = os.path.join(ghb_d, f"GHB_{i+1:04d}.dat")
        if os.path.exists(ghbname):
            ghb = _read_ghb(ghbname)
            ghb.loc[:, 'bhead'] = ref + (y / 100)  # TODO check units of any new data!
            ghb.to_csv(ghbname, header=False, index=False, sep=sep)


def _load_model(m_d):
    mname = [p for p in Path(m_d).glob("*.nam")][0]
    mname = mname.relative_to(m_d).name
    m = flopy.modflow.Modflow.load(
        f=mname,
        model_ws=m_d,
        version='mfnwt',
        exe_name=mfnwt_exe.name,
        verbose=False,
        check=True
    )
    return m


def _try_load_pst(m_d, pst=None):
    assert not all([a is None for a in [m_d, pst]]), ("Need to pass one of "
                                                      "directory or pst object")
    if pst is None:
        pstf = [p for p in Path(m_d).glob("*.pst")][0].name.replace('_rw', '')
        pst = pyemu.Pst(os.path.join(m_d, pstf))
    elif isinstance(pst, str):
        pst = pyemu.Pst(os.path.join(m_d, pst))
    return pst


def try_load_ensemble(pst, fname, kind='par', **kwargs):
    print(f"Trying to load {fname}")
    if fname.split('.')[-1] == 'csv':
        try:
            ensemble = pd.read_csv(fname, low_memory=False, **kwargs)
        except FileNotFoundError:
            fname = fname.replace(".csv", ".jcb")
            print(f"...failed. Trying {fname}...")
            if kind == 'par':
                ensemble = pyemu.ParameterEnsemble.from_binary(pst, fname.replace(".csv", ".jcb"))._df
            elif kind == "obs":
                ensemble = pyemu.ObservationEnsemble.from_binary(pst, fname.replace(".csv", ".jcb"))._df
    elif fname.split('.')[-1] == "jcb":
        try:
            if kind == 'par':
                ensemble = pyemu.ParameterEnsemble.from_binary(pst, fname.replace(".csv", ".jcb"))._df
            elif kind == "obs":
                ensemble = pyemu.ObservationEnsemble.from_binary(pst, fname.replace(".csv", ".jcb"))._df
        except FileNotFoundError:
            fname = fname.replace(".jcb", ".csv")
            print(f"...failed. Trying {fname}...")
            ensemble = pd.read_csv(fname.replace(".jcb", ".csv"), low_memory=False, **kwargs)
    return ensemble