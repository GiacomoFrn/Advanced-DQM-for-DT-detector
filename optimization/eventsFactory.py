import pandas as pd
import numpy as np
from constants import DURATION_BX, VDRIFT
from mappings import Mapping


def map_hit_position(hits_df_, local=False):
    """
    Compute the position of the hit.
    If local=True the positions left/right will 
    be computed in the local frame of reference
    (e.g. X_LEFT_LOC). Else, the positions will be
    computed in the global frame of reference
    """

    hits_df_["HIT_DRIFT_TIME"] = (hits_df_["BX_COUNTER"]+hits_df_["TDC_MEAS"]/30)*DURATION_BX-hits_df_["T0_NS"]

    ref = "LOC" if local else "GLOB"
    hits_df_[f"X_LEFT_{ref}"] = hits_df_[f"WIRE_X_{ref}"] - hits_df_["HIT_DRIFT_TIME"]*VDRIFT
    hits_df_[f"X_RIGHT_{ref}"] = hits_df_[f"WIRE_X_{ref}"] + hits_df_["HIT_DRIFT_TIME"]*VDRIFT

    return hits_df_


def usingTrigger(stream_df, hits_df_, cfg):
    # select all orbits with a trigger signal from
    # the FPGA , saving also trigger angle and position
    trigger_df_ = stream_df[
        (stream_df["HEAD"] == cfg["headers"]["t0_trg"]) & (stream_df["FPGA"] == 0)
    ].copy()

    triggerAngle_df = stream_df[
        (stream_df["HEAD"] == 4) & (stream_df["FPGA"] == 0)
    ].copy()

    triggerPosition_df = stream_df[
        (stream_df["HEAD"] == 5) & (stream_df["FPGA"] == 0)
    ].copy()

    trigger_df_["T_T0"] = (
        trigger_df_["BX_COUNTER"] * 25 + trigger_df_["TDC_MEAS"] * 25 / 30
    )

    triggerAngle_df.rename(columns={"TDC_MEAS": "T_ANGLE"}, inplace=True)

    trigger_df_ = pd.merge(
        trigger_df_,
        triggerAngle_df[["ORBIT_CNT", "T_ANGLE"]],
        left_on="ORBIT_CNT",
        right_on="ORBIT_CNT",
        suffixes=(None, None),
    )

    triggerPosition_df.rename(columns={"TDC_MEAS": "T_POSITION"}, inplace=True)
    trigger_df_ = pd.merge(
        trigger_df_,
        triggerPosition_df[["ORBIT_CNT", "T_POSITION"]],
        left_on="ORBIT_CNT",
        right_on="ORBIT_CNT",
        suffixes=(None, None),
    )

    hits_df_ = pd.merge(
        hits_df_,
        trigger_df_[["ORBIT_CNT", "T_T0", "T_ANGLE", "T_POSITION"]],
        left_on="ORBIT_CNT",
        right_on="ORBIT_CNT",
        suffixes=(None, None),
    )

    return hits_df_


def computeEvents(hits_df_):
    events = [group for _, group in hits_df_.groupby("ORBIT_CNT") if len(group) <= 32]
    #events = [hits_df_[hits_df_["ORBIT_CNT"] == x] for x in pd.unique(hits_df_.ORBIT_CNT)]
    return events


def getEvents(df_fname, cfg, runTimeShift, useTrigger):
    
    #reading df from file
    dtype_dict = { 'HEAD':np.uint8, 'FPGA':np.uint8, 'TDC_CHANNEL':np.uint8, 'ORBIT_CNT':np.uint64, 'BX_COUNTER':np.uint16, 'TDC_MEAS':np.uint8 }
    print("Reading dataset from file...")
    stream_df = pd.read_csv(df_fname, dtype=dtype_dict)
    #drop NaN
    stream_df.dropna(inplace=True)

    # create a dataframe with only valid hits ->
    # trigger words and scintillator hits are removed
    hits_df = stream_df[
        (stream_df.HEAD == cfg["headers"]["valid_hit"]) &
        (stream_df.TDC_CHANNEL <= 127)
        ]
    
    
    # select all orbits with a trigger signal from
    # the scintillators coincidence
    trigger_df = stream_df[
        (stream_df["HEAD"] == cfg["scintillator"]["head"]) & 
        (stream_df["FPGA"] == cfg["scintillator"]["fpga"]) & 
        (stream_df["TDC_CHANNEL"] == cfg["scintillator"]["tdc_ch"])
    ]

    # create a T0 column (in ns)
    trigger_df["T0_NS"] = (trigger_df["BX_COUNTER"] + trigger_df["TDC_MEAS"] / 30) * DURATION_BX

    # select only hits in the same orbit of a scint trigger signal
    hits_df_ = pd.merge(
        hits_df, trigger_df[["ORBIT_CNT","T0_NS"]],
        left_on="ORBIT_CNT", right_on="ORBIT_CNT",
        suffixes=(None, None)
    )

    del trigger_df
    del hits_df
    # print the number of valid hits found in data
    print(f"Valid hits: {hits_df_.shape[0]}")

    # if true consider only the hits in the same orbit of a scint trigger AND FPGA trigger signals
    if (useTrigger == True):
        hits_df_ = usingTrigger(stream_df=stream_df, hits_df_=hits_df_, cfg=cfg)

    del stream_df
    # create mapping with the loaded configurations
    mapper = Mapping(cfg)
    # map hits
    hits_df_ = mapper.global_map(hits_df_)

    # TIME SHIFTING
    # apply scintillator calibration -> they shoul be computed
    # for each run! For this example run, values are provided in
    # the configurations file.

    for sl, offset_sl in cfg["time_offset_sl"].items():
        # correction is in the form:
        # coarse offset valid for all SL + fine tuning 
        hits_df_.loc[
            hits_df_["SL"] == sl, "T0_NS"
        ] -= cfg["time_offset_scint"] -runTimeShift + offset_sl # 6 for 123, 8 for 0

    # having the time allows us to compute hit position 
    # with left / right ambiguity

    hits_df_ = map_hit_position(hits_df_, local=False)

    return computeEvents(hits_df_)