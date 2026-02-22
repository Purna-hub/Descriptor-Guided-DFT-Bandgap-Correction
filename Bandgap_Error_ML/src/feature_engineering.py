import pandas as pd
import numpy as np
from pymatgen.core import Composition, Element

def compute_features(df):

    df["delta_Eg"] = df["Eg_exp"] - df["Eg_DFT"]

    def extract(formula):
        comp = Composition(formula)
        atoms = comp.num_atoms

        chis, radii, ies, volumes = [], [], [], []
        total_Z, total_chi, total_valence, d_atoms = 0, 0, 0, 0

        for el, amt in comp.items():
            e = Element(el.symbol)

            if e.X is not None:
                chis.append(e.X)
                total_chi += e.X * amt

            if e.atomic_radius is not None:
                radii.append(e.atomic_radius)
                volumes.append((4/3)*np.pi*(e.atomic_radius**3))

            if e.ionization_energy is not None:
                ies.append(e.ionization_energy)

            total_Z += e.Z * amt

            if e.group is not None:
                total_valence += e.group * amt

            if e.is_transition_metal:
                d_atoms += amt

        delta_chi = max(chis)-min(chis) if len(chis)>1 else 0
        delta_r = max(radii)-min(radii) if len(radii)>1 else 0
        delta_IE = max(ies)-min(ies) if len(ies)>1 else 0
        delta_vol = max(volumes)-min(volumes) if len(volumes)>1 else 0

        return pd.Series([
            delta_r,
            delta_chi,
            1 - np.exp(-0.25*(delta_chi**2)),
            total_valence/atoms if atoms else 0,
            delta_IE,
            d_atoms/atoms if atoms else 0,
            total_Z/atoms if atoms else 0,
            total_chi/atoms if atoms else 0,
            delta_vol
        ])

    df[[
        "delta_r","delta_chi","pauling_ionicity","VEC",
        "delta_IE","d_fraction","Z_avg","chi_avg","delta_atomic_volume"
    ]] = df["Composition"].apply(extract)

    return df
