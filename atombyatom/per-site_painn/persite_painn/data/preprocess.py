import numpy as np
from tqdm import tqdm

AVAIL_KEYS = ["magmom", "bandfilling", "bandcenter", "charge", "atomiccharges","deltaE", "deltaO", "deltaOH", "deltaOOH", "phonon_bandcenter", "bader"]


def convert_site_prop(data, output_keys, fidelity_keys=None):
    data_converted = {}
    print("Preprocessing...")
    if fidelity_keys is not None:
        print(f"Target Keys: {output_keys} and Fidelity Keys: {fidelity_keys}.")
        check_keys = output_keys + fidelity_keys
        for key in check_keys:
            if key not in AVAIL_KEYS:
                raise NotImplementedError(f"{key} is not an available key")
    else:
        print(f"Target Keys: {output_keys}.")
        for key in output_keys:
            if key not in AVAIL_KEYS:
                raise NotImplementedError(f"{key} is not an available key")

    for key, val in tqdm(data.items()):
        target_key_bin = []
        fidelity_key_bin = []
        fidelity = []
        target = []
        site_prop = val.site_properties
        for i in range(len(val)):
            o_val = []
            for key_o in output_keys:
                target_key_bin.append(key_o)
                if key_o in list(site_prop.keys()) and key_o == "magmom":
                    o_val += [np.abs(site_prop[key_o][i])]
                elif key_o in list(site_prop.keys()) and key_o in ["deltaE", "deltaO", "deltaOH", "deltaOOH"]:
                    E_val = site_prop[key_o][i]
                    if E_val > -5 and E_val < 5:
                        o_val += [E_val]
                    else:
                        o_val += [np.nan]
                elif key_o in list(site_prop.keys()) and key_o not in ["magmom", "deltaE", "deltaO", "deltaOH", "deltaOOH"]:
                    o_val += [site_prop[key_o][i]]
                else:
                    o_val += [np.nan]
            target.append(o_val)
            if fidelity_keys is not None:
                f_val = []
                for key_f in fidelity_keys:
                    fidelity_key_bin.append(key_f)
                    if key_f in list(site_prop.keys()) and key_f == "magmom":
                        f_val += [np.abs(site_prop[key_f][i])]
                    elif key_f in list(site_prop.keys()) and key_f in ["deltaE", "deltaO", "deltaOH", "deltaOOH"]:
                        E_val = site_prop[key_f][i]
                        if E_val > -5 and E_val < 5:
                            f_val += [E_val]
                        else:
                            f_val += [np.nan]
                    elif key_f in list(site_prop.keys()) and key_f not in ["magmom", "deltaE", "deltaO", "deltaOH", "deltaOOH"]:
                        f_val += [site_prop[key_f][i]]
                    else:
                        f_val += [np.nan]

                fidelity.append(f_val)
        if fidelity_keys is not None:
            converted_site_prop = {"target": target, "fidelity": fidelity}
            # print(fidelity)
        else:
            converted_site_prop = {"target": target}

        new_structure = val.copy(site_properties=converted_site_prop)
        data_converted[key] = new_structure

    return data_converted
