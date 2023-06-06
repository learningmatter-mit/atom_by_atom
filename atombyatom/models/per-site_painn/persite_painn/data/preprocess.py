import numpy as np
from tqdm import tqdm

def convert_site_prop(data, output_keys, fidelity_keys=None):
    data_converted = {}
    print("Preprocessing...")
    print(f"Target Keys: {output_keys}.")

    for key, val in tqdm(data.items()):

        target_key_bin = []
        target = []
        site_prop = val.site_properties

        for i in range(len(val)):
            o_val = []
            for key_o in output_keys:
                    if key_o in site_prop.keys():
                        o_val += [site_prop[key_o][i]]
                    else:
                        o_val += [np.nan]
            target.append(o_val)
            
        converted_site_prop = {"target": target}

        new_structure = val.copy(site_properties=converted_site_prop)
        data_converted[key] = new_structure

    return data_converted
