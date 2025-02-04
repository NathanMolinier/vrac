import glob
import json

def main():
    txt_path = "/home/GRAMES.POLYMTL.CA/p118739/data/datasets/article-totalspineseg/config/splits.txt"

    with open(txt_path, "r") as f:
        lines= f.readlines()
    
    c=0
    d = {"MULTI":{}, "SINGLE":{}, "SPIDER":{}, "WHOLE":{}}
    sub_list = []
    train=True

    # Generate config JSON
    config = {
        "TYPE": "IMAGE",
        "CONTRASTS": "",
        "DATASETS_PATH": "/home/GRAMES.POLYMTL.CA/p118739/data/datasets",
        "TRAINING": [],
        "VALIDATION": [],
        "TESTING": []
    }

    data_dict = {
        "MULTI":"data-multi-subject", 
        "SINGLE":"data-single-subject", 
        "SPIDER":"spider-challenge-2023", 
        "WHOLE":"whole-spine"
    }

    for line in lines:
        if train:
            sub_name = line.split('_0000')[0][:-3]
        else:
            sub_name = line.split('_0000')[0]
        if sub_name.startswith("sub-"):
            if sub_name not in sub_list:
                for k in d.keys():
                    if k in sub_name:
                        contrast = sub_name.split('_')[-1]
                        if contrast not in d[k].keys():
                            d[k][contrast]=1
                        else:
                            d[k][contrast]+=1

                        # Fetch image path
                        
                        filename = sub_name.replace(k, '') + '.nii.gz'
                        fullpath = glob.glob(f"/home/GRAMES.POLYMTL.CA/p118739/data/datasets/{data_dict[k]}/**/*" + filename, recursive=True)
                        if len(fullpath) == 1:
                            if train:
                                config["TRAINING"].append({'IMAGE':fullpath[0].replace("/home/GRAMES.POLYMTL.CA/p118739/data/datasets/", "")})
                            else:
                                config["TESTING"].append({'IMAGE':fullpath[0].replace("/home/GRAMES.POLYMTL.CA/p118739/data/datasets/", "")})
                        else:
                            print(f"Too many or not enough paths detected for {filename}")
                c+=1
                sub_list.append(sub_name)

        elif "image" in sub_name:
            print(c)
            print(d)
            print(line)
            c=0
            d={"MULTI":{}, "SINGLE":{}, "SPIDER":{}, "WHOLE":{}}
            if "imagesTs" in sub_name:
                train=False
        elif "label" in sub_name:       
            print(c)
            print(d)
            break
    
    config_path = "/home/GRAMES.POLYMTL.CA/p118739/data/datasets/article-totalspineseg/config/splits.json"
    json.dump(config, open(config_path, 'w'), indent=4)

    

if __name__=='__main__':
    main()