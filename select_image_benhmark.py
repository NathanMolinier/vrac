import json


def main():
    json_path = "/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/code/disc-labeling-benchmark/benchmark.json"
    out_path = "/home/GRAMES.POLYMTL.CA/p118739/data_nvme_p118739/code/disc-labeling-benchmark/benchmark_vanderbilt.json"
    
    # Load json
    with open(json_path, 'r') as file:
        config = json.load(file)
    
    new_path_list = []
    for dic in config['TESTING']:
        if 'lumbar-vanderbilt' in dic["IMAGE"]:
            new_path_list.append(dic)
    
    # Update json
    config['TESTING'] = new_path_list

    # Save json
    json.dump(config, open(out_path, 'w'), indent=4)
    



if __name__ == '__main__':
    main()