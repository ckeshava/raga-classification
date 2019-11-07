import glob

def get_all_ragas():
    # store all ragas in a dictionary
    json_files = glob.glob(os.path.join(config.JSON_FILES, "*.json"))
    assert len(json_files) > 0, "No json Files"

    counter = 0
    all_ragas = {}

    for metadata_file in json_files:
        with open(metadata_file, 'r') as f:
            meta_info = json.load(f)

        if len(meta_info['raaga']) > 0:
            raga = meta_info['raaga'][0]['name']

            if raga not in all_ragas:
                all_ragas[raga] = counter
                counter += 1

    with open('raga_index.json', 'w') as fp:
        json.dump(all_ragas, fp)
        
get_all_ragas()

