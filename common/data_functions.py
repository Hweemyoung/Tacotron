import os

def load_filepaths_and_texts(root, dataset_path, filename, split='|'):
    with open(filename, encoding='utf-8') as f:
        file_paths_and_text = []
        for line in f:
            parts = line.strip().split(split)  # parts[0]: filename of an audio # parts[1]: text
            if len(parts) > 2:
                raise Exception(
                    'incorrect line format for file: {}'.format(filename)
                )
            path = os.path.join(root, parts[0])
            text = parts[1]
            file_paths_and_text.append(tuple(path, text))
    return file_paths_and_text