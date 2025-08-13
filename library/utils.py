import os


def load_token_mappings(files):
    """Load multiple mapping files and return a dict: token_id -> feature name."""
    mapping = {}
    if not files:
        return mapping
    for path in files:
        if not path or not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                fname = os.path.basename(path)
                if 'pc_matches' in fname:
                    token, feature = parts[0], parts[1]
                elif 'kmer_token_id' in fname:
                    feature, token = parts[0], parts[1]
                else:
                    feature = parts[0]
                    token = parts[-1]
                mapping[str(token)] = feature
    return mapping


def token_to_feature(token_id, mapping):
    """Return human readable feature for token_id if available."""
    return mapping.get(str(token_id), str(token_id))
