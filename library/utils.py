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


def load_rgi_annotations(rgi_dir):

    """Parse RGI tabular outputs and map ARO IDs to AMR Gene Family."""

    info = {}
    if not rgi_dir or not os.path.isdir(rgi_dir):
        return info
    for fname in os.listdir(rgi_dir):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(rgi_dir, fname)
        with open(path, 'r') as f:
            header = f.readline().strip().split('\t')
            header = [h.replace(' ', '_') for h in header]
            idx = {h: i for i, h in enumerate(header)}

            aro_i = idx.get('ARO')
            if aro_i is None:
                continue
            gf_i = idx.get('AMR_Gene_Family')

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) <= aro_i:
                    continue
                aro = parts[aro_i]
                gf = parts[gf_i] if gf_i is not None and gf_i < len(parts) else 'NA'

                info[aro] = gf

    return info
