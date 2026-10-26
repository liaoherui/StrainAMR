"""Reading the training protein-cluster file into PC token ids.

There is exactly one copy of this logic, used by both build-train and build-test, because the two
sides MUST agree: the token ids are positional (the n-th non-singleton cluster is token n), so if one
side keeps a cluster the other drops, every id after it is shifted and the two files describe
different clusters under the same names.  That is not a hypothetical -- build-test used to carry its
own copy of the parser without the duplicate-protein step below, which silently renumbered every
validation PC token relative to the training vocabulary the model had learned.

Two rules define the numbering, both inherited from v1:

  * proteins removed as exact duplicates before clustering are put back into their representative's
    cluster FIRST.  A gene carried verbatim by 200 genomes is one sequence in the deduplicated input
    and would otherwise look like a cluster of one;
  * clusters with a single member are then dropped and get no token, since a protein seen in one
    genome only cannot carry cross-genome signal.  Numbering counts only the clusters that survive.
"""

from __future__ import annotations

import os
import re

CLUSTER_FILE = 'merge_train_cdhit.clstr'
DUP_MAP = 'duplicate_proteins_train.tsv'

_MEMBER = re.compile(r'>(\S+?)(?:\.\.\.|\s|$)')


def dup_map_for(cluster_file: str) -> str:
    """The duplicate map that belongs to a cluster file: written beside it by build-train."""
    return os.path.join(os.path.dirname(os.path.abspath(cluster_file)), DUP_MAP)


def read_dups(dup_map: str) -> dict:
    """representative protein id -> the identical proteins that were dropped before clustering."""
    dups = {}
    if dup_map and os.path.exists(dup_map):
        with open(dup_map) as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) == 2:
                    dups.setdefault(parts[1], []).append(parts[0])
    return dups


def read_clusters(cluster_file: str, dup_map: str = None) -> dict:
    """cluster name -> {protein id: ''}, in the order the file lists them, duplicates restored."""
    dups = read_dups(dup_map if dup_map is not None else dup_map_for(cluster_file))
    clusters = {}
    cls = None
    with open(cluster_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if re.search('Cluster', line):
                cls = re.sub(' ', '_', re.sub('>', '', line))
                # dict order is first-seen order, which is what the positional numbering relies on
                clusters.setdefault(cls, {})
                continue
            m = _MEMBER.search(line)
            if not m:
                continue
            pid = m.group(1)
            clusters[cls][pid] = ''
            for dp in dups.get(pid, ()):
                clusters[cls][dp] = ''
    return clusters


def cluster_tokens(cluster_file: str, dup_map: str = None, matches_file: str = None) -> dict:
    """protein id -> PC token id, for every protein in a cluster of two or more."""
    clusters = read_clusters(cluster_file, dup_map)
    tokens = {}
    out = open(matches_file, 'w') if matches_file else None
    try:
        i = 1
        for c in clusters:
            members = clusters[c]
            if len(members) == 1:
                continue
            for p in members:
                tokens[p] = str(i)
                if out:
                    out.write(str(i) + '\t' + p + '\n')
            i += 1
    finally:
        if out:
            out.close()
    return tokens


def summarise(cluster_file: str, dup_map: str = None) -> str:
    clusters = read_clusters(cluster_file, dup_map)
    sizes = [len(v) for v in clusters.values()]
    kept = sum(1 for s in sizes if s > 1)
    return (f"{len(clusters)} clusters, {kept} with two or more members (tokens 1..{kept}), "
            f"{len(clusters) - kept} singletons dropped")
