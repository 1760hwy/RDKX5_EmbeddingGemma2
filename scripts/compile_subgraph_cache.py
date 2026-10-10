"""Reuse only byte-identical HBIR with identical compiler/options after timeout."""
import ast
import hashlib
import json
import logging
import shutil
from pathlib import Path


class CachedPrefixVerified(Exception):
    """Dry run stops before the first genuinely uncompiled subgraph."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(source, cache, log, quantized_sha):
    source, cache = Path(source), Path(cache)
    cache.mkdir(exist_ok=True)
    options = {}
    name = None
    for line in Path(log).read_text().splitlines():
        if 'Compile submodel: ' in line:
            name = line.split('Compile submodel: ', 1)[1].strip()
        elif 'hbdk-cc parameters:' in line and name:
            options[name] = ast.literal_eval(line.split('hbdk-cc parameters:', 1)[1])
    compiler = Path('/usr/local/lib/python3.10/dist-packages/hbdk/bin/hbdk-cc')
    entries = {}
    for name, params in options.items():
        hbir, hbm = source/(name+'.hbir'), source/(name+'.hbm')
        if not hbir.exists() or not hbm.exists():
            continue
        for path in (hbir, hbm):
            shutil.copy2(path, cache/path.name)
            assert digest(path) == digest(cache/path.name)
        entries[name] = dict(hbir_sha256=digest(hbir), hbm_sha256=digest(hbm),
                             params=params, march='bayes-e')
    manifest = dict(quantized_sha256=quantized_sha,
                    compiler_sha256=digest(compiler), entries=entries)
    (cache/'resume_manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def install(module, cache, quantized_sha, dry_run=False):
    cache = Path(cache)
    manifest = json.loads((cache/'resume_manifest.json').read_text())
    assert manifest['quantized_sha256'] == quantized_sha
    assert manifest['compiler_sha256'] == digest('/usr/local/lib/python3.10/dist-packages/hbdk/bin/hbdk-cc')
    original = module.hbdk_cc
    stats = {'reused': [], 'compiled': []}

    def compile_cached(hbir_model, hbm_model, march, params=None, dump_all_models=False):
        name = Path(hbir_model).stem
        entry = manifest['entries'].get(name)
        if entry:
            assert entry['march'] == march and entry['params'] == (params or []), repr({'name':name,'expected':entry['params'],'actual':params,'march':march})
            assert digest(hbir_model) == entry['hbir_sha256'], 'Regenerated HBIR differs: '+name
            assert digest(cache/(name+'.hbir')) == entry['hbir_sha256']
            assert digest(cache/(name+'.hbm')) == entry['hbm_sha256']
            shutil.copy2(cache/(name+'.hbm'), hbm_model)
            stats['reused'].append(name)
            logging.info('Verified cached subgraph: %s', name)
        else:
            if dry_run:
                assert set(stats['reused']) == set(manifest['entries']), 'Not all cached subgraphs verified'
                raise CachedPrefixVerified(name)
            original(hbir_model, hbm_model, march, params, dump_all_models)
            stats['compiled'].append(name)
        (cache/'reuse_status.json').write_text(json.dumps(stats, indent=2))
    module.hbdk_cc = compile_cached
    return stats
