"""Immutable research archive: wait for audited science, stream checked tar to R2.

Archive state lives OUTSIDE all source trees. Uses the already tested OpenAct
multipart transport without altering it. No credentials, environments, model
download caches, deletion, public ACL, or scientific configuration changes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_transport(cfg):
    p = cfg['upload_helper']
    if sha(p) != cfg['upload_helper_sha256']:
        raise ValueError('Frozen multipart helper hash differs')
    spec = importlib.util.spec_from_file_location('openact_archive_transport', p)
    u = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(u)
    return u


def gate(cfg):
    r = Path(cfg['reliability_root'])
    if sha(r / 'protocol.json') != cfg['reliability_protocol_sha256']:
        raise ValueError('Reliability protocol changed')
    status = read(r / 'job_status.json')
    if status['status'] in ('failed', 'cancelled_by_user'):
        raise RuntimeError('Reliability failed or cancelled; preserve source and stop archive')
    if status['status'] != 'complete':
        pid = status.get('pid')
        if not pid or not Path(f'/proc/{pid}').exists():
            raise RuntimeError('Reliability reports running but process is absent')
        return False
    done, audit = read(r / 'COMPLETE.json'), read(r / 'audit.json')
    if not audit['passed'] or done['audit_sha256'] != sha(r / 'audit.json'):
        raise ValueError('Reliability final audit does not match completion')
    if done['protocol_sha256'] != cfg['reliability_protocol_sha256']:
        raise ValueError('Wrong completed protocol')
    if audit['selections'] != 5278 or audit['fixed_refits'] != 377 or audit['data_rows'] != 5000:
        raise ValueError('Reliability coverage is incomplete')
    pid = status.get('pid')
    return not pid or not Path(f'/proc/{pid}').exists()


def stat_entry(path, name):
    s = path.lstat()
    if stat.S_ISLNK(s.st_mode):
        return dict(path=str(path), name=name, kind='symlink', target=os.readlink(path),
                    bytes=0, mtime_ns=s.st_mtime_ns)
    if not stat.S_ISREG(s.st_mode):
        raise ValueError(f'Unsupported source entry: {path}')
    if path.name in ('.env', 'credentials', 'r2-upload.conf') or path.suffix in ('.pem', '.key'):
        raise ValueError(f'Private-file name in experiment tree: {path}')
    return dict(path=str(path), name=name, kind='file', bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def collect_entries(source, alias, exclude_top=()):
    source = Path(source)
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f'Expected real source directory: {source}')
    entries = []
    for base, dirs, files in os.walk(source, followlinks=False):
        dirs.sort(); files.sort()
        if Path(base) == source:
            dirs[:] = [d for d in dirs if d not in exclude_top]
            files = [f for f in files if f not in exclude_top]
        for d in list(dirs):
            p = Path(base) / d
            if p.is_symlink():
                entries.append(stat_entry(p, alias + '/' + p.relative_to(source).as_posix()))
                dirs.remove(d)
        for f in files:
            p = Path(base) / f
            entries.append(stat_entry(p, alias + '/' + p.relative_to(source).as_posix()))
    by_path = {e['path']: e for e in entries}
    # New collection files retain their publisher checksums; do not merely
    # checksum a possibly changed tensor and declare that original data valid.
    for receipt_entry in entries:
        rp = Path(receipt_entry['path'])
        if rp.name != '_COPY_VERIFIED.json' or receipt_entry['kind'] != 'file':
            continue
        receipt = read(rp)
        if not isinstance(receipt.get('files'), dict):
            raise ValueError(f'Unrecognized publisher checksum receipt: {rp}')
        for relative, expected in receipt['files'].items():
            target = rp.parent / relative
            if '..' in Path(relative).parts or Path(relative).is_absolute():
                raise ValueError('Unsafe relative path in publisher receipt')
            item = by_path.get(str(target))
            if not item or item['kind'] != 'file' or item['bytes'] != expected['bytes']:
                raise ValueError(f'Publisher receipt file missing or size differs: {target}')
            item['expected_sha256'] = expected['sha256']
    return sorted(entries, key=lambda e: e['name'])


def chunks(entries, target):
    batch = []; size = 0
    for entry in entries:
        if batch and size + entry['bytes'] > target:
            yield batch; batch = []; size = 0
        batch.append(entry); size += entry['bytes']
    if batch:
        yield batch


def verify_stat(e):
    p = Path(e['path']); s = p.lstat()
    if s.st_mtime_ns != e['mtime_ns']:
        raise ValueError(f'Source changed after inventory: {p}')
    if e['kind'] == 'file':
        if not stat.S_ISREG(s.st_mode) or s.st_size != e['bytes']:
            raise ValueError(f'Source type/size changed: {p}')
    elif not stat.S_ISLNK(s.st_mode) or os.readlink(p) != e['target']:
        raise ValueError(f'Symlink changed: {p}')


def stream(entries, sink, u):
    index = []
    with tarfile.open(fileobj=sink, mode='w|', format=tarfile.PAX_FORMAT,
                      bufsize=1024**2, copybufsize=1024**2) as archive:
        for e in entries:
            verify_stat(e)
            info = tarfile.TarInfo(e['name']); info.mode = 0o600; info.mtime = 0
            info.uid = info.gid = 0; info.uname = info.gname = ''
            if e['kind'] == 'symlink':
                info.type = tarfile.SYMTYPE; info.linkname = e['target']; archive.addfile(info)
                index.append(dict(name=e['name'], kind='symlink', target=e['target'], bytes=0))
            else:
                info.size = e['bytes']
                offset = archive.offset + len(info.tobuf(format=tarfile.PAX_FORMAT))
                with open(e['path'], 'rb') as raw:
                    reader = u.HashReader(raw); archive.addfile(info, reader)
                if reader.count != e['bytes']:
                    raise ValueError('Source truncated while streaming')
                if e.get('expected_sha256') and reader.h.hexdigest() != e['expected_sha256']:
                    raise ValueError(f'Original publisher SHA256 differs: {e["path"]}')
                index.append(dict(name=e['name'], kind='file', bytes=e['bytes'],
                                  offset=offset, sha256=reader.h.hexdigest()))
            verify_stat(e)
    return index


def code_snapshots(cfg, state):
    folder = state / 'code'; folder.mkdir(exist_ok=True)
    result = []
    for repo in cfg['repositories']:
        p = repo['path']; name = repo['name']
        def git(*args):
            return subprocess.check_output(['git', '-C', p, *args], text=True).strip()
        commit = git('rev-parse', 'HEAD')
        marker=folder/(name+'.json')
        if marker.exists() and read(marker)['commit']!=commit:
            raise ValueError(f'Code HEAD changed after snapshot: {name}')
        if git('status', '--porcelain', '--untracked-files=no'):
            raise ValueError(f'Tracked source modifications in {name}')
        for suffix, command in [('tar', ['archive', '--format=tar', f'--output={folder/name}.tar', 'HEAD']),
                                ('bundle', ['bundle', 'create', str(folder/(name+'.bundle')), '--all'])]:
            dest = folder / (name + '.' + suffix)
            if not dest.exists():
                subprocess.run(['git', '-C', p, *command], check=True, capture_output=True)
        record=dict(name=name, path=p, commit=commit, branch=git('branch', '--show-current'),
                    refs=git('show-ref'), archive='code/'+name+'.tar', bundle='code/'+name+'.bundle')
        marker.write_text(json.dumps(record,indent=2)+'\n')
        result.append(record)
    return result


def prepare(cfg, u, s3):
    state = Path(cfg['state_root']); plan_path = state / 'manifest.json'
    if plan_path.exists():
        plan = read(plan_path)
        if plan['config_sha256'] != digest(cfg):
            raise ValueError('Archive configuration changed; preserve existing archive')
        return plan
    prior = cfg['existing_archive']
    obj = s3.get_object(Bucket=prior['bucket'], Key=prior['prefix']+'/_SUCCESS.json')
    body = obj['Body'].read(); obj['Body'].close(); receipt = json.loads(body)
    for k in ['plan_sha256', 'verified_samples', 'verified_shards']:
        if receipt[k] != prior[k]:
            raise ValueError('Previous MATH/MMLU archive receipt mismatch')
    u.save_json(state/'previous_math_mmlu_SUCCESS.json', receipt)
    code = code_snapshots(cfg, state)
    roots = [*cfg['roots'], dict(path=str(state/'code'), name='code', exclude_top=[])]
    objects = []; source_total = 0; files_total = 0; links = []
    for root in roots:
        u.save_json(state/'status.json', dict(stage='inventory', root=root['name'], at=u.now(), pid=os.getpid()))
        entries = collect_entries(root['path'], root['name'], root.get('exclude_top', []))
        links.extend(dict(name=e['name'], target=e['target']) for e in entries if e['kind']=='symlink')
        for i, batch in enumerate(chunks(entries, cfg['chunk_bytes'])):
            name = f'{root["name"]}-{i:05d}'
            path = state/'inventories'/(name+'.json'); u.save_json(path, batch)
            size = sum(e['bytes'] for e in batch)
            objects.append(dict(name=name, inventory=str(path), inventory_sha256=sha(path),
                                files=len(batch), source_bytes=size))
            source_total += size; files_total += len(batch)
    plan = dict(created_at=u.now(), config_sha256=digest(cfg), config={k:v for k,v in cfg.items() if k!='credentials_file'},
                source_bytes=source_total, files=files_total, objects=objects, repositories=code, symlinks=links,
                existing_math_mmlu=prior, reliability_complete=read(Path(cfg['reliability_root'])/'COMPLETE.json'),
                integrity='All archived file SHA256 values computed during streaming; each multipart Content-MD5 and returned ETag, full object composite ETag and length, and three nonempty member Range GET SHA256 checks per archive. No second full archive download claimed.',
                scope='All files in listed experiment/data roots, including unsuccessful attempts as historical artifacts. Completed raw MATH/MMLU are referenced at their pre-existing verified R2 prefix, not reuploaded. Symlinks stored without dereferencing; inspect link targets before restore. Source data retained. No claim every historical experiment passed or every response received semantic review.')
    u.save_json(plan_path, plan)
    return plan


def upload_one(rec, cfg, u, s3):
    state = Path(cfg['state_root']); name = rec['name']; key = cfg['prefix']+'/archives/'+name+'.tar'
    inventory = Path(rec['inventory'])
    if sha(inventory) != rec['inventory_sha256']:
        raise ValueError('Frozen inventory changed')
    rp = state/'receipts'/(name+'.json'); pp = state/'parts'/(name+'.json'); ip=state/'indices'/(name+'.json')
    remote = u.head(s3, cfg['bucket'], key)
    if rp.exists():
        receipt = read(rp)
        if not remote or remote['ContentLength'] != receipt['archive']['bytes'] or remote['ETag'].strip('"') != receipt['archive']['etag']:
            raise ValueError('Verified R2 object missing or changed')
        return receipt
    if remote:
        part = read(pp)
        if remote.get('Metadata', {}).get('source-copy-receipt-sha256') != rec['inventory_sha256'] or not ip.exists():
            raise ValueError('Existing object lacks trusted completion state')
        if part.get('archive_bytes') != remote['ContentLength'] or part.get('expected_etag') != remote['ETag'].strip('"'):
            raise ValueError('Uncertain upload completion failed recovery checks')
        archive = dict(bytes=part['archive_bytes'], sha256=part['archive_sha256'], etag=part['expected_etag'], parts=part['parts_total'])
        index=read(ip)
    else:
        sink = u.MultipartSink(s3, cfg, key, pp, rec['inventory_sha256'])
        try:
            index = stream(read(inventory), sink, u)
            u.save_json(ip, index); archive = sink.finish()
        except BaseException:
            sink.abandon(); raise
    regular = [r for r in index if r['kind']=='file' and r['bytes']>0]
    checks = []
    for i in sorted(set([0, len(regular)//2, len(regular)-1])) if regular else []:
        r = regular[i]
        # Bound range verification for exceptionally large single array members.
        if r['bytes'] > 64*1024**2:
            h=hashlib.sha256(); length=0
            obj=s3.get_object(Bucket=cfg['bucket'], Key=key, Range=f"bytes={r['offset']}-{r['offset']+r['bytes']-1}")
            try:
                for b in iter(lambda:obj['Body'].read(4*1024**2), b''):h.update(b);length+=len(b)
            finally:obj['Body'].close()
            remote_sha=h.hexdigest()
        else:
            obj=s3.get_object(Bucket=cfg['bucket'], Key=key, Range=f"bytes={r['offset']}-{r['offset']+r['bytes']-1}")
            data=obj['Body'].read();obj['Body'].close();length=len(data);remote_sha=hashlib.sha256(data).hexdigest()
        if remote_sha!=r['sha256'] or length!=r['bytes']:raise ValueError('Remote member range hash failed')
        checks.append(r['name'])
    u.put_bytes(s3,cfg,cfg['prefix']+'/indices/'+name+'.json',ip.read_bytes(),'application/json')
    receipt=dict(name=name,bucket=cfg['bucket'],key=key,archive=archive,files=len(index),
                 inventory_sha256=rec['inventory_sha256'],range_checks=checks,verified_at=u.now())
    receipt_key=cfg['prefix']+'/receipts/'+name+'.json'
    if u.head(s3,cfg['bucket'],receipt_key):
        obj=s3.get_object(Bucket=cfg['bucket'],Key=receipt_key);old=json.loads(obj['Body'].read());obj['Body'].close()
        if old['archive']!=archive or old['inventory_sha256']!=rec['inventory_sha256']:raise ValueError('Cloud receipt mismatch')
        receipt=old
    else:u.put_bytes(s3,cfg,receipt_key,(json.dumps(receipt,indent=2)+'\n').encode(),'application/json')
    u.save_json(rp,receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--limit',type=int)
    a=p.parse_args();cfg=read(a.config);u=load_transport(cfg);state=Path(cfg['state_root']);state.mkdir(exist_ok=True,parents=True)
    with u.lock(state/'ARCHIVE.lock'):
        try:
            while not gate(cfg):
                u.save_json(state/'status.json',dict(stage='waiting_for_reliability',at=u.now(),pid=os.getpid()))
                time.sleep(20)
            s3=u.client(cfg);plan=prepare(cfg,u,s3)
            u.put_bytes(s3,cfg,cfg['prefix']+'/manifest.json',(state/'manifest.json').read_bytes(),'application/json')
            u.put_bytes(s3,cfg,cfg['prefix']+'/previous_math_mmlu_SUCCESS.json',(state/'previous_math_mmlu_SUCCESS.json').read_bytes(),'application/json')
            readme=('HSS/OpenAct archive, 2026-09-26.\nDownload manifest.json and all archives/*.tar; verify per-archive SHA256 in receipts/.\n'
                    'Extract archives into one EMPTY directory, preserving hss/, openact-runs/, hss-input-cache/, code/.\n'
                    'For selected member access use indices/ byte offsets and Range GET; verify SHA256.\n'
                    'Inspect symlink targets in the manifest before extraction; no symlink target content is implied.\n'
                    'Clone code/*.bundle or extract code/*.tar. Read frozen experiment protocols for environment and data revision.\n'
                    'MATH/MMLU originals are already at s3://autoact-data/openact/math-mmlu-full-20260916/.\n'
                    '_SUCCESS.json exists only after every planned archive is verified. Local/NFS originals remain intact.\n')
            u.put_bytes(s3,cfg,cfg['prefix']+'/README.txt',readme.encode(),'text/plain')
            tasks=plan['objects'][:a.limit] if a.limit else plan['objects'];errors=[]
            def one(rec):
                for attempt in range(cfg['retries']):
                    try:return upload_one(rec,cfg,u,s3)
                    except (ValueError,PermissionError):raise
                    except Exception:
                        if attempt+1==cfg['retries']:raise
                        time.sleep(min(60,5*2**attempt))
            def status(stage):
                rs=[read(x) for x in (state/'receipts').glob('*.json')]
                result=dict(stage=stage,at=u.now(),pid=os.getpid(),verified_archives=len(rs),total_archives=len(plan['objects']),
                            verified_archive_bytes=sum(x['archive']['bytes'] for x in rs),source_bytes=plan['source_bytes'],
                            verified_files=sum(x['files'] for x in rs),total_files=plan['files'],errors=errors)
                u.save_json(state/'status.json',result);return result
            with ThreadPoolExecutor(cfg['workers']) as pool:
                futures={pool.submit(one,r):r for r in tasks}
                status('uploading')
                for f in as_completed(futures):
                    try:f.result()
                    except Exception as exc:
                        errors.append(dict(archive=futures[f]['name'],error_type=type(exc).__name__))
                    status('uploading')
            if errors:raise RuntimeError('Archive transfer errors; original data and multipart state preserved')
            if a.limit:status('pilot_complete');return
            final=status('verifying_completion')
            if final['verified_archives']!=len(plan['objects']) or final['verified_files']!=plan['files']:raise ValueError('Coverage mismatch')
            # Recheck frozen source sizes/mtimes and cloud heads before committing the collection marker.
            for rec in plan['objects']:
                for e in read(rec['inventory']):verify_stat(e)
                upload_one(rec,cfg,u,s3)
            final.update(stage='complete',bucket=cfg['bucket'],prefix=cfg['prefix'],manifest_sha256=sha(state/'manifest.json'))
            success_key=cfg['prefix']+'/_SUCCESS.json'
            if u.head(s3,cfg['bucket'],success_key):
                obj=s3.get_object(Bucket=cfg['bucket'],Key=success_key);old=json.loads(obj['Body'].read());obj['Body'].close()
                if old['manifest_sha256']!=final['manifest_sha256']:raise ValueError('Conflicting success marker')
                final=old
            else:u.put_bytes(s3,cfg,success_key,(json.dumps(final,indent=2)+'\n').encode(),'application/json')
            u.save_json(state/'_SUCCESS.json',final);u.save_json(state/'status.json',final)
        except BaseException as exc:
            u.save_json(state/'failure.json',dict(at=u.now(),pid=os.getpid(),error_type=type(exc).__name__,message=str(exc)[:500]))
            raise


if __name__=='__main__':main()
