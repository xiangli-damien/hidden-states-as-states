import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest
from scripts.archive_research_r2 import collect_entries, chunks, stream, gate
from scripts.archive_research_r2 import read
from scripts.archive_research_r2 import PaxMultipartClient


class HashReader:
    def __init__(self, file):
        self.file=file;self.count=0;self.h=hashlib.sha256()
    def read(self,n=-1):
        b=self.file.read(n);self.count+=len(b);self.h.update(b);return b


def test_tar_range_offsets_long_names_and_symlinks(tmp_path):
    source=tmp_path/'source';source.mkdir()
    nested=source/('long-'*22);nested.mkdir()
    (nested/'data.bin').write_bytes(bytes(range(256))*31)
    (source/'empty').write_bytes(b'')
    (source/'link').symlink_to('empty')
    entries=collect_entries(source,'hss')
    sink=io.BytesIO();index=stream(entries,sink,SimpleNamespace(HashReader=HashReader))
    data=sink.getvalue()
    for row in index:
        if row['kind']=='file':
            extracted=data[row['offset']:row['offset']+row['bytes']]
            assert hashlib.sha256(extracted).hexdigest()==row['sha256']
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        assert tar.getmember('hss/link').issym()
        assert tar.extractfile('hss/'+nested.name+'/data.bin').read()==(nested/'data.bin').read_bytes()
    second=io.BytesIO();stream(entries,second,SimpleNamespace(HashReader=HashReader))
    assert data==second.getvalue()


def test_freeze_rejects_mutation_and_private_files(tmp_path):
    (tmp_path/'a').write_bytes(b'old');entries=collect_entries(tmp_path,'x')
    (tmp_path/'a').write_bytes(b'new-longer')
    with pytest.raises(ValueError,match='changed'):
        stream(entries,io.BytesIO(),SimpleNamespace(HashReader=HashReader))
    (tmp_path/'secret.pem').write_text('fixture-not-a-key')
    with pytest.raises(ValueError,match='Private-file'):
        collect_entries(tmp_path,'x')


def test_chunks_and_existing_archive_exclusion(tmp_path):
    (tmp_path/'already').mkdir();(tmp_path/'already'/'data').write_text('skip')
    (tmp_path/'keep').write_text('abc')
    rows=collect_entries(tmp_path,'x',['already']);assert len(rows)==1
    assert rows[0]['name']=='x/keep'
    assert [len(r) for r in chunks([dict(bytes=3),dict(bytes=5),dict(bytes=8)],7)]==[1,1,1]


def test_original_publisher_hash_required(tmp_path):
    (tmp_path/'tensor').write_bytes(b'abc')
    receipt={'files':{'tensor':{'bytes':3,'sha256':hashlib.sha256(b'xyz').hexdigest()}}}
    (tmp_path/'_COPY_VERIFIED.json').write_text(json.dumps(receipt))
    entries=collect_entries(tmp_path,'x')
    with pytest.raises(ValueError,match='publisher SHA256'):
        stream(entries,io.BytesIO(),SimpleNamespace(HashReader=HashReader))


def test_gate_requires_valid_complete_audit(tmp_path):
    def put(name,value):(tmp_path/name).write_text(json.dumps(value))
    put('protocol.json',{'scope':'frozen'})
    cfg={'reliability_root':str(tmp_path),'reliability_protocol_sha256':hashlib.sha256((tmp_path/'protocol.json').read_bytes()).hexdigest()}
    put('job_status.json',{'status':'failed'})
    with pytest.raises(RuntimeError):gate(cfg)
    put('job_status.json',{'status':'complete','pid':None})
    put('audit.json',{'passed':True,'selections':5278,'fixed_refits':377,'data_rows':5000})
    put('COMPLETE.json',{'protocol_sha256':cfg['reliability_protocol_sha256'],'audit_sha256':hashlib.sha256((tmp_path/'audit.json').read_bytes()).hexdigest()})
    assert gate(cfg)
    put('audit.json',{'passed':False})
    with pytest.raises(ValueError):gate(cfg)


def test_nfs_status_read_retries_stale_handle(tmp_path,monkeypatch):
    import errno
    p=tmp_path/'status.json';p.write_text('{"status":"running"}')
    original=Path.read_text;calls=[]
    def flaky(self,*args,**kwargs):
        calls.append(1)
        if len(calls)==1:raise OSError(errno.ESTALE,'stale')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',flaky)
    assert read(p)['status']=='running'
    assert len(calls)==2


def test_multipart_declares_actual_pax_format():
    client=SimpleNamespace(create_multipart_upload=lambda **kw:kw)
    result=PaxMultipartClient(client).create_multipart_upload(Bucket='fixture',Metadata={'archive-format':'ustar-v1','source':'hash'})
    assert result['Metadata']=={'archive-format':'pax-v1','source':'hash'}
