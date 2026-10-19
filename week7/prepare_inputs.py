"""Recreate the selected local lecture images from original official MNIST rows.

The live week7-images.npz has changed to a different task. It is not used here.
Existing official IDX archives are reused when available; otherwise download
the exact two original test archives. Never train or select new examples.
"""
import argparse,gzip,hashlib,json,struct,urllib.request
from pathlib import Path
import numpy as np
from inpainting import TEST_IDS,VALIDATION_IDS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('../week6/energy-based/inputs/mnist'))
    p.add_argument('--out',type=Path,default=Path('inputs/images.npz'))
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Use a fresh output path; input already exists')
    files=[('t10k-images-idx3-ubyte.gz','9fb629c4189551a2d022fa330f9573f3'),
           ('t10k-labels-idx1-ubyte.gz','ec29112dd5afa0611ce80d1b7f02629c')]
    raw=[];provenance=[]
    for name,md5 in files:
        source=a.source/name
        url='https://storage.googleapis.com/cvdf-datasets/mnist/'+name
        if source.exists():blob=source.read_bytes();location=str(source)
        else:
            source=a.out.parent/'source'/name;source.parent.mkdir(parents=True,exist_ok=True)
            blob=urllib.request.urlopen(url,timeout=60).read();source.write_bytes(blob);location=str(source)
        assert hashlib.md5(blob).hexdigest()==md5,'Official MNIST file digest mismatch'
        provenance.append(dict(url=url,local_source=location,bytes=len(blob),md5=md5,sha256=hashlib.sha256(blob).hexdigest()))
        raw.append(gzip.decompress(blob))
    assert struct.unpack('>IIII',raw[0][:16])==(2051,10000,28,28)
    assert struct.unpack('>II',raw[1][:8])==(2049,10000)
    images=np.frombuffer(raw[0],np.uint8,offset=16).reshape(10000,28,28)
    labels=np.frombuffer(raw[1],np.uint8,offset=8)
    np.testing.assert_array_equal(labels[TEST_IDS],np.arange(10))
    np.testing.assert_array_equal(labels[VALIDATION_IDS],np.arange(10))
    for d in range(10):
        np.testing.assert_array_equal(np.flatnonzero(labels==d)[:2],[TEST_IDS[d],VALIDATION_IDS[d]])
    def transform(ids):return np.pad(images[ids],((0,0),(2,2),(2,2))).astype(np.float32)/np.float32(127.5)-np.float32(1)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,test=transform(TEST_IDS),validation=transform(VALIDATION_IDS),
                        test_ids=TEST_IDS,validation_ids=VALIDATION_IDS,labels=np.arange(10))
    report=dict(source_archives=provenance,reason='Live course archive belongs to another exercise; reconstruct exact local-PDF rows.',
                test_ids=TEST_IDS.tolist(),validation_ids=VALIDATION_IDS.tolist(),
                labels=np.arange(10).tolist(),normalization='pad uint8 with zeros then float32 / 127.5 - 1',
                arrays_sha256={k:hashlib.sha256(transform(ids).tobytes()).hexdigest() for k,ids in [('test',TEST_IDS),('validation',VALIDATION_IDS)]},
                archive_sha256=hashlib.sha256(a.out.read_bytes()).hexdigest())
    Path('evidence').mkdir(exist_ok=True)
    Path('evidence/input-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
