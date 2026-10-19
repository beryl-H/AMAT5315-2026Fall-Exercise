"""Integration checks for the declared executable and wrong-version image archives."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from inpainting import load_images


def test_wrong_course_archive_is_rejected(tmp_path):
    p=tmp_path/'images.npz'
    np.savez(p,test=np.zeros((4,3,32,32),np.float32),validation=np.zeros((4,3,32,32),np.float32))
    with pytest.raises(ValueError,match='local MNIST lecture'):load_images(p)


def test_real_dct_cli_contract_and_no_truth_input(tmp_path):
    source=tmp_path/'case.npz';out=tmp_path/'result'
    y=np.zeros((32,32),np.float32);m=np.zeros((32,32),bool)
    y[0,:]=.7;m[0,:]=True
    np.savez(source,observations=y,mask=m)
    result=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'inpaint'),
                           '--method','dct','--input',str(source),'--k','20','--iterations','5','--out',str(out)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.splitlines()[0]=='observed_max_error\tmissing_pixels'
    with np.load(out/'reconstruction.npz') as z:
        assert z.files==['reconstruction'];x=z['reconstruction']
    assert x.shape==(32,32) and x.dtype==np.float64
    np.testing.assert_array_equal(x[m],y[m])
    meta=json.loads((out/'run.json').read_text())
    assert meta==dict(method='dct',input=str(source),n=32,k=20,iterations=5)


def test_cli_rejects_missing_dct_parameter(tmp_path):
    source=tmp_path/'case.npz'
    np.savez(source,observations=np.zeros((32,32),np.float32),mask=np.zeros((32,32),bool))
    result=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'inpaint'),
                           '--method','dct','--input',str(source),'--iterations','5','--out',str(tmp_path/'out')],capture_output=True,text=True)
    assert result.returncode==2 and 'requires --k and --iterations' in result.stderr
    assert not (tmp_path/'out').exists()
