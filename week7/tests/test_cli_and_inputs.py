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


def test_study_new_output_preserves_identical_existing_case(tmp_path):
    from inpainting import TEST_IDS,VALIDATION_IDS
    source=tmp_path/'images.npz'
    np.savez(source,test=np.zeros((10,32,32),np.float32),validation=np.zeros((10,32,32),np.float32),
             test_ids=TEST_IDS,validation_ids=VALIDATION_IDS,labels=np.arange(10))
    script=Path(__file__).resolve().parents[1]/'study.py'
    original_case=None
    for name in ['first-study','second-study']:
        result=subprocess.run([sys.executable,str(script),'--input',str(source),'--out',str(tmp_path/name)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        if original_case is None:original_case=(tmp_path/'case.npz').read_bytes()
        else:assert original_case==(tmp_path/'case.npz').read_bytes()
    with np.load(tmp_path/'case.npz') as f:
        np.testing.assert_array_equal(f['observations'],np.zeros((32,32)))
    # A different existing case must remain protected.
    np.savez(tmp_path/'case.npz',observations=np.ones((32,32)),mask=np.ones((32,32),bool))
    result=subprocess.run([sys.executable,str(script),'--input',str(source),'--out',str(tmp_path/'conflicting-study')],capture_output=True,text=True)
    assert result.returncode!=0 and 'different case' in result.stderr
