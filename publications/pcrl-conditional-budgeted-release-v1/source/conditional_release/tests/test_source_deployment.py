import tempfile,unittest
from pathlib import Path
import numpy as np
from conditional_release.source import admitted_path,read_exact,sha_file,load_allowed_role,group_folds
from conditional_release.deployment import validate_law,identity_law,CachedRelease

class Tests(unittest.TestCase):
    def test_barred_before_io(self):
        for name in ('HEAD_VALIDATION/a.npz','all_roles/a.npz','teacher/release.npz','rel__s0__U/release.npz','OSF_DEVELOPMENT_ASSESSMENT/a.npz'):
            with self.assertRaises(PermissionError):sha_file(name)
    def test_inner_sex_before_manifest_access(self):
        with self.assertRaises(PermissionError):load_allowed_role({},0,'INNER_SELECTION',{'sex'})
    def test_unrequested_archive_field_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.npz';np.savez(p,allowed=np.ones(1),forbidden=np.ones(1))
            with self.assertRaises(PermissionError):read_exact({'path':str(p),'sha256':sha_file(p)},{'allowed'})
    def test_duplicate_fold(self):
        f=group_folds(np.array([1,1,2],dtype=np.int64));self.assertEqual(f[0],f[1])
    def test_law_decisions_and_boundary(self):
        bank={'recipients':{'1':{'probability':[[.8,.2],[.6,.4],[.2,.8]]},'2':{'probability':[[.8,.2],[.6,.4],[.2,.8]]}}}
        c=np.array([0],np.int64);a=np.array([[0,1]],np.int64);q=np.array([[.2,.8]],np.float64)
        validate_law(bank,c,c,a,a,q)
        with self.assertRaises(ValueError):validate_law(bank,c,c,np.array([[0,2]],np.int64),a,q)
        with self.assertRaises(ValueError):validate_law(bank,c,c,a,a,q*.9)
    def test_person_coins_stable_and_separate(self):
        class Service:
            seed=0;bank={'source_binding_sha256':'test'}
        with tempfile.TemporaryDirectory() as d:
            obj=CachedRelease(Service(),identity_law,policy_sha256='abc',cache_path=Path(d)/'cache',secret=b'x'*32)
            self.assertEqual(obj.uniform('person1'),obj.uniform('person1'))
            self.assertNotEqual(obj.uniform('person1'),obj.uniform('person2'))
            self.assertGreaterEqual(obj.uniform('person1'),0);self.assertLess(obj.uniform('person1'),1)
            obj.close()

if __name__=='__main__':unittest.main()
