"""Frozen sampler laws, distinct from method/utility synthetic laws."""
import unittest
import numpy as np
from conditional_release.deployment import sample_tuple,CachedRelease,identity_law
from conditional_release.tests.test_cached_person_wire import FakeService
from pathlib import Path
import tempfile

A=np.array([0,0,1,1],np.int64)
B=np.array([0,1,0,1],np.int64)

class Tests(unittest.TestCase):
    def frequencies(self,q,kind,n1,n2):
        counts=np.zeros((2,2),np.int64)
        for u1 in (np.arange(n1)+.5)/n1:
            for u2 in (np.arange(n2)+.5)/n2:
                a,b=sample_tuple(A,B,q,.123,float(u1),float(u2),kind);counts[a,b]+=1
        return counts/(n1*n2)
    def test_product_exact_finite_coins(self):
        q=np.array([.1,.15,.3,.45],np.float64)
        np.testing.assert_allclose(self.frequencies(q,'product',20,20),q.reshape(2,2),rtol=0,atol=1e-14)
    def test_sequential_both_orders_exact(self):
        q=np.array([.1,.2,.3,.4],np.float64)
        for kind,n1,n2 in (('seq12',20,210),('seq21',60,20)):
            np.testing.assert_allclose(self.frequencies(q,kind,n1,n2),q.reshape(2,2),rtol=0,atol=1e-14)
    def test_joint_is_not_product(self):
        q=np.array([.5,0,0,.5],np.float64)
        self.assertEqual(sample_tuple(A,B,q,.2,.2,.8,'joint'),(0,0))
        with self.assertRaises(ValueError):sample_tuple(A,B,q,.2,.2,.8,'product')
    def test_duplicate_emitted_actions_combine(self):
        a=np.array([0,0,1],np.int64);b=np.array([0,0,0],np.int64);q=np.array([.2,.3,.5],np.float64)
        self.assertEqual(sample_tuple(a,b,q,.1,.2,.8,'product'),(0,0))
        self.assertEqual(sample_tuple(a,b,q,.1,.8,.2,'product'),(1,0))
    def test_domains_and_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            api=CachedRelease(FakeService(),identity_law,policy_sha256='fixture',cache_path=Path(d)/'private',secret=b'a'*32,sampling_kind='product')
            self.assertNotEqual(api.uniform('same',1),api.uniform('same',2));api.close()
        self.assertEqual(sample_tuple(A,B,np.array([0,0,1,0],np.float64),.3,.1,.7,'deterministic'),(1,0))
        with self.assertRaises(ValueError):sample_tuple(A,B,np.array([.5,0,0,.5],np.float64),.3,.1,.7,'deterministic')

if __name__=='__main__':unittest.main()
