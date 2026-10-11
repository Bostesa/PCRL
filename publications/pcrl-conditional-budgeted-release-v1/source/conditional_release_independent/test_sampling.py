"""Pure exact sampler checks; no optimizer, labels, fitted policy or data."""
import unittest
from .sampling import uniform,sampled_pair

class Sampling(unittest.TestCase):
    def test_joint_order_and_exact_cutoff(self):
        self.assertEqual(sampled_pair([2,0,2],[3,1,3],[.25,.5,.25],'joint',.25,0,0),(0,1))
        self.assertEqual(sampled_pair([2,0,2],[3,1,3],[.25,.5,.25],'joint',.99,0,0),(2,3))
    def test_product_two_coins(self):
        self.assertEqual(sampled_pair([0,0,1,1],[0,1,0,1],[.125,.375,.125,.375],'product',.01,.9,.1),(1,0))
    def test_sequential_conditions_actual_tuple(self):
        args=([0,0,1,1],[0,1,0,1],[.1,.4,.4,.1])
        self.assertEqual(sampled_pair(*args,'seq12',.01,.9,.5),(1,0))
        self.assertEqual(sampled_pair(*args,'seq21',.01,.5,.9),(0,1))
    def test_hmac_domains_and_grid(self):
        s=bytes(range(32));u=uniform(s,'joint','a'*64,'b'*64,0,71)
        self.assertEqual(u,uniform(s,'joint','a'*64,'b'*64,0,71))
        self.assertEqual(u*2**53,int(u*2**53))
        self.assertNotEqual(u,uniform(s,'product','a'*64,'b'*64,0,71))
        self.assertNotEqual(uniform(s,'product','a'*64,'b'*64,0,71,1),uniform(s,'product','a'*64,'b'*64,0,71,2))
    def test_invalid_product_refused(self):
        with self.assertRaises(ValueError):sampled_pair([0,1],[0,1],[.5,.5],'product',0,.4,.7)

if __name__=='__main__':unittest.main()
