"""Synthetic cache contract, no scientific sources or labels."""
import tempfile,unittest
from pathlib import Path
import numpy as np
from conditional_release.deployment import CachedRelease

class FakeService:
    seed=0
    names=tuple('x'+str(i) for i in range(83))
    bank={'source_binding_sha256':'a'*64,'release_id':'fixture-release',
          'recipients':{'1':{'probability':[[.8,.2],[.6,.4]],'bank_sha256':'b'*64},
                        '2':{'probability':[[.8,.04,.04,.04,.04,.04],[.6,.08,.08,.08,.08,.08]],'bank_sha256':'c'*64}}}
    def route_raw(self,raw,names):
        x=np.asarray(raw)
        if tuple(names)!=self.names or x.ndim!=2 or x.shape[1]!=83 or not np.isfinite(x).all():raise ValueError('raw83 refusal')
        outputs=[]
        for r in ('1','2'):
            p=np.asarray(self.bank['recipients'][r]['probability']);c=np.zeros(len(x),dtype=np.int64)
            outputs.append((c,p[c],p[c].argmax(1)))
        return outputs,None

def law(c1,c2):
    a=np.tile(np.array([0,1],np.int64),(len(c1),1))
    return a,a,np.full((len(c1),2),.5,np.float64)

class Tests(unittest.TestCase):
    def test_duplicate_cached_draw_and_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            cache=Path(d)/'private.db';x=np.zeros((3,83),np.float32);keys=['one','one','two']
            obj=CachedRelease(FakeService(),law,policy_sha256='fixture',cache_path=cache,secret=b'z'*32)
            w=obj.release(x,FakeService.names,keys)
            self.assertEqual(w[0][0],w[0][1]);self.assertEqual(w[1][0],w[1][1]);obj.close()
            obj=CachedRelease(FakeService(),law,policy_sha256='fixture',cache_path=cache,secret=b'z'*32)
            self.assertEqual(w,obj.release(x,FakeService.names,keys))
            for view in w:
                for wire in view:
                    for forbidden in ('person_key','unit','row_id','uniform','fallback','input_code','action_position'):
                        self.assertNotIn(forbidden,wire);self.assertNotIn(forbidden,wire.get('public_metadata',{}))
            with self.assertRaises(ValueError):obj.release(np.ones((1,83),np.float32),FakeService.names,['one'])
            obj.close()
    def test_schema_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            obj=CachedRelease(FakeService(),law,policy_sha256='fixture',cache_path=Path(d)/'private.db',secret=b'z'*32)
            with self.assertRaises(ValueError):obj.release(np.zeros((1,83),np.float32),FakeService.names[::-1],['one'])
            with self.assertRaises(ValueError):obj.release(np.zeros((1,84),np.float32),FakeService.names,['one'])
            with self.assertRaises(ValueError):obj.release(np.zeros((1,83),np.float32),FakeService.names,[])
            obj.close()

if __name__=='__main__':unittest.main()
