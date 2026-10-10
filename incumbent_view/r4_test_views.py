"""Data-free exact-wire regression checks; no fitting or role loading."""
import unittest
import numpy as np
from incumbent_view.r4_views import package_from_arrays, prepare_three_views


def fixture():
    income = np.array([[.55,.45],[.85,.15],[.60,.40],[.80,.20]], np.float64)
    occupation = np.array([[.5,.1,.1,.1,.1,.1],[.4,.12,.12,.12,.12,.12]], np.float64)
    identity = {"release_id":"exact-historical-release", "source_binding_sha256":"a"*64,
                "bank_r1":income,"bank_r2":occupation}
    def arrays(rows, codes):
        codes = np.array(codes,np.int64)
        c2 = np.zeros(len(codes),np.int64)
        return {"row_id":np.array(rows,np.int64),"unit":np.array(rows,np.int64),
                "code_r1":codes,"code_r2":c2,"prob_r1":income[codes],
                "prob_r2":occupation[c2],"decision_r1":np.argmax(income[codes],axis=1),
                "decision_r2":np.argmax(occupation[c2],axis=1)}
    return identity, arrays([1,2,3],[0,1,2]), arrays([4,5],[2,3])


class ExactWireTests(unittest.TestCase):
    def test_preserves_source_ids_metadata_and_da(self):
        identity,a,i=fixture()
        att=package_from_arrays(a,identity=identity,role="ATTACK_FIT_NEW")
        inner=package_from_arrays(i,identity=identity,role="INNER_SELECTION")
        self.assertEqual(att.wires1[0]["release_id"],identity["release_id"])
        self.assertEqual(att.wires1[2]["code_id"],2)
        self.assertEqual(att.wires1[0]["public_metadata"]["code_length_bits"],2)
        views=prepare_three_views(att,inner)
        for view in views.values():
            self.assertIsNotNone(view.att.x_defense_aware)
            self.assertIsNotNone(view.inner.x_defense_aware)
        self.assertIn("public_metadata.canonical_json",views["r1"].att.wire_components)
        self.assertIn("pair_code_bit_products",views["pair"].schema["public_derived"])

    def test_inner_unknown_id_does_not_renumber(self):
        identity,a,i=fixture()
        views=prepare_three_views(package_from_arrays(a,identity=identity,role="ATTACK_FIT_NEW"),
                                  package_from_arrays(i,identity=identity,role="INNER_SELECTION"))
        field=next(f for f in views["r1"].schema["fields"] if f["name"]=="code_id")
        self.assertEqual(field["att_categories"],["0","1","2"])
        self.assertEqual(views["r1"].inner.cell_key.tolist(),['["2"]','["3"]'])

    def test_byte_different_decoder_rejected(self):
        identity,a,_=fixture()
        a["prob_r1"][0,0]+=1e-15
        with self.assertRaises(ValueError): package_from_arrays(a,identity=identity,role="ATTACK_FIT_NEW")

    def test_duplicate_cache_violation_rejected(self):
        identity,a,_=fixture()
        a["unit"][1]=a["unit"][0]
        with self.assertRaises(ValueError): package_from_arrays(a,identity=identity,role="ATTACK_FIT_NEW")

    def test_unsent_mean_trap(self):
        identity,_,_=fixture()
        p=identity["bank_r1"][:,0]
        self.assertAlmostEqual(p[:2].mean(),p[2:].mean())
        self.assertFalse(set(p[:2]).intersection(p[2:]))
        self.assertTrue(np.all(np.argmax(identity["bank_r1"],axis=1)==0))
        # Actual sampled support separates SEX despite equal means/decisions.
        self.assertEqual(len(set(p)),4)


if __name__=="__main__": unittest.main()
