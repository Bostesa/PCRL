"""Data-free exact-wire regression checks; no fitting or role loading."""
import unittest
import numpy as np
from incumbent_view.r4_views import package_from_arrays, prepare_three_views
from incumbent_view.r4_public_law import uniform_output_masses,row_log_mass


def fixture():
    income = np.array([[.45,.55],[.15,.85],[.40,.60],[.20,.80]], np.float64)
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
        masses=uniform_output_masses(identity["bank_r1"],identity["bank_r2"])
        views=prepare_three_views(att,inner,uniform_masses=masses)
        for view in views.values():
            self.assertIsNotNone(view.att.x_defense_aware)
            self.assertIsNotNone(view.inner.x_defense_aware)
        self.assertIn("public_metadata.canonical_json",views["r1"].att.wire_components)
        self.assertIn("pair_code_bit_products",views["pair"].schema["public_derived"])
        self.assertIn("pair_public_uniform_source_log_output_mass",views["pair"].schema["public_derived"])

    def test_inner_unknown_id_does_not_renumber(self):
        identity,a,i=fixture()
        views=prepare_three_views(package_from_arrays(a,identity=identity,role="ATTACK_FIT_NEW"),
                                  package_from_arrays(i,identity=identity,role="INNER_SELECTION"),
                                  uniform_masses=uniform_output_masses(identity["bank_r1"],identity["bank_r2"]))
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
        p=identity["bank_r1"][:,1]
        self.assertAlmostEqual(p[:2].mean(),p[2:].mean())
        self.assertFalse(set(p[:2]).intersection(p[2:]))
        self.assertTrue(np.all(np.argmax(identity["bank_r1"],axis=1)==1))
        # Actual sampled support separates SEX despite equal means/decisions.
        self.assertEqual(len(set(p)),4)

    def test_uniform_policy_feature_does_not_read_other_local_code(self):
        identity,_,_=fixture()
        masses=uniform_output_masses(identity["bank_r1"],identity["bank_r2"])
        self.assertAlmostEqual(float(masses["pair"].sum()),1.)
        self.assertAlmostEqual(float(masses["r1"].sum()),1.)
        self.assertTrue(np.allclose(masses["pair"],1/8))
        self.assertEqual(row_log_mass(masses,"r1",np.array([0],np.int64)).shape,(1,1))
        with self.assertRaises(ValueError):
            row_log_mass(masses,"r1",np.array([0],np.int64),np.array([1],np.int64))

    def test_public_law_feature_enumerates_fixed_bank_without_labels(self):
        from incumbent_view.r2_space import Space
        identity,_,_=fixture();p1,p2=identity["bank_r1"],identity["bank_r2"]
        a1=np.repeat(np.arange(4,dtype=np.int64)[:,None],4,axis=1)
        a1[:,1]=[1,0,0,0]
        a2=np.repeat(np.arange(2,dtype=np.int64)[:,None],4,axis=1)
        v1=np.zeros((4,4),bool);v1[:,:2]=True
        v2=np.zeros((2,4),bool);v2[:,0]=True
        class FixedPublicPolicy:
            space=Space(p1,p2,a1,a2,v1,v2)
            def law_prepared(self,prepared):
                q=np.zeros((len(prepared["code_r1"]),16),np.float64)
                q[:,0]=.5;q[:,4]=.5
                return q
        masses=uniform_output_masses(p1,p2,FixedPublicPolicy())
        np.testing.assert_allclose(masses["r1"],[.5,.25,.125,.125],rtol=0,atol=1e-15)
        np.testing.assert_allclose(masses["r2"],[.5,.5],rtol=0,atol=1e-15)
        self.assertEqual(masses["source_prior"],"uniform_over_all_original_P0_global_code_pairs")


if __name__=="__main__": unittest.main()
