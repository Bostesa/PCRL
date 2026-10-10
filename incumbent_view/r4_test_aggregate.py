"""Synthetic accounting checks; execute through the shared queue."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from incumbent_view.r4_aggregate import aggregate,fixed_slots


class AggregateChecks(unittest.TestCase):
    def test_forty_slots_and_three_views_even_without_result(self):
        slots=fixed_slots()
        self.assertEqual(len(slots),40)
        self.assertEqual(len({x["unit_id"] for x in slots}),40)
        with tempfile.TemporaryDirectory() as name:
            repo=Path(name);root=repo/"results/pcrl_incumbent_view_risk_v1/r4"
            root.mkdir(parents=True)
            p=root/"prepared.json";e=root/"effective.json"
            p.write_text(json.dumps({"schema":"pcrl-incumbent-r4-prepared-index-v1",
                                     "all_slots":[dict(s,status="POINT_INELIGIBLE") for s in slots]}))
            e.write_text(json.dumps({"schema":"pcrl-incumbent-r4-effective-attack-map-v1",
                                     "slots":{s["unit_id"]:None for s in slots}}))
            pin=lambda path:{"path":str(path.relative_to(repo)),
                             "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
            result,rows=aggregate(repo,pin(p),pin(e))
            self.assertEqual((result["fixed_slots"],result["completed_instances"],len(rows)),(40,0,120))
            self.assertTrue(all(row["primary_envelope_auc"]=="" for row in rows))

    def test_missing_prospective_slot_is_rejected(self):
        slots=fixed_slots()
        with tempfile.TemporaryDirectory() as name:
            repo=Path(name);p=repo/"p.json";e=repo/"e.json"
            p.write_text(json.dumps({"schema":"pcrl-incumbent-r4-prepared-index-v1",
                                     "all_slots":[dict(s,status="POINT_INELIGIBLE") for s in slots]}))
            e.write_text(json.dumps({"schema":"pcrl-incumbent-r4-effective-attack-map-v1",
                                     "slots":{s["unit_id"]:None for s in slots[:-1]}}))
            pin=lambda path:{"path":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
            with self.assertRaises(PermissionError):aggregate(repo,pin(p),pin(e))


if __name__=="__main__":unittest.main()
