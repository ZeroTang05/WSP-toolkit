import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from audit_search import audit
from delta_bank import fingerprint


class SearchAuditTests(unittest.TestCase):
    def fixture(self,p):
        def save(name,value):
            path=p/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
        files={'model.safetensors':'test-base-hash'}
        source=b'test-source';(p/'code').mkdir();(p/'code/module.py').write_bytes(source)
        contract=save('contracts/methods.json',{'test':'fixture'})
        payload=p/'delta/adapter.safetensors';payload.parent.mkdir();payload.write_bytes(b'file-binding fixture only')
        direction=save('delta/manifest.json',dict(base_id=fingerprint(files),tensor_sha256=hashlib.sha256(payload.read_bytes()).hexdigest()))
        protocol=save('protocol.json',dict(protocol_version='paper-grid-v2',queries=2,guard_n=2,guard_margin=.02,
            guard_mode='norm_ce',without_guard=False,code_sha256={'module.py':hashlib.sha256(source).hexdigest()},
            contract_spec_sha256=hashlib.sha256(contract.read_bytes()).hexdigest(),delta=str(payload.parent),
            delta_manifest_sha256=hashlib.sha256(direction.read_bytes()).hexdigest()))
        rows=[dict(index=i,coordinates=[float(i)]*4,domain_risk=1-i*.5,guard_ce_per_item=[1.,1.],
            guard_diagnostics={'norm_ce':[1.,1.]},guard_upper95=0.,guard_ce=1.,guard_margin=.02,feasible=True) for i in range(2)]
        (p/'trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        save('base_manifest.json',dict(files=files))
        save('selected_plan.json',dict(coordinates=[1.]*4,base_id=fingerprint(files),
            delta_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),selection_protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest()))
        save('complete.json',dict(queries=2,selected_index=1,coordinates=[1.]*4))

    def test_valid_trace_and_wrong_incumbent(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);self.fixture(p)
            self.assertEqual(audit(p)['selected_index'],1)
            path=p/'complete.json';value=json.loads(path.read_text());value['selected_index']=0
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):audit(p)

    def test_direction_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);self.fixture(p)
            (p/'delta/adapter.safetensors').write_bytes(b'different direction')
            with self.assertRaises(ValueError):audit(p)

    def budget_fixture(self,p):
        self.fixture(p)
        protocol=json.loads((p/'protocol.json').read_text())
        protocol.update(protocol_version='resident-budget-v1',budget_seconds=1.5)
        (p/'protocol.json').write_text(json.dumps(protocol))
        rows=[json.loads(x) for x in (p/'trace.jsonl').read_text().splitlines()]
        for i,row in enumerate(rows):row.update(active_seconds=float(i+1),within_budget=i==0)
        (p/'trace.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
        value=json.loads((p/'complete.json').read_text())
        value.update(selected_index=0,coordinates=[0.]*4,budget_seconds=1.5,eligible_queries=1,
            active_seconds=2.1,overrun_seconds=2.1-1.5,fallback_identity_without_completed_observation=False)
        (p/'complete.json').write_text(json.dumps(value))
        plan=json.loads((p/'selected_plan.json').read_text());plan.update(coordinates=[0.]*4,
            selection_protocol_sha256=hashlib.sha256((p/'protocol.json').read_bytes()).hexdigest())
        (p/'selected_plan.json').write_text(json.dumps(plan))

    def test_better_candidate_after_deadline_is_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);self.budget_fixture(p)
            self.assertEqual(audit(p)['selected_index'],0)
            rows=[json.loads(x) for x in (p/'trace.jsonl').read_text().splitlines()]
            rows[1]['within_budget']=True
            (p/'trace.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            with self.assertRaisesRegex(ValueError,'deadline eligibility'):audit(p)

    def test_no_candidate_within_budget_returns_unmodified_base(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);self.budget_fixture(p);(p/'trace.jsonl').write_text('')
            complete=json.loads((p/'complete.json').read_text())
            complete.update(queries=0,selected_index=None,eligible_queries=0,
                fallback_identity_without_completed_observation=True)
            (p/'complete.json').write_text(json.dumps(complete))
            self.assertTrue(audit(p)['identity_selected'])
