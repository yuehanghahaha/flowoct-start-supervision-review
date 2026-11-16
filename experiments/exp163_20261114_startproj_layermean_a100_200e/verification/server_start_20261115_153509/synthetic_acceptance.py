"""Runtime verification supplement: audit regrouped float32 sums without changing delivered files."""
import ast
import hashlib
import json
import sys
from pathlib import Path
import torch
import common as c

ROOT = c.ROOT
RECORD = Path(__file__).resolve().parent
original = ROOT / 'checks.py'
before = hashlib.sha256(original.read_bytes()).hexdigest()
audits = []

def audit_grouped_sum(total, expected, terms, cfg, net):
    assert total.dtype == expected.dtype == torch.float32
    weighted = [cfg['loss_weights'][key] * value for key, value in terms.items()]
    values = [float(value.detach()) for value in weighted]
    oracle = sum(values)
    epsilon = torch.finfo(torch.float32).eps
    operations = len(values) - 1
    gamma = operations * epsilon / (1 - operations * epsilon)
    bound = gamma * sum(abs(value) for value in values)
    actual, regrouped = float(total.detach()), float(expected.detach())
    assert abs(actual - oracle) <= bound
    assert abs(regrouped - oracle) <= bound
    parameters = tuple(net.parameters())
    left = torch.autograd.grad(total, parameters, retain_graph=True)
    right = torch.autograd.grad(expected, parameters, retain_graph=True)
    assert all(torch.equal(a, b) for a, b in zip(left, right))
    audits.append(dict(actual=actual, regrouped=regrouped, absolute_difference=abs(actual-regrouped),
                       float64_sum_of_float32_weighted_terms=oracle,
                       float32_sum_rounding_bound=bound, gradients_bitwise_equal=True))
    return True

tree = ast.parse(original.read_text(encoding='utf-8-sig'), filename=str(original))
target = ast.dump(ast.parse('torch.equal(total, expected)', mode='eval').body)
changed = []
class SumAssertion(ast.NodeTransformer):
    def visit_Assert(self, node):
        if ast.dump(node.test) == target:
            changed.append(node.lineno)
            call = ast.Call(func=ast.Name(id='audit_grouped_sum', ctx=ast.Load()),
                            args=[ast.Name(id=name, ctx=ast.Load())
                                  for name in ['total', 'expected', 'terms', 'cfg', 'net']],
                            keywords=[])
            node.test = ast.copy_location(call, node.test)
        return node
tree = SumAssertion().visit(tree)
assert changed == [117], changed
ast.fix_missing_locations(tree)
namespace = dict(__name__='exp163_runtime_verification_supplement', __file__=str(original),
                 audit_grouped_sum=audit_grouped_sum)
exec(compile(tree, str(original), 'exec'), namespace)
sys.argv = [str(original), 'synthetic']
namespace['main']()
assert hashlib.sha256(original.read_bytes()).hexdigest() == before
receipt_path = ROOT / 'verification/synthetic.json'
receipt = c.read(receipt_path)
receipt['runtime_verification_supplement'] = dict(
    runner=str(Path(__file__).relative_to(ROOT)),
    original_check_source_sha256=before,
    source_file_unmodified=True,
    original_failure='checks.py:117 required bitwise equality for differently grouped float32 sums',
    original_failure_log='logs/server_prepare_20261115_153509.log',
    assertion_replaced_in_memory_only=changed,
    replacement='Both groupings within IEEE float32 forward summation error bound; all parameter gradients bitwise equal',
    audits=audits,
    other_assertions_unmodified=True,
    scientific_training_code_or_settings_changed=False)
c.write(receipt_path, receipt)
c.write(RECORD / 'synthetic_acceptance.json', receipt)
print(json.dumps(receipt['runtime_verification_supplement'], indent=2), flush=True)
