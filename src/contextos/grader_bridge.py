"""Runs inside the pinned grader's own Python. Usage: grader_bridge.py OP ROOT INPUT OUTPUT"""

import json
import sys
from pathlib import Path

operation, root, source, output = sys.argv[1:5]
sys.path.insert(0, root)
from swebench.harness import constants, grading, test_spec  # noqa: E402

payload = json.loads(Path(source).read_text())
spec = test_spec.make_test_spec(payload["row"])
if operation == "script":
    result = {"script": spec.eval_script, "marker": constants.APPLY_PATCH_PASS}
else:
    prediction = {
        "instance_id": payload["row"]["instance_id"],
        "model_name_or_path": "contextos",
        "model_patch": payload["patch"],
    }
    result = grading.get_eval_report(spec, prediction, payload["log"], include_tests_status=True)
Path(output).write_text(json.dumps(result))
