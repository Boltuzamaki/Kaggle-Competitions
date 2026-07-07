import sys; sys.path.insert(0, 'data/neurogolf_utils')
import onnx, onnxruntime, numpy as np, json, copy
import neurogolf_utils as ngu

m = onnx.load('repairs/task134.onnx')
san = ngu.sanitize_model(copy.deepcopy(m))
so = onnxruntime.SessionOptions()
so.log_severity_level = 3
so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
s = onnxruntime.InferenceSession(san.SerializeToString(), so)

with open('data/task134.json') as f:
    d = json.load(f)

all_ex = d['train'] + d['test'] + d['arc-gen']
nfail = 0
for ex_id, ex in enumerate(all_ex):
    b = ngu.convert_to_numpy(ex)
    if not b:
        continue
    try:
        out = ngu.run_network(s, b['input'])
        if not np.array_equal(out, b['output']):
            nfail += 1
            if nfail <= 5:
                pred = np.argmax(out[0], axis=0)
                expected = np.argmax(b['output'][0], axis=0)
                print("FAIL ex=", ex_id)
                print("Predicted (3x3):")
                print(pred[:3, :3])
                print("Expected (3x3):")
                print(expected[:3, :3])
                # Check if padding area has non-zero
                if np.any(pred[3:, :] != 0) or np.any(pred[:, 3:] != 0):
                    print("  >> Has values outside 3x3 region!")
    except Exception as e:
        nfail += 1
        if nfail <= 5:
            print("ERROR ex=", ex_id, str(e)[:200])

print("Total failures:", nfail, "out of", len(all_ex))
